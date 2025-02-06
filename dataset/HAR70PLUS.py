import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, lfilter
import os
import re
from pathlib import Path


class HAR70PLUS():
    def __init__(self, train, validation, test, current_directory):
        self.train_participant = train
        self.validation_participant = validation
        self.test_participant = test

        self.training = None
        self.test = None
        self.validation = None

        self.training_cleaned = None
        self.test_cleaned = None
        self.validation_cleaned = None

        self.training_normalized = None
        self.test_normalized = None
        self.validation_normalized = None

        self.training_normalized_segmented = None
        self.test_normalized_segmented = None
        self.validation_normalized_segmented = None

        self.training_final = None
        self.validation_final = None
        self.test_final = None

        self.training_sensor_participant, self.validation_sensor_participant, self.test_sensor_participant = None, None, None

        self.period = 1 / 100 ## Sampling period

        self.PATH = current_directory

        self.headers_initial = ["timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"]
        self.headers_final = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "activityID"]

        self.dataset_name = 'HAR70PLUS'
        self.num_participants = 18
        self.mapping_labels = {1:1,3:2,4:3,5:4,6:5,7:6,8:7}

    def _mapping_participants(self, files):
        # Build the mapping from original participant id to new sequential id.
        pattern_subject = re.compile(r"(\d+).csv", re.IGNORECASE)
        orig_ids = []
        for file in files:
            match = pattern_subject.match(file)
            if match:
                orig_ids.append(int(match.group(1)))
        orig_ids = sorted(set(orig_ids))
        return {orig: new + 1 for new, orig in enumerate(orig_ids)}


    def get_datasets(self):
        """
            Processes the HAR70PLUS dataset.

            Each CSV file corresponds to one participant and contains the following columns:
                timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label

            The function does the following:
              - Reads each CSV file.
              - Drops the timestamp column.
              - Renames the "label" column to "activityID".
              - Uses a mapping function to reassign participant numbers to sequential IDs (starting at 0).

            Returns:
              processed_data: dict mapping new participant id (int) to a DataFrame containing:
                  back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, activityID
              user_mapping: dict mapping original participant id to new participant id.
            """

        training = {a: 0 for a in self.train_participant}
        test = {a: 0 for a in self.test_participant}
        validation = {a: 0 for a in self.validation_participant}

        # Define the base directory containing subject folders.
        dataset_folder = Path(self.PATH) / "datasets" / self.dataset_name / "normal"
        pattern_subject = re.compile(r"(\d+).csv", re.IGNORECASE)


        # Get list of CSV files in the folder.
        files = [f for f in os.listdir(dataset_folder) if pattern_subject.match(f)]


        assert len(files) == self.num_participants
        user_mapping = self._mapping_participants(files)

        processed_data = {}
        for file in files:
            file_path = dataset_folder / file
            match_subject = pattern_subject.match(file)
            if match_subject:
                new_id = user_mapping[int(match_subject.group(1))]
            else:
                print(f"Something happened in with file {file_path}")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            if list(df.columns) != self.headers_initial:
                print(f"Warning: File {file} does not have the expected columns.")
            # Drop the timestamp column.
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])
            df.columns = self.headers_final


            df['activityID'] = df['activityID'].map(self.mapping_labels)




            if new_id in training:
                training[new_id] = df
            elif new_id in validation:
                validation[new_id] = df
            elif new_id in test:
                test[new_id] = df
            else:
                print(f"Volunteer {new_id} in file {file} is not assigned to any split.")

        # Save the dictionaries to the instance variables.
        self.training = training
        self.validation = validation
        self.test = test


    def normalize(self):
        training_normalized = {a: 0 for a in self.training_cleaned.keys()}
        test_normalized = {a: 0 for a in self.test_cleaned.keys()}
        validation_normalized = {a: 0 for a in self.validation_cleaned.keys()}

        max = pd.DataFrame(np.zeros((1, len(self.headers_final))), columns=self.headers_final)
        min = pd.DataFrame(np.zeros((1, len(self.headers_final))), columns=self.headers_final)

        min_aux, max_aux = None, None

        # print(self.validation_cleaned)

        for a in training_normalized.keys():
            max_aux = self.training_cleaned[a].max(axis='rows')
            min_aux = self.training_cleaned[a].min(axis='rows')
            for indx, a in enumerate(max):
                if max.iloc[0, indx] < max_aux.iloc[indx]:
                    max.iloc[0, indx] = max_aux.iloc[indx]
                if min.iloc[0, indx] > min_aux.iloc[indx]:
                    min.iloc[0, indx] = min_aux.iloc[indx]

        print("I have passed this")

        for a in training_normalized.keys():
            training_normalized[a] = pd.DataFrame(((self.training_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers_final)
            training_normalized[a]["activityID"] = self.training_cleaned[a]["activityID"]
            # print(np.unique(training_normalized[a]["activityID"]))
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values) / (max.values - min.values),columns=self.headers_final)
            test_normalized[a]["activityID"] = self.test_cleaned[a]["activityID"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(((self.validation_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers_final)
            validation_normalized[a]["activityID"] = self.validation_cleaned[a]["activityID"]

        self.training_normalized = training_normalized
        self.test_normalized = test_normalized
        self.validation_normalized = validation_normalized

        # print(validation_normalized)

    def segment_data(self, data_dict, window_size, overlap):
        """
        Segments the data into fixed-width windows with overlapping.

        :param data_dict: Dictionary with participant ID as keys and DataFrames as values.
        :param window_size: The size of each window (number of rows).
        :param overlap: The overlap between consecutive windows (number of rows).
        :return: A dictionary with the same keys as data_dict and values as lists of segmented DataFrames.
        """
        segmented_data = {}

        for participant_id, df in data_dict.items():
            num_rows = len(df)
            segments = []
            start = 0
            while start < num_rows:
                end = start + window_size
                if end > num_rows:
                    break
                segment = df.iloc[start:end, :]
                # Check if the segment contains more than one unique label, if so, skip this segment
                if len(segment.iloc[:, -1].unique()) > 1:
                    start += overlap
                    continue
                segments.append(segment)
                start += overlap
            segmented_data[participant_id] = segments
        return segmented_data

    def clean_nan(self, data):
        data_clean = {a: 0 for a in data.keys()}
        for a in data.keys():
            data_aux = data[a].ffill(axis=0).bfill(axis=0)
            data_clean[a] = data_aux
        return data_clean

    def butter_lowpass(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def preprocessing(self):
        training_cleaned_aux = self.clean_nan(self.training)
        test_cleaned_aux = self.clean_nan(self.test)
        validation_cleaned_aux = self.clean_nan(self.validation)

        lenght = 0
        for a in training_cleaned_aux.keys():
            training_cleaned_aux[a] = training_cleaned_aux[a][training_cleaned_aux[a]["activityID"] != 0]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)
            # print(np.unique(training_cleaned_aux[a]["activityID"]))
            lenght = lenght + len(training_cleaned_aux[a])
        for a in validation_cleaned_aux.keys():
            validation_cleaned_aux[a] = validation_cleaned_aux[a][validation_cleaned_aux[a]["activityID"] != 0]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(validation_cleaned_aux[a])
        for a in test_cleaned_aux.keys():
            test_cleaned_aux[a] = test_cleaned_aux[a][test_cleaned_aux[a]["activityID"] != 0]
            test_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(test_cleaned_aux[a])


        for a in training_cleaned_aux.keys():
            # print(training_cleaned_aux[a].iloc[::2].shape)
            training_cleaned_aux[a] = training_cleaned_aux[a]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)
        for a in validation_cleaned_aux.keys():
            validation_cleaned_aux[a] = validation_cleaned_aux[a]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)
        for a in test_cleaned_aux.keys():
            test_cleaned_aux[a] = test_cleaned_aux[a]
            test_cleaned_aux[a].reset_index(drop=True, inplace=True)

        self.training_cleaned = training_cleaned_aux
        self.test_cleaned = test_cleaned_aux
        self.validation_cleaned = validation_cleaned_aux


    def data_segmentation(self):
        train_data_segmented = self.segment_data(self.training_normalized, 256, 128)
        validation_data_segmented = self.segment_data(self.validation_normalized, 256, 128)
        test_data_segmented = self.segment_data(self.test_normalized, 256, 128)

        self.training_normalized_segmented = train_data_segmented
        self.test_normalized_segmented = test_data_segmented
        self.validation_normalized_segmented = validation_data_segmented

    def prepare_dataset(self):

        training, validation, testing = [], [], []
        for a in self.training_normalized_segmented.keys():
            for b in self.training_normalized_segmented[a]:
                training.append((np.transpose(b.iloc[:,0:-1].to_numpy()), int(b.iloc[0, -1])-1, int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                validation.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1])-1, int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                testing.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1])-1, int(a)))

        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing
