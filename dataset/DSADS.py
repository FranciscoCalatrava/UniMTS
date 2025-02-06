import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import os
import re
from pathlib import Path

class DSADS():
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
        self.PATH = current_directory

        self.final_headers = [*(f"T_acc_{axis}" for axis in ['x','y','z']),
           *(f"T_gyro_{axis}" for axis in ['x','y','z']),
           *(f"RA_acc_{axis}" for axis in ['x','y','z']),
           *(f"RA_gyro_{axis}" for axis in ['x','y','z']),
           *(f"LA_acc_{axis}" for axis in ['x','y','z']),
           *(f"LA_gyro_{axis}" for axis in ['x','y','z']),
           *(f"RL_acc_{axis}" for axis in ['x','y','z']),
           *(f"RL_gyro_{axis}" for axis in ['x','y','z']),
           *(f"LL_acc_{axis}" for axis in ['x','y','z']),
           *(f"LL_gyro_{axis}" for axis in ['x','y','z']),
            'activityID']

        self.initial_headers = [
        *(f"T_{sensor}_{axis}" for sensor in ['acc', 'gyro', 'mag'] for axis in ['x', 'y', 'z']),
        *(f"RA_{sensor}_{axis}" for sensor in ['acc', 'gyro', 'mag'] for axis in ['x', 'y', 'z']),
        *(f"LA_{sensor}_{axis}" for sensor in ['acc', 'gyro', 'mag'] for axis in ['x', 'y', 'z']),
        *(f"RL_{sensor}_{axis}" for sensor in ['acc', 'gyro', 'mag'] for axis in ['x', 'y', 'z']),
        *(f"LL_{sensor}_{axis}" for sensor in ['acc', 'gyro', 'mag'] for axis in ['x', 'y', 'z']),
        "activityID"]


        self.dataset_name = 'DSADS'
        self.original_frequency = 25



    def get_datasets(self):
        base_path = Path(self.PATH) / 'datasets' / self.dataset_name / 'normal'
        # Define regex patterns
        pattern_activity = re.compile(r"a(\d+)", re.IGNORECASE)  # Match activities (e.g., a1, A2)
        pattern_subject = re.compile(r"p(\d+)", re.IGNORECASE)  # Match subjects (e.g., p1, P2)
        pattern_signal = re.compile(r"s(\d+)\.txt", re.IGNORECASE)  # Match signals (e.g., s1.txt, S2.TXT)
        subject_data = {}
        training = {a: 0 for a in self.train_participant}
        test = {a: 0 for a in self.test_participant}
        validation = {a: 0 for a in self.validation_participant}

        for activity in os.listdir(base_path):
            activity_match = pattern_activity.match(activity)
            if activity_match:
                activity_id = int(activity_match.group(1))  # Extract numeric activity ID
                activity_path = os.path.join(base_path, activity)
                # Loop through subject folders inside activity
                for subject in os.listdir(activity_path):
                    subject_match = pattern_subject.match(subject)
                    if subject_match:
                        subject_id = int(subject_match.group(1))  # Extract subject ID (e.g., 'p3')
                        subject_path = os.path.join(activity_path, subject)
                        # Read all signal files and concatenate
                        signals_list = []
                        for file in os.listdir(subject_path):
                            signal_match = pattern_signal.match(file)
                            if signal_match:
                                file_path = os.path.join(subject_path, file)

                                # Read CSV file (assuming data is comma-separated)
                                df = pd.read_csv(file_path, header=None, delimiter=",")
                                df['activityID'] = activity_id
                                df.columns = self.initial_headers


                                # Append data
                                signals_list.append(df)

                        # If we found signals, concatenate and store
                        if signals_list:
                            subject_df = pd.concat(signals_list, axis=0, ignore_index=True)

                            # Store in dictionary with subject as key
                            if subject_id not in subject_data:
                                subject_data[subject_id] = subject_df
                            else:
                                subject_data[subject_id] = pd.concat([subject_data[subject_id], subject_df], axis=0, ignore_index=True)
                    # Assign to the corresponding split
                    if subject_id in training:
                        training[subject_id] = subject_data[subject_id][self.final_headers]
                    elif subject_id in validation:
                        validation[subject_id] = subject_data[subject_id][self.final_headers]
                    elif subject_id in test:
                        test[subject_id] = subject_data[subject_id][self.final_headers]
                    else:
                        print(f"Volunteer {subject_id} in file {file} is not assigned to any split.")

        # Optionally, assign the dictionaries to instance attributes.
        self.training = training
        self.validation = validation
        self.test = test

    def normalize(self):
        training_normalized = {a: 0 for a in self.training_cleaned.keys()}
        test_normalized = {a: 0 for a in self.test_cleaned.keys()}
        validation_normalized = {a: 0 for a in self.validation_cleaned.keys()}

        max = pd.DataFrame(np.zeros((1, len(self.final_headers))), columns=self.final_headers)
        min = pd.DataFrame(np.zeros((1, len(self.final_headers))), columns=self.final_headers)

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
            training_normalized[a] = pd.DataFrame(
                ((self.training_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.final_headers)
            training_normalized[a]["activityID"] = self.training_cleaned[a]["activityID"]
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values) / (max.values - min.values),
                                              columns=self.final_headers)
            test_normalized[a]["activityID"] = self.test_cleaned[a]["activityID"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(
                ((self.validation_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.final_headers)
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

        for a in training_cleaned_aux.keys():
            training_cleaned_aux[a] = training_cleaned_aux[a][training_cleaned_aux[a]["activityID"] != 0]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)
        for a in validation_cleaned_aux.keys():
            validation_cleaned_aux[a] = validation_cleaned_aux[a][validation_cleaned_aux[a]["activityID"] != 0]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)
        for a in test_cleaned_aux.keys():
            test_cleaned_aux[a] = test_cleaned_aux[a][test_cleaned_aux[a]["activityID"] != 0]
            test_cleaned_aux[a].reset_index(drop=True, inplace=True)

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
                training.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1]) - 1, int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                validation.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1]) - 1, int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                testing.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1]) - 1, int(a)))

        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing