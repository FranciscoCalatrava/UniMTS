import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import os
import re
from pathlib import Path


class TNDAHAR():
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

        self.period = 32 ## Sampling period

        self.PATH = current_directory

        self.headers = [ ]



        self.dataset_name = 'TNDAHAR'

    def get_datasets(self):
        training = {a: pd.DataFrame() for a in self.train_participant}
        test = {a: pd.DataFrame() for a in self.test_participant}
        validation = {a: pd.DataFrame() for a in self.validation_participant}

        """
            Reads the WHARF dataset from the given dataset_path and groups the files per subject 
            (volunteer) into training, validation, and test dictionaries.

            Expected directory structure:
                dataset_path/
                    Brush_teeth/
                        Accelerometer-2011-03-24-10-24-39-climb_stairs-f1.txt
                        ...
                    Climb_stairs/
                        ...
                    Comb_hair/
                        ...
                    ... (other activity folders)

            File naming convention:
                Accelerometer-[START_TIME]-[HMP]-[VOLUNTEER].txt

                where:
                 - [START_TIME] is a timestamp in the format YYYY-MM-DD-HH-MM-SS,
                 - [HMP] is the name of the HMP (activity) performed,
                 - [VOLUNTEER] is the volunteer ID in the format [g][N] (e.g., f1, m2).

            Assumes that the instance (self) has the following attributes:
              - self.train_volunteers: a list (or set) of volunteer IDs for training (e.g., ["f1", "m3", ...])
              - self.validation_volunteers: for validation
              - self.test_volunteers: for testing

            Each file is read into a pandas DataFrame (assumed to be whitespace-separated and headerless)
            and augmented with the metadata extracted from the filename.

            Returns:
                A tuple of three dictionaries: (training, validation, test)
                where each dictionary has keys equal to volunteer IDs and values as lists of DataFrames.
            """

        base_dir = os.path.join(self.PATH, f"datasets/{self.dataset_name}/normal")
        # Convert dataset_path to a Path object (if not already)
        dataset_folder = Path(base_dir)

        # Iterate over all CSV files in the folder.
        for file in os.listdir(dataset_folder):
            # print(file)
            if not file.endswith(".csv"):
                continue

            file_path = dataset_folder / file
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Select only columns that contain "Acc" or "Gyr"
            # (This automatically discards any columns with "Mag" and "class".)
            selected_columns = [col for col in df.columns if ("Acc" in col or "Gyr" in col)]

            # print(df.columns[-1])
            if df.columns[-1] == 'Class':
                selected_columns = selected_columns + ['Class']
                mapping = {'Class': 'activityID'}
            else:
                selected_columns = selected_columns + ['class']
                mapping = {'class': 'activityID'}
            self.headers = selected_columns
            df_processed = df[selected_columns].copy()

            # Optionally, if you want to also keep the label (the class column), you could do:
            # df_processed["activityID"] = df["class"]

            # Extract a subject identifier from the file name.
            # Here we assume the filename (without extension) is the subject id;
            # you can modify this if your filenames have a different format.
            subject_str = file_path.stem

            pattern = re.compile(r"Subject(\d+)", re.IGNORECASE)
            match = pattern.search(subject_str)
            if match:
                subject_id = int(match.group(1))
            else:
                raise ValueError(f"No subject id found in '{subject_str}'")

            # Remove leading/trailing spaces from column names.
            df_processed.columns = df_processed.columns.str.strip()

            # Print the cleaned column names.
            # print("After stripping whitespace:", df_processed.columns.tolist())

            # Rename the column to 'activityID'. Adjust the key if the actual name is 'class' or another variant.
            df_processed = df_processed.rename(columns=mapping)
            # Alternatively, if the column name is in lowercase, use:
            # df_processed = df_processed.rename(columns={'class': 'activityID'})

            # Check if renaming was successful:
            # print("After renaming:", df_processed.columns.tolist())

            # Force the 'activityID' column to be of type int.
            df_processed['activityID'] = df_processed['activityID'].astype(int)
            final_df = df_processed.copy()

            # Convert the "activityID" column to integer.
            #df_processed['activityID'] = df_processed['activityID'].astype(int)


            if subject_id in training:
                if training[subject_id].empty:
                    training[subject_id] = final_df.astype(float)
                else:
                    training[subject_id] = pd.concat([training[subject_id], final_df], ignore_index=True).astype(float)
            elif subject_id in validation:
                if validation[subject_id].empty:
                    validation[subject_id] = final_df.astype(float)
                else:
                    validation[subject_id] = pd.concat([validation[subject_id], final_df], ignore_index=True).astype(float)
            elif subject_id in test:
                if test[subject_id].empty:
                    test[subject_id] = final_df.astype(float)
                else:
                    test[subject_id] = pd.concat([test[subject_id], final_df], ignore_index=True).astype(float)
            else:
                print(f"Volunteer {subject_id}  is not assigned to any split.")



            if subject_id in training:
                training[subject_id] = df_processed
            elif subject_id in validation:
                validation[subject_id] = df_processed
            elif subject_id in test:
                test[subject_id] = df_processed
            else:
                print(f"Volunteer {subject_id} in file {file} is not assigned to any split.")

        # Optionally, assign the dictionaries to instance attributes.
        self.training = training
        self.validation = validation
        self.test = test

        return training, validation, test

    def normalize(self):
        training_normalized = {a: 0 for a in self.training_cleaned.keys()}
        test_normalized = {a: 0 for a in self.test_cleaned.keys()}
        validation_normalized = {a: 0 for a in self.validation_cleaned.keys()}

        max = pd.DataFrame(np.zeros((1, len(self.headers))), columns=self.headers)
        min = pd.DataFrame(np.zeros((1, len(self.headers))), columns=self.headers)

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
                ((self.training_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers)
            training_normalized[a]["activityID"] = self.training_cleaned[a]["activityID"]
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values) / (max.values - min.values),
                                              columns=self.headers)
            test_normalized[a]["activityID"] = self.test_cleaned[a]["activityID"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(
                ((self.validation_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers)
            validation_normalized[a]["activityID"] = self.validation_cleaned[a]["activityID"]

        self.training_normalized = self.training_cleaned
        self.test_normalized = self.test_cleaned
        self.validation_normalized = self.validation_cleaned

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

    def check_timestamp(self, df):
        frequency_ms = (1 / self.period)
        # print(df.head(10))

        expected_timestamps = pd.Series(np.arange(df.index.min(), df.index.max() + frequency_ms, frequency_ms))
        missing_timestamps = expected_timestamps[~expected_timestamps.isin(df.index)]
        return missing_timestamps

    def preprocessing(self):

        training_cleaned_aux = self.clean_nan(self.training)
        test_cleaned_aux = self.clean_nan(self.test)
        validation_cleaned_aux = self.clean_nan(self.validation)

        length = 0

        for a in training_cleaned_aux.keys():
            length += len(training_cleaned_aux[a])
            training_cleaned_aux[a] = training_cleaned_aux[a][training_cleaned_aux[a]["activityID"] != 0]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)

        for a in validation_cleaned_aux.keys():
            length += len(validation_cleaned_aux[a])
            validation_cleaned_aux[a] = validation_cleaned_aux[a][validation_cleaned_aux[a]["activityID"] != 0]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)

        for a in test_cleaned_aux.keys():
            length += len(test_cleaned_aux[a])
            test_cleaned_aux[a] = test_cleaned_aux[a][test_cleaned_aux[a]["activityID"] != 0]
            test_cleaned_aux[a].reset_index(drop=True, inplace=True)

        # print(f"The lenght is {length}")
        # exclude_columns = ['activityID']

        # for a in training_cleaned_aux.keys():
        #     for col in training_cleaned_aux[a].columns:
        #         if col not in exclude_columns:
        #             training_cleaned_aux[a][col] = self.butter_lowpass_filter(training_cleaned_aux[a][col], 15, 50, 4)

        # for a in test_cleaned_aux.keys():
        #     for col in test_cleaned_aux[a].columns:
        #         if col not in exclude_columns:
        #             test_cleaned_aux[a][col] = self.butter_lowpass_filter(test_cleaned_aux[a][col], 15, 50, 4)

        # for a in validation_cleaned_aux.keys():
        #     for col in validation_cleaned_aux[a].columns:
        #         if col not in exclude_columns:
        #             validation_cleaned_aux[a][col] = self.butter_lowpass_filter(validation_cleaned_aux[a][col], 15, 50, 4)
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
                training.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1])-1, int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                validation.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1])-1, int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                testing.append((np.transpose(b.iloc[:, 0:-1].to_numpy()), int(b.iloc[0, -1])-1, int(a)))
        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing
