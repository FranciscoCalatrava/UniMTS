
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, lfilter
import os
import re
from pathlib import Path



class OPPORTUNITY():
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

        self.period = 100 ## Sampling period

        self.PATH = current_directory

        self.headers = [
            "acc_x",   # Accelerometer X-axis (g)
            "acc_y",   # Accelerometer Y-axis (g)
            "acc_z",   # Accelerometer Z-axis (g)
            "gyro_x",  # Gyroscope X-axis (dps)
            "gyro_y",  # Gyroscope Y-axis (dps)
            "gyro_z",  # Gyroscope Z-axis (dps)
            "activityID" # Activity label (numeric identifier for the activity)
        ]
        self.mapping_labels_locomotion = {0:0,1:1, 2:2,4:3,5:4}
        self.mapping_labels_ML_both_arms = {
            1: 406516,
            2: 406517,
            3: 404516,
            4: 404517,
            5: 406520,
            6: 404520,
            7: 406505,
            8: 404505,
            9: 406519,
            10: 404519,
            11: 406511,
            12: 404511,
            13: 406508,
            14: 404508,
            15: 408512,
            16: 407521,
            17: 405506
        }



        self.dataset_name = 'OPPORTUNITY'

    def get_datasets(self):
        """
           Reads the dataset from the file system. The folder structure is assumed to be:

               {self.PATH}/dataset/{self.dataset_name}/normal/subjectX/*.mat

           Each .mat file is named as "a{activity_number}t{trial_number}.mat" and contains the 13
           fields described in the data specification.

           This function extracts only the sensor readings and the activity label from each file.
           The sensor readings are the 6 channels:

               acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

           and the activity label is taken from the activity_number field.

           The data is organized into dictionaries based on the subject distribution
           (training, validation, and test) as defined in self.train_participant,
           self.validation_participant, and self.test_participant.
           """

        training = {a: pd.DataFrame() for a in self.train_participant}
        test = {a: pd.DataFrame() for a in self.test_participant}
        validation = {a: pd.DataFrame() for a in self.validation_participant}


        # Ensure dataset_path is a Path object
        dataset_path = os.path.join(self.PATH, f"datasets/{self.dataset_name}/normal")

        # Dictionary to hold the data grouped by subject
        subjects_data = {}

        # Regular expression to match file names like S1-ADL1.dat (case-insensitive)
        file_pattern = re.compile(r"^S(\d+)-(?:ADL\d+|Drill)\.dat$", re.IGNORECASE)

        # Loop through each file in the provided directory
        for file_name in os.listdir(dataset_path):
            match = file_pattern.match(file_name)
            if match:
                # Extract the subject number as an integer
                subject_id = int(match.group(1))
                file_path = Path(dataset_path) / file_name

                # Read the file.
                # Here we assume that the file is whitespace-separated and has no header row.
                # The file is expected to have 250 columns.
                df = pd.read_csv(file_path, sep = '\s+', header=None)

                # Verify that the file has the expected number of columns (250)
                if df.shape[1] != 250:
                    print(f"Warning: {file_name} does not have 250 columns. Skipping this file.")
                    continue

                # Define the column indices to keep:
                # Inertial sensor columns: 38 to 134 (1-indexed) => indices 37 to 133 (0-indexed)
                # Label columns: Locomotion is column 244 (index 243) and ML_Both_Arms is column 250 (index 249)
                inertial_indices = list(range(37, 134))
                label_indices = [243, 249]
                cols_to_keep = inertial_indices + label_indices

                # Select the desired columns
                inertial_df = df.iloc[:, cols_to_keep].copy()

                # (Optional) Rename columns to a simplified header.
                # For example, we create generic names for the inertial sensor columns and assign the two label names.
                n_inertial = len(inertial_indices)
                new_column_names = [f"IMU_{i}" for i in range(1, n_inertial + 1)]
                new_column_names += ["Locomotion", "ML_Both_Arms"]
                self.headers = new_column_names
                inertial_df.columns = new_column_names
                inertial_df['Locomotion'] = inertial_df['Locomotion'].map(self.mapping_labels_locomotion)
                inertial_df['ML_Both_Arms'] = inertial_df['ML_Both_Arms'].map(self.mapping_labels_ML_both_arms)


                if subject_id in training:
                    if training[subject_id].empty:
                        training[subject_id] = inertial_df
                    else:
                        training[subject_id] = pd.concat([training[subject_id], inertial_df], ignore_index=True)
                elif subject_id in validation:
                    if validation[subject_id].empty:
                        validation[subject_id] = inertial_df
                    else:
                        validation[subject_id] = pd.concat([validation[subject_id], inertial_df], ignore_index=True)
                elif subject_id in test:
                    if test[subject_id].empty:
                        test[subject_id] = inertial_df
                    else:
                        test[subject_id] = pd.concat([test[subject_id], inertial_df], ignore_index=True)
                else:
                    print(f"Volunteer {subject_id}  is not assigned to any split.")
                # Save the dictionaries to the instance variables.
                self.training = training
                self.validation = validation
                self.test = test

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
            training_normalized[a] = pd.DataFrame(((self.training_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers)
            training_normalized[a]["Locomotion"] = self.training_cleaned[a]["Locomotion"]
            training_normalized[a]["ML_Both_Arms"] = self.training_cleaned[a]["ML_Both_Arms"]
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values) / (max.values - min.values),columns=self.headers)
            test_normalized[a]["Locomotion"] = self.test_cleaned[a]["Locomotion"]
            test_normalized[a]["ML_Both_Arms"] = self.test_cleaned[a]["ML_Both_Arms"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(((self.validation_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers)
            validation_normalized[a]["Locomotion"] = self.validation_cleaned[a]["Locomotion"]
            validation_normalized[a]["ML_Both_Arms"] = self.validation_cleaned[a]["ML_Both_Arms"]

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
                if len(segment.iloc[:, -2].unique()) > 1:
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
            training_cleaned_aux[a] = training_cleaned_aux[a][(training_cleaned_aux[a]["Locomotion"] != 0) & (training_cleaned_aux[a]["ML_Both_Arms"] != 0)]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(training_cleaned_aux[a])
        for a in validation_cleaned_aux.keys():
            validation_cleaned_aux[a] = validation_cleaned_aux[a][(validation_cleaned_aux[a]["Locomotion"] != 0) & (validation_cleaned_aux[a]["ML_Both_Arms"] != 0)]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(validation_cleaned_aux[a])
        for a in test_cleaned_aux.keys():
            test_cleaned_aux[a] = test_cleaned_aux[a][(test_cleaned_aux[a]["Locomotion"] != 0) & (test_cleaned_aux[a]["ML_Both_Arms"] != 0)]
            test_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(test_cleaned_aux[a])

        # print(f"The lenght is {lenght}")

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
                training.append((np.transpose(b.iloc[:,0:-2].to_numpy()), int(b.iloc[0, -2])-1, int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                validation.append((np.transpose(b.iloc[:, 0:-2].to_numpy()), int(b.iloc[0, -2])-1, int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                testing.append((np.transpose(b.iloc[:, 0:-2].to_numpy()), int(b.iloc[0, -2])-1, int(a)))

        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing
