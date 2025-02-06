import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, lfilter
import os
import re
from pathlib import Path


class SHOAIB():
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



        self.dataset_name = 'SHOAIB'

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


        # Define the base directory containing subject folders.
        dataset_folder = Path(os.path.join(self.PATH, f"datasets/{self.dataset_name}/normal"))
        # Define the sensor names in the order they appear.
        sensors = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
        # The signals we want from each sensor block.
        signals = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

        processed_data = {}

        for file in os.listdir(dataset_folder):
            if not file.startswith("Participant_") or not file.endswith(".csv"):
                continue

            # Parse participant number from the filename.
            try:
                participant_number = int(file.split("_")[1].split(".")[0])
            except Exception as e:
                print(f"Error parsing participant number from {file}: {e}")
                continue

            file_path = dataset_folder / file
            try:
                # Read CSV with two header rows so that a MultiIndex is created.
                raw = pd.read_csv(file_path, header=None)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # The first two rows:
            header1 = raw.iloc[0, :-1]  # exclude the last column (label)
            header2 = raw.iloc[1, :-1]
            # The data starts at row 2 (zero-indexed row 2 is the third row) and excludes the label column.
            data = raw.iloc[2:, :-1].reset_index(drop=True)
            # The label column is the last column.
            label = raw.iloc[2:, -1].reset_index(drop=True)

            # Fill forward the sensor names from header1.
            sensor_names = header1.ffill()
            # Create a flat header by combining sensor name and variable name.
            new_header = sensor_names.astype(str) + "_" + header2.astype(str)
            data.columns = new_header

            # Define the sensors and the variables we want to keep.
            sensors = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
            # For each sensor block, we want these variables:
            desired_vars = ["time_stamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

            sensor_blocks = {}
            # Process each sensor block.
            for sensor in sensors:
                # Select columns that start with sensor + "_"
                # and whose suffix (after the underscore) is in desired_vars.
                cols = [col for col in data.columns
                        if col.startswith(sensor + "_") and col[len(sensor) + 1:] in desired_vars]
                if not cols:
                    print(f"Sensor {sensor} not found in {file_path.name}.")
                    continue
                # Reorder the columns by the order in desired_vars.
                ordered_cols = sorted(cols, key=lambda x: desired_vars.index(x[len(sensor) + 1:]))
                # Extract the block.
                block = data[ordered_cols].copy()
                sensor_blocks[sensor] = block

            # Check that we have a Left_pocket block to use as reference.
            if "Left_pocket" not in sensor_blocks:
                print(f"Left_pocket block not found in {file_path.name}; skipping file.")
                return None

            # Use Left_pocket time_stamp as reference.
            ref_ts = sensor_blocks["Left_pocket"][f"Left_pocket_time_stamp"]

            # For each sensor block (other than Left_pocket), if its time_stamp does not match ref_ts,
            # set its signals (Ax, Ay, Az, Gx, Gy, Gz) to NaN.
            for sensor, block in sensor_blocks.items():
                ts_col = f"{sensor}_time_stamp"
                if ts_col not in block.columns:
                    continue
                if sensor == "Left_pocket":
                    # For the reference sensor, we can drop the time_stamp column later.
                    continue
                mismatches = block[ts_col] != ref_ts
                for var in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
                    col_name = f"{sensor}_{var}"
                    if col_name in block.columns:
                        block.loc[mismatches, col_name] = np.nan
                # (We keep the time_stamp for now if needed; we will drop it later.)
                sensor_blocks[sensor] = block

            # Now, drop the time_stamp column from each sensor block and rename the remaining columns.
            for sensor in sensors:
                if sensor not in sensor_blocks:
                    continue
                block = sensor_blocks[sensor]
                ts_col = f"{sensor}_time_stamp"
                if ts_col in block.columns:
                    block = block.drop(columns=[ts_col])
                # Rename remaining columns to be just sensor_signal.
                new_cols = {col: sensor + "_" + col.split("_")[1] for col in block.columns}
                block = block.rename(columns=new_cols)
                sensor_blocks[sensor] = block

            # Concatenate sensor blocks horizontally.
            final_df = pd.concat([sensor_blocks[sensor] for sensor in sensors if sensor in sensor_blocks], axis=1)
            # Append the activity label as a new column.
            final_df["activityID"] = label
            #print(final_df)
            # Define the activity mapping
            activity_mapping = {
                'walking': 1,
                'sitting': 2,
                'standing': 3,
                'jogging': 4,
                'biking': 5,
                'upstairs': 6,
                'downstairs': 7
            }

            # print(np.unique(final_df['activityID']))

            # Apply the mapping to final_df["activityID"]
            final_df["activityID"] = final_df["activityID"].map(activity_mapping)


            self.headers = final_df.columns
            if participant_number in training:
                if training[participant_number].empty:
                    training[participant_number] = final_df.astype(float)
                else:
                    training[participant_number] = pd.concat([training[participant_number], final_df], ignore_index=True).astype(float)
            elif participant_number in validation:
                if validation[participant_number].empty:
                    validation[participant_number] = final_df.astype(float)
                else:
                    validation[participant_number] = pd.concat([validation[participant_number], final_df], ignore_index=True).astype(float)
            elif participant_number in test:
                if test[participant_number].empty:
                    test[participant_number] = final_df.astype(float)
                else:
                    test[participant_number] = pd.concat([test[participant_number], final_df], ignore_index=True).astype(float)
            else:
                print(f"Volunteer {participant_number}  is not assigned to any split.")


            # if participant_number in training:
            #     training[participant_number] = final_df
            # elif participant_number in validation:
            #     validation[participant_number] = final_df
            # elif participant_number in test:
            #     test[participant_number] = final_df
            # else:
            #     print(f"Volunteer {participant_number} in file {file} is not assigned to any split.")



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

    def preprocessing(self):
        training_cleaned_aux = self.clean_nan(self.training)
        test_cleaned_aux = self.clean_nan(self.test)
        validation_cleaned_aux = self.clean_nan(self.validation)

        lenght = 0
        for a in training_cleaned_aux.keys():
            training_cleaned_aux[a] = training_cleaned_aux[a][training_cleaned_aux[a]["activityID"] != 0]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(training_cleaned_aux[a])
        for a in validation_cleaned_aux.keys():
            validation_cleaned_aux[a] = validation_cleaned_aux[a][validation_cleaned_aux[a]["activityID"] != 0]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)
            lenght = lenght + len(validation_cleaned_aux[a])
        for a in test_cleaned_aux.keys():
            test_cleaned_aux[a] = test_cleaned_aux[a][test_cleaned_aux[a]["activityID"] != 0]
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
