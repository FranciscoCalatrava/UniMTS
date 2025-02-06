import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import scipy
import os
import re
from pathlib import Path
from io import StringIO
import io


class WISDM():
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
        self.label_mapping = {
            'A': 1,  # walking
            'B': 2,  # jogging
            'C': 3,  # stairs
            'D': 4,  # sitting
            'E': 5,  # standing
            'F': 6,  # typing
            'G': 7,  # teeth
            'H': 8,  # soup
            'I': 9,  # chips
            'J': 10,  # pasta
            'K': 11,  # drinking
            'L': 12,  # sandwich
            'M': 13,  # kicking
            'O': 14,  # catch
            'P': 15,  # dribbling
            'Q': 16,  # writing
            'R': 17,  # clapping
            'S': 18  # folding
        }



        self.dataset_name = 'WISDM'

    def load_wisdm_file(self,filepath, device, sensor):
        """
        Load a WISDM text file.

        Assumptions:
          - Each row ends with a semicolon.
          - Columns are separated by commas.
          - The columns (without header) are: user_id, activity_label, timestamp, x, y, z.
        After loading, the three measurement columns are renamed to include device and sensor type.
        """
        # with open(filepath, 'r') as f:
        #     lines = f.readlines()
        #
        # # Remove trailing semicolon and whitespace from each line.
        # lines = [line.strip().rstrip(';') for line in lines if line.strip()]
        # csv_str = "\n".join(lines)

        # Read CSV from the string.
        df = pd.read_csv(filepath, sep = r',|;', header=None,engine='python')
        df.columns = ['user_id', 'activity_label', 'timestamp', 'x', 'y', 'z','nothing']

        # Rename measurement columns so they reflect device and sensor.
        rename_map = {
            'x': f"{device}_{sensor}_x",
            'y': f"{device}_{sensor}_y",
            'z': f"{device}_{sensor}_z",
            'activity_label': f"activity_label_{device}_{sensor}",
            'nothing': 'nothing'
        }
        df = df.rename(columns=rename_map)

        # We only need the timestamp, activity_label, and the 3 renamed measurement columns.
        df = df[['timestamp', f'activity_label_{device}_{sensor}', f"{device}_{sensor}_x", f"{device}_{sensor}_y", f"{device}_{sensor}_z"]]

        return df

    def get_datasets(self):
        training = {a: pd.DataFrame() for a in self.train_participant}
        test = {a: pd.DataFrame() for a in self.test_participant}
        validation = {a: pd.DataFrame() for a in self.validation_participant}

        # Set the base directory of your WISDM dataset
        base_path = os.path.join(self.PATH, f"datasets/{self.dataset_name}/normal")  # CHANGE THIS to your dataset path
        root_path = Path(base_path)

        pattern = re.compile(r'data_(\d+)_((?:accel)|(?:gyro))_((?:phone)|(?:watch))\.txt')

        # Create a dictionary to group files by user.
        # For each user, we store a dict whose keys are tuples (device, sensor) and values are full file paths.
        files_by_user = {}
        for root, dirs, files in os.walk(base_path):
            for file in files:
                m = pattern.match(file)
                if m:
                    user_id, sensor, device = m.groups()
                    full_path = os.path.join(root, file)
                    if user_id not in files_by_user:
                        files_by_user[user_id] = {}
                    files_by_user[user_id][(device, sensor)] = full_path

        print("Found files for the following users and combinations:")
        for user, combos in files_by_user.items():
            print(f"User {user}: {list(combos.keys())}")
        # We expect four combinations for each user.
        expected_combinations = [('phone', 'accel'),
                                 ('phone', 'gyro'),
                                 ('watch', 'accel'),
                                 ('watch', 'gyro')]

        # Dictionary to hold the merged DataFrame for each original user.

        merged_data_by_user = {}
        for user_id, file_dict in files_by_user.items():
            dfs = {a:{b:pd.DataFrame()} for a, b in expected_combinations}
            for combo in expected_combinations:
                device, sensor = combo
                if combo in file_dict:
                    try:
                        df = self.load_wisdm_file(file_dict[combo], device, sensor)
                    except Exception as e:
                        print(f"[ERROR] Loading file for user {user_id} combo {combo}: {e}")
                        # Create an empty DataFrame with the expected columns.
                        df = pd.DataFrame(columns=['timestamp',
                                                   f"activity_label_{device}_{sensor}",
                                                   f"{device}_{sensor}_x", f"{device}_{sensor}_y",
                                                   f"{device}_{sensor}_z"])
                else:
                    # File is missing: create an empty DataFrame with the expected columns.
                    df = pd.DataFrame(columns=['timestamp',
                                               f"activity_label_{device}_{sensor}",
                                               f"{device}_{sensor}_x", f"{device}_{sensor}_y", f"{device}_{sensor}_z"])
                dfs[device][sensor] = df

            # Merge the DataFrames on 'timestamp' using an outer join.
            merged_df = None
            for df_key in expected_combinations:
                device,sensor = df_key
                if merged_df is None:
                    merged_df = dfs[device][sensor]
                else:
                    # Now that each df has a uniquely named activity_label column,
                    # we can merge without causing duplicate column conflicts.
                    merged_df = pd.merge(merged_df, dfs[device][sensor], on='timestamp', how='outer')



            # Combine all activity_label columns into a single one.
            activity_cols = [col for col in merged_df.columns if col.startswith('activity_label')]
            # print(activity_cols)
            if activity_cols:
                # For each row, take the first non-null value across the activity label columns.
                merged_df['activityID'] = merged_df[activity_cols].bfill(axis=1).iloc[:, 0]
                merged_df.drop(columns=activity_cols, inplace=True)

            # Reorder columns: timestamp, activity_label, and then the 12 signal columns.
            final_signal_order = [
                'phone_accel_x', 'phone_accel_y', 'phone_accel_z',
                'phone_gyro_x', 'phone_gyro_y', 'phone_gyro_z',
                'watch_accel_x', 'watch_accel_y', 'watch_accel_z',
                'watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z'
            ]
            merged_df = merged_df.reindex(columns=final_signal_order+ ['timestamp', 'activityID'])

            # Sort by timestamp (optional)
            merged_df = merged_df.sort_values(by='timestamp').reset_index(drop=True)
            merged_df['activityID'] = merged_df['activityID'].map(self.label_mapping)
            merged_data_by_user[user_id] = merged_df

        # --- Step 4: Create a 0-indexed User Mapping and Final Dictionary ---
        unique_users = sorted(merged_data_by_user.keys(), key=lambda x: int(x))
        user_mapping = {orig: new+1 for new, orig in enumerate(unique_users)}

        final_data_by_user = {}
        for orig_id, df in merged_data_by_user.items():
            new_id = user_mapping[orig_id]
            final_data_by_user[new_id] = df

        # --- Optional: Print Summary Information ---
        print("User mapping (original -> new):")
        # print(user_mapping)
        for uid, df in final_data_by_user.items():
            print(
                f"User {uid} (original {[k for k, v in user_mapping.items() if v == uid][0]}) has data shape: {df.shape}")
            if uid in training:
                if training[uid].empty:
                    training[uid] = df.drop(columns=['timestamp']).astype(float)
                    self.headers = training[uid].columns
                else:
                    training[uid] = pd.concat([training[uid], df.drop(columns=['timestamp'])], ignore_index=True).astype(float)
            elif uid in validation:
                if validation[uid].empty:
                    validation[uid] = df.drop(columns=['timestamp']).astype(float)
                else:
                    validation[uid] = pd.concat([validation[uid], df.drop(columns=['timestamp'])], ignore_index=True).astype(float)
            elif uid in test:
                if test[uid].empty:
                    test[uid] = df.drop(columns=['timestamp']).astype(float)
                else:
                    test[uid] = pd.concat([test[uid], df.drop(columns=['timestamp'])], ignore_index=True).astype(float)
            else:
                print(f"Volunteer {uid} is not assigned to any split.")

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
            training_normalized[a] = pd.DataFrame(((self.training_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers)
            training_normalized[a]["activityID"] = self.training_cleaned[a]["activityID"]
            # print(np.unique(training_normalized[a]["activityID"]))
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values) / (max.values - min.values),columns=self.headers)
            test_normalized[a]["activityID"] = self.test_cleaned[a]["activityID"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(((self.validation_cleaned[a].values - min.values) / (max.values - min.values)), columns=self.headers)
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
        # # exclude_columns = ['activityID']

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
