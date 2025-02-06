import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, lfilter
import os
import re


class USCHAD():
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



        self.dataset_name = 'USCHAD'

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
        base_dir = os.path.join(self.PATH, f"datasets/{self.dataset_name}/normal")

        # Regular expression pattern to parse file names like "a1t1.mat"
        activity_trial_pattern = re.compile(r'^a(\d+)t(\d+)\.mat$')

        # Define header for the DataFrame: sensor readings plus the activity label.

        # Loop over each folder in the base_dir.
        for subject_folder in os.listdir(base_dir):
            subject_path = os.path.join(base_dir, subject_folder)

            # Process only directories that start with "subject"
            if os.path.isdir(subject_path) and subject_folder.startswith("Subject"):
                # List all .mat files in the subject folder.
                mat_files = [f for f in os.listdir(subject_path) if f.endswith(".mat")]


                for mat_file in mat_files:
                    file_path = os.path.join(subject_path, mat_file)

                    # Optionally, extract the activity and trial numbers from the filename.
                    match = activity_trial_pattern.match(mat_file)
                    if match:
                        activity_from_filename = match.group(1)  # activity number as string
                        trial_from_filename = match.group(2)  # trial number as string
                        # These can be used for verification or logging if needed.

                    # Load the .mat file.
                    try:
                        mat_data = scipy.io.loadmat(file_path)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue  # Skip this file if there is an error.

                    # Ensure that both 'sensor_readings' and 'activity_number' are available.
                    if "sensor_readings" not in mat_data or "activity_number" not in mat_data:
                        print(f"Missing 'sensor_readings' or 'activity_number' in {file_path}")
                        continue

                    # Extract sensor readings.
                    sensor_data = mat_data["sensor_readings"]
                    # Sometimes MATLAB files introduce extra dimensions. Squeeze removes them.
                    sensor_data = np.squeeze(sensor_data)

                    # If sensor_data is 1D with 6 elements, convert it to 2D with one row.
                    if sensor_data.ndim == 1:
                        if sensor_data.shape[0] == 6:
                            sensor_data = sensor_data.reshape(1, -1)
                        else:
                            print(f"Unexpected sensor_data shape in {file_path}: {sensor_data.shape}")
                            continue

                    # Verify that the sensor_data has 6 channels.
                    if sensor_data.shape[1] != 6:
                        print(f"Unexpected number of sensor channels in {file_path}: {sensor_data.shape[1]}")
                        continue

                    # Extract the activity number.
                    try:
                        activity_val = int(activity_from_filename)
                        # Create a column for the activity label that repeats for each sensor reading sample.
                        activity_column = np.full((sensor_data.shape[0], 1), activity_val)

                        # Combine the sensor readings and activity label.
                        combined_data = np.hstack((sensor_data, activity_column))

                        # Create a DataFrame using the specified header order.
                        df = pd.DataFrame(combined_data, columns=self.headers)

                        # The pattern matches "subject" (case-insensitive) followed by one or more digits.
                        pattern = re.compile(r"(?i)subject(\d+)")
                        match = pattern.search(subject_folder)
                        if match:
                            subject_number = int(match.group(1))
                            if subject_number in training:
                                if training[subject_number].empty:
                                    training[subject_number] = df.astype(float)
                                else:
                                    training[subject_number] = pd.concat([training[subject_number], df],
                                                                         ignore_index=True).astype(float)
                            elif subject_number in validation:
                                if validation[subject_number].empty:
                                    validation[subject_number] = df.astype(float)
                                else:
                                    validation[subject_number] = pd.concat([validation[subject_number], df],
                                                                           ignore_index=True).astype(float)
                            elif subject_number in test:
                                if test[subject_number].empty:
                                    test[subject_number] = df.astype(float)
                                else:
                                    test[subject_number] = pd.concat([test[subject_number], df],
                                                                     ignore_index=True).astype(
                                        float)
                            else:
                                print(f"Volunteer {subject_number}  is not assigned to any split.")
                        else:
                            raise ValueError(f"No subject number found in '{subject_folder}'")
                        # Append the DataFrame to the appropriate dictionary based on the subject.


                    except Exception as e:
                        print(f"Error extracting activity_number from {file_path}: {e}")
                        continue
                        # Save the dictionaries to the instance variables.
        self.training = training
        self.validation = validation
        self.test = test

                    # try:
                    #     # Typically activity_number is stored as a 2D array with one element.
                    #     activity_val = int(activity_raw[0][0])
                    # except Exception as e:
                    #     try:
                    #         activity_val = int(activity_raw)
                    #     except Exception as e:
                    #         print(f"Error extracting activity_number from {file_path}: {e}")
                    #         continue




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
