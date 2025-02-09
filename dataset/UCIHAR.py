import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter


class UCIHAR():
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

        self.period = 1 / 100

        self.PATH = current_directory

        self.headers = [
            "total_acc_x", "total_acc_y", "total_acc_z","body_acc_x", "body_acc_y", "body_acc_z","gyro_x", "gyro_y", "gyro_z", "activityID"]

        self.headers_old = [
            "total_acc_x", "total_acc_y", "total_acc_z", "body_acc_x", "body_acc_y", "body_acc_z", "gyro_x", "gyro_y",
            "gyro_z"]

    def create_dataframe(self,X, y):
        """ Create a DataFrame from the dataset """
        # Assuming X is a 3D array of shape (samples, timesteps, signals)
        # We need to reshape it into 2D array of shape (samples * timesteps, signals)
        num_samples, num_timesteps, num_signals = X.shape
        reshaped_X = X.reshape(num_samples * num_timesteps, num_signals)

        # Repeat labels for each timestep
        repeated_y = np.repeat(y, num_timesteps)

        # Create the DataFrame
        df = pd.DataFrame(reshaped_X, columns=self.headers_old)
        df['activityID'] = repeated_y
        return df

    # Load a single file as a numpy array
    def load_file(self, filepath):
        # print(filepath)
        dataframe = pd.read_csv(filepath, header=None, sep='\s+',dtype = float)
        return dataframe.values

    # Load a list of files and return as a 3d numpy array
    def load_group(self, filenames, prefix=''):
        loaded = list()
        # print(filenames)
        for name in filenames:
            data = self.load_file(prefix + name)
            loaded.append(data)
        # stack group so that features are the 3rd dimension
        loaded = np.dstack(loaded)
        return loaded

    # Load a dataset group, such as train or test
    def load_dataset_group(self, group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
        # body acceleration
        filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
        # body gyroscope
        filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
        # load input data
        X = self.load_group(filenames, filepath)
        # load class output
        y = self.load_file(prefix + group + '/y_' + group + '.txt')
        # print(prefix + group + '/subject_' + group + '.txt')
        subject = self.load_file(prefix + group + '/subject_' + group + '.txt')
        return X, y,subject

    # Load the dataset, returns train and test X and y elements
    def load_dataset(self, prefix=''):
        # load all train
        trainX, trainy, train_subject = self.load_dataset_group('train', prefix + 'datasets/UCIHAR/normal/')
        # print('X train shape: {}, y train shape: {}'.format(trainX.shape, trainy.shape))
        # load all test
        testX, testy, test_subject = self.load_dataset_group('test', prefix + 'datasets/UCIHAR/normal/')
        # print('X test shape: {}, y test shape: {}'.format(testX.shape, testy.shape))
        # zero-offset class values
        trainy = trainy - 1
        testy = testy - 1
        return trainX, trainy,train_subject, testX, testy, test_subject

    def split_data_by_subject(self, X, y, subjects, subject_dict):
        """ Splits the data based on the subject distribution provided in subject_dict """
        subject_ids = list(subject_dict.keys())  # Get all subject IDs from the dictionary
        mask = np.squeeze(np.isin(subjects, subject_ids))  # Create a mask for subjects that are in the subject_dict
        # print(mask.shape)
        return X[mask,:,:], y[mask,:], subjects[mask,:]

    def get_datasets(self):
        training = {a: 0 for a in self.train_participant}
        test = {a: 0 for a in self.test_participant}
        validation = {a: 0 for a in self.validation_participant}

        trainX, trainy, train_subject, testX, testy, test_subject = self.load_dataset(self.PATH)

        # # Split the data based on subject IDs
        # trainX, trainy, train_subject = self.split_data_by_subject(trainX, trainy, train_subject, training)
        # testX, testy, test_subject = self.split_data_by_subject(testX, testy, test_subject, test)
        # validationX, validationy, validation_subject = self.split_data_by_subject(trainX, trainy, train_subject,validation)  # Adjust source arrays if necessary
        #
        for a in training.keys():
            mask_train = np.squeeze(np.isin(train_subject, a))  # Create a mask for subjects that are in the subject_dict
            mask_test = np.squeeze(np.isin(test_subject, a))
            concat_X = np.concatenate((trainX[mask_train,:,:], testX[mask_test,:,:]), axis=0)
            concat_y = np.concatenate((trainy[mask_train,:], testy[mask_test,:]), axis=0)
            final = self.create_dataframe(concat_X, concat_y)
            training[a] = final
        for a in validation.keys():
            mask_train = np.squeeze(np.isin(train_subject, a))  # Create a mask for subjects that are in the subject_dict
            mask_test = np.squeeze(np.isin(test_subject, a))
            concat_X = np.concatenate((trainX[mask_train,:,:], testX[mask_test,:,:]), axis=0)
            concat_y = np.concatenate((trainy[mask_train,:], testy[mask_test,:]), axis=0)
            final = self.create_dataframe(concat_X, concat_y)
            validation[a] = final
        for a in test.keys():
            mask_train = np.squeeze(np.isin(train_subject, a))  # Create a mask for subjects that are in the subject_dict
            mask_test = np.squeeze(np.isin(test_subject, a))
            concat_X = np.concatenate((trainX[mask_train,:,:], testX[mask_test,:,:]), axis=0)
            concat_y = np.concatenate((trainy[mask_train,:], testy[mask_test,:]), axis=0)
            final = self.create_dataframe(concat_X, concat_y)
            test[a] = final
        # Create DataFrames
        self.training = training
        self.test = test
        # Assuming validationX and validationy are defined
        self.validation = validation

        # print(self.training.head())
        # print(self.test.head())
        # print(self.validation.head())

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
        sensors = {
            1: [2, 3, 4],
            2: [15, 16, 17, 18, 19, 20],
            3: [28, 29, 30, 31, 32, 33],
            4: [41, 42, 43, 44, 45, 46],
            5: [54, 55, 56, 57, 58, 59],
            6: [67, 68, 69, 70, 71, 72],
            7: [80, 81, 82, 83, 84, 85],
            8: [93, 94, 95, 96, 97, 98],
            9: [106, 107, 108, 109, 110, 111]
        }

        # all_sensors = [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 33, 41, 42, 43, 44, 45, 46, 54, 55,
        #                56, 57, 58, 59, 67, 68, 69, 70, 71, 72, 80, 81, 82, 83, 84, 85, 93, 94, 95, 96, 97, 98, 106, 107,
        #                108, 109, 110, 111]
        SENSORS = [0,1,2,3,4,5,6,7,8]

        for a in self.training_normalized_segmented.keys():
            for b in self.training_normalized_segmented[a]:
                training.append((np.transpose(b.iloc[:,SENSORS].to_numpy()), int(b.iloc[0, -1]) - 1, int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                validation.append((np.transpose(b.iloc[:, SENSORS].to_numpy()), int(b.iloc[0, -1]) - 1, int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                testing.append((np.transpose(b.iloc[:, SENSORS].to_numpy()), int(b.iloc[0, -1]) - 1, int(a)))

        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing