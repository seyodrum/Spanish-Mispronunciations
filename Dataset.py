import os.path
import tensorflow as tf
import numpy as np
import pandas as pd


#   Classes
class Dataset(object):
    def __init__(self, conf, train_percentage=0.7):
        """
        Constructor

        :param conf: Dataset configurations
        :param train_percentage: Proportion of the training Dataset
        """
        self.dataframe = pd.core.frame.DataFrame
        self.dataframe_len = 0
        self.labels = []
        self.sample_rate = 0
        self.train_percentage = train_percentage
        self.train = {}
        self.test = tf.data.Dataset
        self.get_dataframe(conf)

    # Main Functions
    def init_train_test(self, conf):
        """
        Initialize Train and Test datasets.

        :return:
        """
        self.labels = self.dataframe['y1'].unique()
        self.labels.sort()
        self.balance_dataframe(conf)
        self.dataframe_len = len(self.dataframe.index)
        self.shuffle_dataframe(conf)
        dataset = self.create_dataset(self.dataframe)
        dataset_dict = self.split_dataset(dataset)
        self.train['all'] = dataset_dict['train']
        self.test = dataset_dict['test']

        print('New Dataset:', self.dataframe_len, 'words')

    # Secondary functions
    def balance_dataframe(self, conf):
        """

        :param conf:
        """
        dataframe_dict = {}
        dataframes = []

        for label in self.labels:
            print('Label: ', label)
            dataframe = self.dataframe.loc[self.dataframe['y1'] == label]

            if len(dataframe.index) > conf['max_class_size']:
                dataframe = dataframe[: conf['max_class_size']]

            dataframes.append(dataframe)
            dataframe_dict = self.split_dataframe(dataframe)
            self.train[label] = self.create_dataset(dataframe_dict['train'])

        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print('Max size per label: ', conf['max_class_size'])

    def create_dataset(self, dataframe):
        """
        Turn dataframe into Audio Dataset.

        :param dataframe: Dataframe of the dataset
        :return: dataset
        """
        self.sample_rate = self.get_sample_rate(dataframe['x'].iloc[0])
        dataset_data_path = tf.data.Dataset.from_tensor_slices(dataframe.x)
        dataset_label1 = tf.data.Dataset.from_tensor_slices(tf.cast(dataframe.y1, tf.int8))
        dataset = tf.data.Dataset.zip((dataset_data_path, dataset_label1))
        dataset = dataset.map(self.load_audio, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def get_dataframe(self, conf):
        """
        Get the dataframe representation of the CSV Dataset.

        :param conf: Dataset configurations
        :return:
        """
        dataframe = pd.read_csv(os.path.join(conf['path'], conf['csv'] + '.csv'))
        dataframe['x'] = dataframe['x'].apply(lambda name: os.path.join(conf['path'], 'words', name + '.wav'))
        self.dataframe = dataframe
        self.dataframe_len = len(dataframe.index)
        print('CSV columns:', dataframe.columns.values)
        print('Initial Dataset:', self.dataframe_len, 'words')

    def get_dataset(self, labels=[]):
        """
        Get a concatenated Dataset with the specified labels.

        :param conf: Dataset configurations
        :return:
        """
        if len(labels) > 1:
            dataset = self.train[labels[0]]
            for index, label in enumerate(labels):
                if index == 0:
                    continue

                dataset = dataset.concatenate(self.train[label])

            return dataset
        elif len(labels) == 1:
            return self.train[labels[0]]
        else:
            return self.train['all']

    def get_random_integers(self):
        """

        :return:
        """
        rng = np.random.default_rng()
        return rng.choice(a=self.dataframe_len, size=self.dataframe_len, replace=False).astype(np.int16)

    def get_sample_rate(self, file_path):
        """ """
        file = tf.io.read_file(file_path)
        _, sample_rate = tf.audio.decode_wav(file, 1, 1)
        return sample_rate.numpy()

    def load_audio(self, file_path, label):
        """
        Load 16-bits WAV mono-audio.

        :param file_path: File path were the audio is located
        :param label: Classification label of the audio
        :return: audio, label
        """
        file = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(file, 1)
        audio = self.trim_audio(audio[:, 0])
        return audio, label

    @staticmethod
    def save_dataframe(dataframe, logDir='', name='dataframe'):
        try:
            compression_opts = dict(method='zip', archive_name=name + '.csv')
            dataframe.to_csv(os.path.join(logDir, name + '.zip'), index=False, compression=compression_opts)
        except Exception as ex:
            print(ex)

    @staticmethod
    def save_dataset(dataset, logDir='', name='dataset'):
        try:
            tf.data.experimental.save(dataset, os.path.join(logDir, name))
        except Exception as ex:
            print(ex)

    def shuffle_dataframe(self, conf):
        """

        :param conf: Dataset configurations
        :return:
        """
        random_ints = []
        try:
            if conf['random_ints']:
                random_ints = np.load(os.path.join(conf['path'], str(conf['random_ints']) + '.npy'))
            else:
                random_ints = self.get_random_integers()
        except FileNotFoundError:
            random_ints = self.get_random_integers()
            np.save(os.path.join(conf['path'], conf['random_ints'] + '.npy'), random_ints)
        except Exception as ex:
            print(ex)
        finally:
            self.dataframe = self.dataframe.loc[self.dataframe.index[random_ints]]

    def split_dataframe(self, dataframe):
        """
        Split dataframe into Train and Test dataframe. The proportion is set by the user.

        :param dataset: Dataframe to be split
        :return:
        """
        train_size = np.rint(self.train_percentage * dataframe.index.shape[0]).astype(np.int32)
        dataframe_dict = {'train': dataframe[:train_size], 'test': dataframe[train_size:]}
        print(
            'Train data:',
            dataframe_dict['train'].index.shape[0],
            ' words',
            '-->',
            "{:.2%}".format(self.train_percentage),
        )
        print(
            'Test data:',
            dataframe_dict['test'].index.shape[0],
            ' words',
            '-->',
            "{:.2%}".format(1 - self.train_percentage),
        )
        return dataframe_dict

    def split_dataset(self, dataset):
        """
        Split dataset into Train and Test datasets. The proportion is set by the user.

        :param dataset: Dataset to be split
        :return:
        """
        train_size = np.rint(self.train_percentage * dataset.cardinality().numpy())
        dataset_dict = {'train': dataset.take(train_size), 'test': dataset.skip(train_size)}
        print(
            'Train data:',
            dataset_dict['train'].cardinality().numpy(),
            ' words',
            '-->',
            "{:.2%}".format(self.train_percentage),
        )
        print(
            'Test data:',
            dataset_dict['test'].cardinality().numpy(),
            ' words',
            '-->',
            "{:.2%}".format(1 - self.train_percentage),
        )
        return dataset_dict

    @staticmethod
    def trim_audio(audio, min_value=1e-2):
        min_value = tf.constant([min_value])
        trim = tf.math.logical_or(tf.math.greater(audio, min_value), tf.math.less(audio, -min_value))
        trim = tf.where(trim)
        return audio[trim[0, 0] : trim[-1, 0] + 1]
