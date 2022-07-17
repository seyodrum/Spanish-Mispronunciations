import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#   Constants
type = {'padding': 'padding', 'resize': 'resize'}

#   Classes
class Preprocessing(object):
    def __init__(self, train_dataset, test_dataset, batch_size):
        """
        Constructor.

        :param train_dataset: Train dataset to be preprocessed
        :param test_dataset: Test dataset to be preprocessed
        :param batch_size: Number of consecutive elements in a single batch
        """
        self.batch_size = batch_size
        self.input_shape = 0
        self.normalization_layer = keras.layers.experimental.preprocessing.Normalization()
        self.steps_per_epoch = 0
        self.test = test_dataset
        self.type = ''
        self.train = train_dataset
        self.transformation = ''
        self.dataset_max_samples = self.get_dataset_max_samples()

    # Dispatcher
    def case(self, conf={'transformation': 'stft', 'type': 'resize'}, dnn='cnn', plotProcess=True):
        preprocessing_case = getattr(self, 'case_' + conf['transformation'], lambda: 'Invalid transformation')
        self.transformation = conf['transformation']
        self.type = conf['type']
        print('Getting', self.transformation, 'representation with', self.type)
        return preprocessing_case(dnn, plotProcess)

    # Main functions
    def case_stft(self, dnn, plotProcess):
        # Train and Normalization layer
        if self.type == type['padding']:
            self.train = self.train.map(self.padding_audio, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            audio = list(self.train.as_numpy_iterator())[0][0]

        self.train = self.train.map(self.get_stft, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            spectrogram = list(self.train.as_numpy_iterator())[0][0]
            fig, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(audio.shape[0])
            axes[0].plot(timescale, audio)
            axes[0].set_title('Waveform')
            axes[0].set_xlim([0, audio.shape[0]])
            axes[0].set_ylabel('Amplitude')
            axes[0].grid()
            self.plot_spectrogram(spectrogram, axes[1])
            axes[1].set_title('Spectrogram')
            axes[1].grid()
            plt.show()

        self.steps_per_epoch = self.train.cardinality().numpy() / self.batch_size
        self.train = self.train.map(self.expand_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        if self.type == type['resize']:
            self.train = self.train.map(self.resize_img, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            spectrogram = list(self.train.as_numpy_iterator())[0][0]
            plt.imshow(spectrogram)
            plt.show()

        if dnn == 'lstm':
            self.train = self.train.map(self.reduce_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        self.input_shape = self.get_input_shape()
        self.normalization_layer.adapt(self.train.map(lambda x, _: x))
        self.train = self.train.shuffle(self.train.cardinality().numpy())
        self.train = self.train.batch(self.batch_size)

        # Test
        if self.type == type['padding']:
            self.test = self.test.map(self.padding_audio, num_parallel_calls=tf.data.AUTOTUNE)

        self.test = self.test.map(self.get_stft, num_parallel_calls=tf.data.AUTOTUNE)
        self.test = self.test.map(self.expand_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        if self.type == type['resize']:
            self.test = self.test.map(self.resize_img, num_parallel_calls=tf.data.AUTOTUNE)

        if dnn == 'lstm':
            self.test = self.test.map(self.reduce_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        self.test = self.test.batch(1)

        # Data Pipeline
        self.set_data_pipeline()

    def case_mfc(self, dnn, plotProcess):
        # Train and Normalization layer
        if self.type == type['padding']:
            self.train = self.train.map(self.padding_audio, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            audio = list(self.train.as_numpy_iterator())[0][0]

        self.train = self.train.map(self.get_mfc, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            spectrogram = list(self.train.as_numpy_iterator())[0][0]
            fig, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(audio.shape[0])
            axes[0].plot(timescale, audio)
            axes[0].set_title('Waveform')
            axes[0].set_xlim([0, audio.shape[0]])
            axes[0].set_ylabel('Amplitude')
            axes[0].grid()
            self.plot_spectrogram(spectrogram, axes[1])
            axes[1].set_title('Spectrogram')
            axes[1].grid()
            plt.show()

        self.steps_per_epoch = self.train.cardinality().numpy() / self.batch_size
        self.train = self.train.map(self.expand_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        if self.type == type['resize']:
            self.train = self.train.map(self.resize_img, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            spectrogram = list(self.train.as_numpy_iterator())[0][0]
            plt.imshow(spectrogram)
            plt.show()

        if dnn == 'lstm':
            self.train = self.train.map(self.reduce_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        self.input_shape = self.get_input_shape()
        self.normalization_layer.adapt(self.train.map(lambda x, _: x))
        self.train = self.train.shuffle(self.train.cardinality().numpy())
        self.train = self.train.batch(self.batch_size)

        # Test
        if self.type == type['padding']:
            self.test = self.test.map(self.padding_audio, num_parallel_calls=tf.data.AUTOTUNE)

        self.test = self.test.map(self.get_mfc, num_parallel_calls=tf.data.AUTOTUNE)
        self.test = self.test.map(self.expand_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        if self.type == type['resize']:
            self.test = self.test.map(self.resize_img, num_parallel_calls=tf.data.AUTOTUNE)

        if dnn == 'lstm':
            self.test = self.test.map(self.reduce_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        self.test = self.test.batch(1)

        # Data Pipeline
        self.set_data_pipeline()

    def case_mfccs(self, dnn, plotProcess):
        # Train and Normalization layer
        if self.type == type['padding']:
            self.train = self.train.map(self.padding_audio, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            audio = list(self.train.as_numpy_iterator())[0][0]

        self.train = self.train.map(self.get_mfccs, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            spectrogram = list(self.train.as_numpy_iterator())[0][0]
            fig, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(audio.shape[0])
            axes[0].plot(timescale, audio)
            axes[0].set_title('Waveform')
            axes[0].set_xlim([0, audio.shape[0]])
            axes[0].set_ylabel('Amplitude')
            axes[0].grid()
            self.plot_spectrogram(spectrogram, axes[1])
            axes[1].set_title('Spectrogram')
            axes[1].grid()
            plt.show()

        self.steps_per_epoch = self.train.cardinality().numpy() / self.batch_size
        self.train = self.train.map(self.expand_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        if self.type == type['resize']:
            self.train = self.train.map(self.resize_img, num_parallel_calls=tf.data.AUTOTUNE)

        if plotProcess:
            spectrogram = list(self.train.as_numpy_iterator())[0][0]
            plt.imshow(spectrogram)
            plt.show()

        if dnn == 'lstm':
            self.train = self.train.map(self.reduce_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        self.input_shape = self.get_input_shape()
        self.normalization_layer.adapt(self.train.map(lambda x, _: x))
        self.train = self.train.shuffle(self.train.cardinality().numpy())
        self.train = self.train.batch(self.batch_size)

        # Test
        if self.type == type['padding']:
            self.test = self.test.map(self.padding_audio, num_parallel_calls=tf.data.AUTOTUNE)

        self.test = self.test.map(self.get_mfccs, num_parallel_calls=tf.data.AUTOTUNE)
        self.test = self.test.map(self.expand_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        if self.type == type['resize']:
            self.test = self.test.map(self.resize_img, num_parallel_calls=tf.data.AUTOTUNE)

        if dnn == 'lstm':
            self.test = self.test.map(self.reduce_data_dimension, num_parallel_calls=tf.data.AUTOTUNE)

        self.test = self.test.batch(1)

        # Data Pipeline
        self.set_data_pipeline()

    # Secondary functions
    @staticmethod
    def expand_data_dimension(data, label):
        """
        Insert a dimension of length 1 at the end of the tensor.
        Allow Conv2d layers to interpret data as 1-D image

        :param data: Data to be expanded
        :param label: Classification label
        :return: data_expanded, label
        """
        data_expanded = tf.expand_dims(data, -1)
        return data_expanded, label

    def get_dataset_max_samples(self):
        """
        Get max length of samples in dataset

        :return: max_samples
        """
        max_samples = 0
        for audio in self.train.as_numpy_iterator():
            if max_samples < np.size(audio[0]):
                max_samples = np.size(audio[0])
        for audio in self.test.as_numpy_iterator():
            if max_samples < np.size(audio[0]):
                max_samples = np.size(audio[0])
        print('Max length of samples detected in Database:', max_samples)
        return max_samples

    def get_input_shape(self):
        """
        Get input shape of the dataset

        :return:
        """
        for data, _ in self.train.take(1):
            print('Input shape:', data.shape)
            return data.shape

    @staticmethod
    def get_mfc(audio, label):
        """
        Get Mel Spectrogram representation of the audio

        :param audio: Audio to be represented as a mel spectrogram
        :param label: Classification label of the audio
        :return: mel_spectrogram, label
        """
        sample_rate, lower_edge_hertz, upper_edge_hertz, num_mel_bins = 44100, 64, 8192, 53
        spectrogram = tf.signal.stft(audio, frame_length=513, frame_step=256, fft_length=512)

        spectrogram = tf.abs(spectrogram)
        num_spectrogram_bins = spectrogram.shape.as_list()[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        mel_spectrogram = tf.math.log(mel_spectrogram + tf.keras.backend.epsilon())
        return mel_spectrogram, label
        # return linear_to_mel_weight_matrix, label

    @staticmethod
    def get_mfccs(audio, label):
        """
        Get Mel-frequency cepstral coefficients of the audio

        :param audio: Audio to be represented as Mel-frequency Cepstral Coefficients
        :param label: Classification label of the audio
        :return: mfcc, label
        """
        sample_rate, lower_edge_hertz, upper_edge_hertz, num_mel_bins = 44100, 64, 8192, 53
        spectrogram = tf.signal.stft(audio, frame_length=513, frame_step=256, fft_length=512)

        spectrogram = tf.abs(spectrogram)
        num_spectrogram_bins = spectrogram.shape.as_list()[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        mel_spectrogram = tf.math.log(mel_spectrogram + tf.keras.backend.epsilon())
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrogram)[..., :13]

        return mfccs, label

    @staticmethod
    def get_stft(audio, label):
        """
        Get Spectrogram representation of the audio

        :param audio: Audio to be represented as a spectrogram
        :param label: Classification label of the audio
        :return: spectrogram, label
        """
        spectrogram = tf.signal.stft(audio, frame_length=513, frame_step=256, fft_length=512, name='STFT')  # 255 603
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.log(spectrogram + tf.keras.backend.epsilon())
        return spectrogram, label

    def padding_audio(self, audio, label):
        """
        Concatenate audio with padding so that all audio clips will be of the same length

        :param audio: Audio to be padded
        :param label: Classification label of the audio
        :return: padded_audio, label
        """
        zero_padding = tf.zeros(self.dataset_max_samples - tf.shape(audio), dtype=tf.float32)
        audio = tf.cast(audio, tf.float32)
        padded_audio = tf.concat([audio, zero_padding], 0)
        return padded_audio, label

    @staticmethod
    def plot_spectrogram(spectrogram, ax):
        """


        :return:
        """
        log_spec = spectrogram.T
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Coefficient')

    @staticmethod
    def reduce_data_dimension(data, label):
        """ """
        data_reduced = tf.squeeze(data)
        return data_reduced, label

    @staticmethod
    def resize_img(image, label):
        resize = tf.image.resize(image, [256, 256])
        return resize, label

    def set_data_pipeline(self):
        """
        Build the input pipeline of the train dataset

        :return:
        """
        self.train = self.train.cache().prefetch(tf.data.AUTOTUNE).repeat()
        self.test = self.test.cache().prefetch(tf.data.AUTOTUNE)
