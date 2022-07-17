import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#   Classes
class DataAugmentation(object):
    def __init__(self, train_dataset, signal_noise_rate=10, shift_samples=4800, plotProcess=True, labels={}):
        self.augmented = 0
        self.noise = tf.data.Dataset
        self.roll = tf.data.Dataset
        self.signal_noise_rate = signal_noise_rate
        self.shift_samples = shift_samples
        self.init_data_augmentation(train_dataset, plotProcess, labels)

    # Main functions
    def init_data_augmentation(self, train_dataset, plotProcess, labels):
        """


        :return:
        """
        print('Getting White Noise Augmentation')
        self.noise = train_dataset.map(self.white_noise, num_parallel_calls=tf.data.AUTOTUNE)
        self.augmented += self.noise.cardinality().numpy()
        print('Getting Roll Augmentation')
        self.roll = train_dataset.map(self.roll_data, num_parallel_calls=tf.data.AUTOTUNE)
        self.augmented += self.roll.cardinality().numpy()

        if plotProcess:
            self.plot_data_augmentation(train_dataset, labels)

    # Secondary functions
    def plot_data_augmentation(self, train_dataset, labels):
        """


        :return:
        """
        audio = list(train_dataset.as_numpy_iterator())[0][0]
        fig, axes = plt.subplots(3, figsize=(12, 8))
        timescale = np.arange(audio.shape[0])
        label = labels[list(train_dataset.as_numpy_iterator())[0][1]]

        axes[0].plot(timescale, audio)
        axes[0].set_title('Waveform: ' + label)
        axes[0].set_xlim([0, audio.shape[0]])
        axes[0].xaxis.set_ticklabels([])
        axes[0].grid()

        audio = list(self.noise.as_numpy_iterator())[0][0]
        axes[1].plot(timescale, audio)
        axes[1].set_title('White Noise: SNR = ' + str(self.signal_noise_rate))
        axes[1].set_xlim([0, audio.shape[0]])
        axes[1].xaxis.set_ticklabels([])
        axes[1].set_ylabel('Amplitude')
        axes[1].grid()

        audio = list(self.roll.as_numpy_iterator())[0][0]
        axes[2].plot(timescale, audio)
        axes[2].set_title('Roll: ' + str(self.shift_samples) + ' Samples')
        axes[2].set_xlim([0, audio.shape[0]])
        axes[2].set_xlabel('Sample')
        axes[2].grid()

        plt.show()

    def roll_data(self, audio, label):
        """
        Roll n samples in the audio. The samples are set by the user.

        :param audio:
        :param label:
        :return:
        """
        roll_audio = tf.roll(audio, self.shift_samples, 0)
        return roll_audio, label

    def white_noise(self, audio, label):
        """
        Add gaussian white noise to the audio.

        :param audio: Audio to be Augmented
        :param label: Classification label of the audio
        :return: noise_audio, label
        """
        power_audio = tf.math.square(audio)
        power_audio = tf.tensordot(power_audio, tf.ones(tf.shape(audio)), 1)
        power_audio = tf.math.divide(power_audio, tf.cast(tf.shape(audio), tf.float32))
        noise_variance = tf.math.divide(power_audio, self.signal_noise_rate)
        noise = tf.random.normal(tf.shape(audio), stddev=tf.math.sqrt(noise_variance))
        noise_audio = tf.math.add(audio, noise)

        return noise_audio, label
