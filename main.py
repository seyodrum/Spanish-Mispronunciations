import os
import io

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset
from DataAugmentation import DataAugmentation
from Preprocessing import Preprocessing
import confusion_matrix_pretty_print as cfm
import metrics

#   Classes
class TrainingCallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        Stops training when accuracy is reached

        :param epoch:
        :return:
        """
        if logs.get('accuracy') > 0.99:
            print('\nReached', '{:.2f}%'.format(logs.get('accuracy') * 100), 'accuracy so stopping training!')
            self.model.stop_training = True


#   Functions
def get_labels(ds):
    """
    Get labels of the dataset

    :param ds: Dataset with data and labels
    :return: labels
    """
    labels = []
    for _, label in ds:
        labels.append(label.numpy())
    labels = np.array(labels)
    return labels


def save_figure_tensorboard(name, logDir):
    """
    Save the last plotted Figure in Tensorboard

    :param name: Name of the Figure
    :return:
    """
    file_writer = tf.summary.create_file_writer(os.path.join(logDir, 'validation'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    image = tf.expand_dims(image, 0)
    with file_writer.as_default():
        tf.summary.image(name, image, step=0)


def executeProcess(conf):
    #   Make Directories
    os.makedirs(conf['logDir'])

    #   Get Dataset
    dataset = Dataset(conf['dataset'])
    dataset.init_train_test(conf['dataset'])

    if conf['dataAugmentation']['enable']:
        #   Data Augmentation
        dataAugmentation = DataAugmentation(
            dataset.get_dataset(conf['dataAugmentation']['labels']),
            plotProcess=conf['plotProcess'],
            labels=conf['labels'],
        )
        print('Data Augmentation: ', dataAugmentation.augmented, 'words')
        dataset.train['all'] = (
            dataset.train['all'].concatenate(dataAugmentation.noise).concatenate(dataAugmentation.roll)
        )

    #   Pre-Processing data
    batch_size = 10
    print('Batch size:', batch_size)
    preprocessing = Preprocessing(dataset.train['all'], dataset.test, batch_size)
    preprocessing.case(conf['preprocessing'], conf['model']['dnn'], conf['plotProcess'])

    #   Neural Network Model
    model = keras.Sequential([keras.layers.Input(shape=preprocessing.input_shape), preprocessing.normalization_layer])

    for layer in range(conf['model']['dnn_layers']):
        if conf['model']['dnn'] == 'cnn':
            model.add(
                keras.layers.Conv2D(
                    filters=pow(2, conf['model']['dnn_layers'] + 4 - layer),
                    kernel_size=(3, 3),
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01),
                )
            )
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(keras.layers.LSTM(units=pow(2, conf['model']['dnn_layers'] + 5 - layer), return_sequences=True))
            model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=len(dataset.labels), activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    #   Model Training
    model_callbacks = [TrainingCallbacks()]

    if conf['tensorboard']['enable']:
        model_callbacks.append(
            keras.callbacks.TensorBoard(log_dir=conf['logDir'], histogram_freq=conf['tensorboard']['histogram_freq'])
        )

    model.fit(
        preprocessing.train,
        validation_data=preprocessing.test,
        steps_per_epoch=preprocessing.steps_per_epoch,
        epochs=30,
        verbose=2,
        callbacks=model_callbacks,
    )

    #   Model Evaluation (Summary, Metrics, Confusion Matrix)
    model_path = os.path.join(os.path.dirname(conf['logDir']), 'model')

    if not os.path.exists(model_path):
        model.summary()
        model.save(model_path)
        keras.utils.plot_model(model=model, to_file=model_path + '.pdf', show_shapes=True, show_layer_names=True)

    dataset.save_dataframe(dataset.dataframe, conf['logDir'])

    test_labels = get_labels(preprocessing.test)
    labels = [str(label) for label in dataset.labels]
    classifications = np.argmax(model.predict(preprocessing.test), axis=-1)
    cfm.plot_confusion_matrix_from_data(
        test_labels, classifications, xlabels=labels, ylabels=labels, directory=conf['logDir']
    )

    if conf['tensorboard']['enable']:
        save_figure_tensorboard('Confusion Matrix', conf['logDir'])

    metrics.plot_classification_metrics(test_labels, classifications, labels=labels, directory=conf['logDir'])

    if conf['tensorboard']['enable']:
        save_figure_tensorboard('Classification metrics', conf['logDir'])

    #   Show images
    if conf['plotProcess']:
        plt.show()

    #   Clean up memory resources
    plt.close('all')
    keras.backend.clear_session()
