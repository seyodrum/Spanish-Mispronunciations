from genericpath import isfile
import os
import signal
from datetime import datetime
import numpy as np

import metrics
import main as Main
import sys

#   Functions
def massive_test(experiment):
    conf = {
        'dataAugmentation': {'enable': experiment['enable_dataAugmentation'], 'labels': []},
        'dataset': {
            'csv': experiment['csv'],
            'max_class_size': experiment['max_class_size'],
            'path': 'dataset',
            'random_ints': '',
        },
        'labels': {
            0: '/r/ Vibrante alveolar simple',
            1: '/r/ Vibrante alveolar múltiple',
            2: '<tr> Combinación ortográfica',
            3: '<dr> Combinación ortográfica',
            4: 'Bien pronunciada',
        },
        'logDir': '',
        'model': {'dnn': experiment['dnn'], 'dnn_layers': experiment['dnn_layers']},
        'plotProcess': experiment['enable_tensorboard'] or experiment['plotProcess'],
        'preprocessing': {'transformation': experiment['transformation'], 'type': experiment['type']},
        'tensorboard': {
            'enable': experiment['enable_tensorboard'],
            'histogram_freq': experiment['tensorboard_histogram_freq'],
        },
    }

    if conf['tensorboard']['enable']:
        conf['plotProcess'] = True

    sys.stdout = open('log.txt', 'x')
    print(
        '############################    START:',
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        '   ############################',
    )
    print(conf, '\n\n')

    for dataset in range(experiment['number_subdatasets']):
        conf['dataset']['random_ints'] = experiment['random_ints'] + '_' + str(dataset)

        for cont in range(experiment['number_repetitions']):
            conf['logDir'] = os.path.join(experiment['logDir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
            print(
                '########################    Experiment:',
                conf['dataset']['random_ints'],
                ' -',
                cont,
                '   ########################',
            )
            Main.executeProcess(conf)
            print('\n\n\n')

    print(
        '############################    END:',
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        '   ############################',
    )
    sys.stdout.close()
    os.rename('log.txt', os.path.join(experiment['logDir'], 'log.txt'))
    plot_boxplot(experiment)
    os.kill(os.getpid(), signal.SIGTERM)


def plot_boxplot(experiment):
    list_dir = [
        os.path.join(experiment['logDir'], dir)
        for dir in os.listdir(experiment['logDir'])
        if not isfile(os.path.join(experiment['logDir'], dir))
    ]
    confusion_matrix = []
    precision = []
    recall = []
    f1_score = []

    #   Confusion Matrix
    for dir in list_dir:
        path_load = os.path.join(dir, 'confusionMatrix.npy')

        if os.path.exists(path_load):
            list_metrics = np.load(path_load).T
            confusion_matrix.append(np.around(list_metrics.diagonal() / np.sum(list_metrics, axis=0), decimals=4))

    confusion_matrix = np.array(confusion_matrix).T.tolist()
    labels = [str(label) for label in range(len(confusion_matrix))]
    metrics.plot_boxplots(confusion_matrix, labels, experiment['dnn'].upper() + ' - Accuracy', experiment)

    #   Metrics
    for dir in list_dir:
        path_load = os.path.join(dir, 'metrics.npy')

        if os.path.exists(path_load):
            list_metrics = np.around(np.load(path_load), decimals=4)
            precision.append(list_metrics[:, 0])
            recall.append(list_metrics[:, 1])
            f1_score.append(list_metrics[:, 2])

    precision = np.array(precision).T.tolist()
    recall = np.array(recall).T.tolist()
    f1_score = np.array(f1_score).T.tolist()

    metrics.plot_boxplots(precision, labels, experiment['dnn'].upper() + ' - Precision', experiment)
    metrics.plot_boxplots(recall, labels, experiment['dnn'].upper() + ' - Recall', experiment)
    metrics.plot_boxplots(f1_score, labels, experiment['dnn'].upper() + ' - F1-Score', experiment)


if __name__ == '__main__':
    #   User Input
    experiment = {
        'csv': 'tags3',
        'dnn': 'cnn',
        'dnn_layers': 3,
        'enable_dataAugmentation': True,
        'enable_tensorboard': False,
        'max_class_size': 200,
        'name': 'test',
        'number_repetitions': 10,
        'number_subdatasets': 4,
        'plotProcess': False,
        'random_ints': 'rints_tags3',
        'tensorboard_histogram_freq': 1000,
        'transformation': 'stft',
        'type': 'resize',
    }
    experiment['logDir'] = os.path.join('tensorboard', experiment['name'])

    print('0 : massive_test\n1 : plot_boxplot')
    option = int(input('Enter value:'))
    if option == 0:
        massive_test(experiment)
    elif option == 1:
        plot_boxplot(experiment)
    print('End')
