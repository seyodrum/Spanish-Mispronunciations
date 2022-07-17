#                               CLASSIFICATION METRICS

#   Imports
import numpy as np
import seaborn as sn
import sklearn.metrics as skm
from pandas import DataFrame
import matplotlib.pyplot as plt
from os import path
from matplotlib.patches import Polygon


#   Functions
def plot_classification_metrics(y_true, y_predicted, labels, directory=''):
    metrics_columns = ['Precision', 'Recall', 'F1-Score']

    metrics = np.asarray(skm.precision_recall_fscore_support(y_true, y_predicted))[:3, :].T
    np.save(path.join(directory, 'metrics.npy'), metrics)
    print('Classification Metrics:')
    print(metrics)
    data_frame = DataFrame(metrics, index=labels, columns=metrics_columns)
    fig1 = plt.figure('Metrics', [9, 9])
    ax1 = fig1.gca()
    ax = sn.heatmap(
        data_frame,
        annot=True,
        annot_kws={"size": 11},
        linewidths=0.5,
        ax=ax1,
        vmin=0,
        vmax=1,
        cbar=True,
        cmap='Blues',
        linecolor='w',
        fmt='.4f',
    )
    fig1.savefig(fname=path.join(directory, 'metrics.pdf'))
    return fig1


def plot_boxplots(data, labels, title='boxplot', conf={'logDir': '', 'plotProcess': True}):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Classification Results')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.1)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.yaxis.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.7)
    ax1.set(axisbelow=True, title='Classification Results: ' + title, xlabel='Class', ylabel='Value')

    # Now fill the boxes with desired colors
    box_colors = ['orange', 'royalblue', 'limegreen', 'turquoise', 'silver']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate colors
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % len(labels)]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.05
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_yticks(np.linspace(0, 1, 11), minor=False)
    ax1.set_xticklabels(labels, rotation=0, fontsize=10)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 4)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick
        ax1.text(
            pos[tick],
            0.10,
            upper_labels[tick],
            transform=ax1.get_xaxis_transform(),
            horizontalalignment='center',
            size='medium',
            weight=weights[0],
            color=box_colors[k],
        )

    if conf['plotProcess']:
        plt.show()
    fig.savefig(fname=path.join(conf['logDir'], title + '.pdf'))
    return fig


def plot_train_history(history={}, conf={'logDir': '', 'plotProcess': True}):
    keys = history.keys()

    if len(keys):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        epochs = len(history['accuracy']) - 1

        #   Accuracy
        axes[0].plot(history['accuracy'], label='Train')

        if 'val_accuracy' in keys:
            axes[0].plot(history['val_accuracy'], label='Test')

        axes[0].set_title('Epoch Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlim([0, epochs])
        axes[0].set_ylim([0, 1])
        axes[0].grid()
        axes[0].legend()

        #   Loss
        axes[1].plot(history['loss'], label='Train')

        if 'val_loss' in keys:
            axes[1].plot(history['val_loss'], label='Test')

        axes[1].set_title('Epoch Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlim([0, epochs])
        axes[1].grid()
        axes[1].legend()

    if conf['plotProcess']:
        plt.show()
    fig.savefig(fname=path.join(conf['logDir'], 'train_history.pdf'))
    return fig


#   Test
if __name__ == '__main__':
    y_t = [1, 2, 3, 1]
    y_p = [3, 2, 3, 1]
    # plot_classification_metrics(y_t, y_p, ['A', 'B', 'C'])

    N = 50
    norm = np.random.normal(1, 1, N)
    logn = np.random.lognormal(1, 1, N)
    expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    bootstrap_indices = np.random.randint(0, N, N)
    data = [
        norm,
        norm[bootstrap_indices],
        logn,
        logn[bootstrap_indices],
        expo,
        expo[bootstrap_indices]
        # gumb, gumb[bootstrap_indices],
        # tria, tria[bootstrap_indices],
    ]
    plot_boxplots(data, ['0', '1', '2'])
    plt.show()
