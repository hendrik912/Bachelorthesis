import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import eeggan.Bachelorarbeit.util as util

# --------------------------------------------------------------------------

def plot_classifer_results(result_dict, base_path, show=True, show_title=True):
    """
    Creates plots for the classifiers for different metrics

    Parameters
    ----------
    result_dict: The dictionary with the results of the classifier
    base_path: Root path for the results
    show: Whether or not to show the plots
    show_title: Whether or not to show the titles
    """
    metrics = ["accuracy", "precision", "recall", "fscore"]

    for m in metrics:
        graphs_all_div_over_ratio(result_dict, base_path, m, show=show, show_title=show_title)

# --------------------------------------------------------------------------

def graphs_all_div_over_ratio(result_dict, path, metric, show=True, show_title=True):
    """
    Creates Graphs for a division of the original dataset with the ratio as the x-axis
    and lines for different sample lengths.

    Parameters
    ----------
    result_dict: The dictionary with the results of the classifier
    path: Path to store the images to
    metric: The metric to use for the plots
    show: Whether or not to show the plots
    show_title: Whether or not to show the titles
    """
    divisions = [k for k in result_dict]
    ratios = [k for k in result_dict[1][4]["separate"]]
    ratios.insert(0, 0)
    sample_lengths = [s for s in result_dict[1]]

    print(">> ", path)

    cols = 2
    fig, ax = plt.subplots(2, cols, sharey=True)

    for i, div in enumerate(divisions):
        lines = []
        legend = []

        for sl in sample_lengths:
            line = []

            for r in ratios:
                if r == 0:
                    type_ = "base"
                else:
                    type_ = "separate"
                line.append(result_dict[div][sl][type_][r][metric]["avg"])

            legend.append("SL " + str(sl))
            lines.append(line)

        x_label = "Ratio"
        title = "1/" + str(div)
        coord = util.get_1d_as_n_m(i, cols)
        axis = ax[coord[0], coord[1]]
        graph_to_subplot(axis, title, ratios, lines, legend, y_label=metric.title(), x_label=x_label)

    title = metric.title() + ", all div, over ratio"
    if show_title:
        fig.suptitle(title , fontsize=12)
    fig.set_size_inches(7, 7)
    fig.tight_layout()
    plt.savefig(os.path.join(path, title), bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------

def graph_to_subplot(axis, title, x_vals, res, legend, y_label="", x_label=""):
    """
    Adds a graph to an axis
    """

    axis.set_title(title, size=9)
    matplotlib.rcParams.update({'font.size': 6})

    x_ticks = [y for y in x_vals]
    x = range(len(x_vals))

    for i,r in enumerate(res):
        axis.xaxis.set_ticks(x)
        axis.xaxis.set_ticklabels(x_ticks, size=8)
        axis.yaxis.set_tick_params(labelsize=8)

        axis.plot(x, r, label=legend[i])

        axis.legend(loc='upper right')
        axis.set_xlabel(x_label, size=8)
        axis.set_ylabel(y_label, size=8)
        axis.margins(0)

# --------------------------------------------------------------------------

def plot_confusion_matrices(title, data, figure_path, save=True, show=True, show_title=True, info_text=""):
    """
    Plot confusion matrices as subplots

    Parameters
    ----------
    title: Title of plot
    data: list of tuples with predictions and references
    figure_path: path to store the figure to
    save: whether or not to store the image
    show: whether or not to show the image
    show_title: whether or not to show the title
    info_text: additional text to be printed in the figure
    """

    fig, ax = plt.subplots(len(data), len(data[0]))
    if show_title:
        fig.suptitle(title + " " + info_text, fontsize=12)

    for i, row in enumerate(data):
        for j, col in enumerate(row):

            preds = col[0][0]
            labels = col[0][1]

            print(labels)

            subplot_title = col[1]

            if len(data) > 1:
                axis = ax[i][j]
            else:
                axis = ax[j]

            font_dict = {'family': 'arial', 'weight': 'normal', 'size': 8}
            axis.set_title(subplot_title, fontdict=font_dict)
            axis.set_xlabel('Prädiktion', fontdict=font_dict)
            axis.set_ylabel('Referenz', fontdict=font_dict)

            cm = confusion_mat(preds.astype(int), labels.astype(int))
            cax = axis.matshow(cm)
            fig.colorbar(cax, ax=axis, shrink=1)

    fig.tight_layout()
    if save: plt.savefig(os.path.join(figure_path, title), bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------

def confusion_mat(prediction, reference):
    """
    Create a confusion matrix from predictions and the actual labels (reference) and return it.
    """

    classes=np.unique(reference)
    matrix = np.zeros((len(classes),len(classes)))
    for s in range(len(reference)):
        matrix[reference[s],prediction[s]]+=1
    return matrix

# --------------------------------------------------------------------------

