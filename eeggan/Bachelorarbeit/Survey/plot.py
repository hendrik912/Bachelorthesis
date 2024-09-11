import os
import eeggan.Bachelorarbeit.util as util
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------

def plot_ratings(subjects, result_path, show, show_titles):
    """
    main function to bunble the functionality to get plot the ratings

    Parameters
    ----------
    subjects: list of participants
    result_path: path where to store the plots
    show: whether or not to show the plots
    show_titles: whether or not to show the titles
    """

    ratings_conf = []
    ratings_noise = []

    c, n = get_ratings(subjects, False, False, None)
    ratings_conf.append((c, "equal weight"))
    ratings_noise.append((n, "equal weight"))

    c, n = get_ratings(subjects, True, False, None)
    ratings_conf.append((c, "weighted for experience"))
    ratings_noise.append((n, "weighted for experience"))

    for i in range(0, 2):
        if i == 0:
            title1 = "Confidence that the data is real"
            ratings = ratings_conf
        else:
            title1 = "Rating of the noise level"
            ratings = ratings_noise

        data = []
        labels = []
        category_labels = []

        for r in ratings:
            category_labels.append(r[1])

            for t in r[0]:
                reduced = util.reduce_one_dim(np.array(t[0]))
                data.append(reduced)
                labels.append(t[1])

        plot_boxplot_survey(title1, data, labels, category_labels, result_path + "/", True, show, "",
                                 show_title=show_titles)

# --------------------------------------------------------------------------

def plot_boxplot_survey(title, data, labels, category_labels, figure_path, save, show, info_text, show_title=False):
    """
    Create boxplots for the survey

    """

    path = figure_path + "boxplots/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.rcParams["figure.figsize"] = [7.50, 3.50]

    widths = 0.40
    fig, ax = plt.subplots(1, 2, sharey=True)

    for i in range(0, len(data)):
        ax_ind = int(i/3)

        vals = data[i]
        bp = ax[ax_ind].boxplot(vals, positions=[i % 3], notch=False, widths=widths,
                    boxprops=dict(linewidth=1),
                    flierprops=dict(marker='o', markersize=6, linestyle='none'),
                    showmeans=True, labels=[labels[i % 3]])

        ax[ax_ind].set_title(category_labels[ax_ind])
        plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], loc='upper right', framealpha=0.1)

    fig.subplots_adjust(wspace=0, hspace=0)

    if show_title:
        ti = fig.suptitle(title + " " + info_text)
        ti.set_position([.5, 1.05])

    # # ax.set_title('Title', pad=20)
    if save: plt.savefig(path + title)
    if show: plt.show()

# --------------------------------------------------------------------------

def get_subjects_as_string(subjects):
    """
    turn the list of subjects into a string and return it
    """

    output = ""
    for s in subjects:
        output += str(s) + "\n"
    return output

# --------------------------------------------------------------------------

def get_ratings(subjects, weighted, get_by_exp, experience=None):
    """
    Get the ratings from the subjects

    Parameters
    ----------
    subjects: list of the participants with the ratings they gave
    weighted: whether or not to weigh the results
    get_by_exp: whether or not to only take the ratings of people with a specific experience
    experience: the amount of time the participants have been analysing and/or processing eeg data.

    Returns
    -------

    """

    conf_all = []
    conf_real = []
    conf_fake = []
    noise_all = []
    noise_real = []
    noise_fake = []

    for s in subjects:
        if get_by_exp:
            if s.experience != experience:
                continue

        weight = s.weight if weighted else 1

        for i in range(weight):
            conf_all.append(s.conf_ratings_all)
            conf_real.append(s.conf_ratings_real)
            conf_fake.append(s.conf_ratings_fake)
            noise_all.append(s.noise_ratings_all)
            noise_real.append(s.noise_ratings_real)
            noise_fake.append(s.noise_ratings_fake)

    ratings_conf = [(conf_all, "all"),
                    (conf_real, "real"),
                    (conf_fake, "sim")]

    ratings_noise = [(noise_all, "all"),
                     (noise_real, "real"),
                     (noise_fake, "sim")]

    return ratings_conf, ratings_noise

# --------------------------------------------------------------------------
