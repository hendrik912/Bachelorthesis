import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import eeggan.Bachelorarbeit.GAN.util as g_util
import eeggan.Bachelorarbeit.util as util
import eeggan.Bachelorarbeit.persistence as persistence
import os
import eeggan.plotting.plots as _plot
import eeggan.Bachelorarbeit.constants as constants
from eeggan.Bachelorarbeit.GAN.MetricType import MetricType
import eeggan.Bachelorarbeit.GAN.metrics as metrics
from eeggan.data.dataset import Data
import mne

# --------------------------------------------------------------------------

def plot_psd_channels(result_dict, base_result_path, figure_path, parameters):
    """
    Plots the power spectral density for each channel for the data in result_dict
    """

    for i in range(len(parameters)):

        sl = parameters[i][0]
        bs = parameters[i][1]
        fade = parameters[i][2]
        comb = parameters[i][3]
        postp = parameters[i][4]
        p = (bs, fade, comb, postp)

        if (comb) or sl!=4 or (not postp):
            continue

        num = 1500
        info = mne.create_info(ch_names=constants.CHNS, sfreq=constants.FS, ch_types='eeg')

        ds_name = "all_subjects_" + str(parameters[i][0]) + "s.dataset"
        dataset_path = os.path.join(base_result_path, "datasets")
        real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
        real_data.train_data.X = real_data.train_data.X[:num]
        real_data.train_data.y = real_data.train_data.y[:num]
        real_data.train_data.y_onehot = real_data.train_data.y_onehot[:num]

        fake = result_dict[p][sl]["samples"]
        fake.X = fake.X[:num]
        fake.y = fake.y[:num]
        fake.y_onehot = fake.y_onehot[:num]

        X_real_c0 = util.get_single_class(real_data.train_data, 0)
        X_real_c1 = util.get_single_class(real_data.train_data, 1)

        X_real_c0_reduced = util.reduce_one_dim(X_real_c0.X)
        X_real_c1_reduced = util.reduce_one_dim(X_real_c1.X)
        raw_real_c0 = mne.io.RawArray(X_real_c0_reduced, info)
        raw_real_c1 = mne.io.RawArray(X_real_c1_reduced, info)

        X_fake_c0 = util.get_single_class(fake, 0)
        X_fake_c1 = util.get_single_class(fake, 1)
        X_fake_c0_reduced = util.reduce_one_dim(X_fake_c0.X)
        X_fake_c1_reduced = util.reduce_one_dim(X_fake_c1.X)
        raw_fake_c0 = mne.io.RawArray(X_fake_c0_reduced, info)
        raw_fake_c1 = mne.io.RawArray(X_fake_c1_reduced, info)

        for k in range(0, 2):
            fig, ax = plt.subplots(len(constants.CHNS), 2, sharex=True, sharey=True)

            if k==0:
                fig.suptitle("Real")
                raw_c0 = raw_real_c0
                raw_c1 = raw_real_c1
                fn = "real"
            else:
                fig.suptitle("Simuliert")
                raw_c0 = raw_fake_c0
                raw_c1 = raw_fake_c1
                fn = "fake"

            for j in range(0, len(constants.CHNS)):
                axis = ax[j][0]
                raw_c0.plot_psd(picks=constants.CHNS[j], fmax=40, ax=axis, show=False)
                if j == 0: axis.set_title("internal attention")
                else: axis.set_title(None)
                axis.yaxis.label.set_size(6)

                axis = ax[j][1]
                raw_c1.plot_psd(picks=constants.CHNS[j], fmax=40, ax=axis, show=False)
                if j == 0: axis.set_title("external attention")
                else: axis.set_title(None)
                axis.set_ylabel(None)

            rows = ['{}'.format(row) for row in constants.CHNS]

            pad = 5  # in points

            for ax_, row in zip(ax[:, 0], rows):
                ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                            xycoords=ax_.yaxis.label, textcoords='offset points',
                            size='small', ha='right', va='center')

            fig.tight_layout()
            fig.subplots_adjust(left=0.15, wspace=0, hspace=0)
            fig.set_size_inches(7, 10)
            plt.savefig(os.path.join(figure_path, fn))
            plt.show()

# --------------------------------------------------------------------------

def plot_psd(result_dict, base_result_path, figure_path, parameters):
    """
    Plots the power spectral density for the data in result_dict
    """

    for k in range(0,2):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        sls = []

        for i in range(len(parameters)):
            sl = parameters[i][0]
            bs = parameters[i][1]
            fade = parameters[i][2]
            comb = parameters[i][3]
            postp = parameters[i][4]
            p = (bs, fade, comb, postp)

            if comb or sl != 4: continue
            if k==0 and postp: continue
            if k==1 and (not postp): continue

            sls.append(sl)

            print("-" * 20)
            print(i, "/", len(parameters))

            num = 1000

            ds_name = "all_subjects_" + str(parameters[i][0]) + "s.dataset"
            dataset_path = os.path.join(base_result_path, "datasets")

            real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
            real_data.train_data.X = real_data.train_data.X[:num]
            real_data.train_data.y = real_data.train_data.y[:num]
            real_data.train_data.y_onehot = real_data.train_data.y_onehot[:num]

            fake = result_dict[p][sl]["samples"]
            fake.X = fake.X[:num]
            fake.y = fake.y[:num]
            fake.y_onehot = fake.y_onehot[:num]

            real_c0 = util.get_single_class(real_data.train_data, 0)
            real_c1 = util.get_single_class(real_data.train_data, 1)
            fake_c0 = util.get_single_class(fake, 0)
            fake_c1 = util.get_single_class(fake, 1)

            max_freq = 40

            #_plot.spectral_plot(real_data.train_data.X, fake.X, constants.FS, ax[idx][0], title="all", max_freq=max_freq)
            _plot.spectral_plot(real_c0.X, fake_c0.X, constants.FS, ax[0], title="internal", max_freq=max_freq)
            _plot.spectral_plot(real_c1.X, fake_c1.X, constants.FS, ax[1], title="external", max_freq=max_freq)

            plt.setp(ax[0].get_xticklabels()[-1], visible=False)

            ax[1].set_ylabel(None)
            #ax[idx][0].set_ylabel(None)
            #ax[idx][1].set_ylabel(None)


        fig.tight_layout()
        fig.subplots_adjust(wspace=0)
        fig.set_size_inches(6, 3)

        if k==0:
            # plt.savefig(path + ".pdf", bbox_inches='tight')
            plt.savefig(os.path.join(figure_path, "psd_raw.pdf"), bbox_inches='tight')

# --------------------------------------------------------------------------


def plot_absolute_values(result_dict, base_result_path, figure_path, parameters, boxplots=True, histograms=True, show=True):
    """
    Plots the absolute values as histograms for the data in result_dict
    """

    for i in range(len(parameters)):
        print("-" * 20)
        print(str(i+1), "/", len(parameters))
        sl = parameters[i][0]
        bs = parameters[i][1]
        fade = parameters[i][2]
        comb = parameters[i][3]
        postp = parameters[i][4]

        if sl != 4: continue
        if postp: continue
        if comb: continue

        ds_name = "all_subjects_" + str(parameters[i][0]) + "s.dataset"
        dataset_path = os.path.join(base_result_path, "datasets")

        num_all = 10000
        real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
        real_data.train_data.X = real_data.train_data.X[:num_all]
        real_data.train_data.y = real_data.train_data.y[:num_all]
        real_data.train_data.y_onehot = real_data.train_data.y_onehot[:num_all]

        p1 = (bs, fade, comb, False)

        fake = result_dict[p1][sl]["samples"]
        fake.X = fake.X[:num_all]
        fake.y = fake.y[:num_all]

        # Absolute Values: Histograms & Boxplots
        real_c0 = util.get_single_class(real_data.train_data, 0)
        fake_c0 = util.get_single_class(fake, 0)

        real_c1 = util.get_single_class(real_data.train_data, 1)
        fake_c1 = util.get_single_class(fake, 1)

        break

    n_bins = 120
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)


    # font_size = 8
    # matplotlib.rcParams.update({'font.size': 12})

    X_abs1 = np.abs(util.reduce_one_dim(util.reduce_one_dim(real_c0.X)))
    X_abs2 = np.abs(util.reduce_one_dim(util.reduce_one_dim(fake_c0.X)))
    ax[0].hist([X_abs1, X_abs2], n_bins, histtype="step", stacked=False, label=['real', 'sim'])
    ax[0].legend()
    ax[0].semilogy()

    X_abs1 = np.abs(util.reduce_one_dim(util.reduce_one_dim(real_c1.X)))
    X_abs2 = np.abs(util.reduce_one_dim(util.reduce_one_dim(fake_c1.X)))
    ax[1].hist([X_abs1, X_abs2], n_bins, histtype="step", stacked=False, label=['real', 'sim'])
    ax[1].legend()
    ax[1].semilogy()

    ax[0].set_title("internal")
    ax[1].set_title("external")

    ax[0].set_ylabel("Amount")
    ax[0].set_xlabel("Magnitude")
    ax[1].set_xlabel("Magnitude")

    #ax[j][0].set_ylabel(row_labels[j], rotation=0, size=10, labelpad=15)
    #ax[j][0].yaxis.label.set_size(10)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.set_size_inches(6, 2.5)

    name = os.path.join(figure_path, "histogram.pdf")
    plt.savefig(name, bbox_inches='tight')
    plt.show()

    plt.close(fig)



def plot_absolute_values1(result_dict, base_result_path, figure_path, parameters, boxplots=True, histograms=True, show=True):
    """
    Plots the absolute values as histograms for the data in result_dict
    """
    data_list_sep = []
    data_list_comb = []
    label_list_sep = []
    label_list_comb = []
    row_labels = []
    col_labels = ["alle", "internal attention", "external attention"]

    for i in range(len(parameters)):
        print("-" * 20)
        print(str(i+1), "/", len(parameters))
        sl = parameters[i][0]
        bs = parameters[i][1]
        fade = parameters[i][2]
        comb = parameters[i][3]
        postp = parameters[i][4]

        if sl != 4: continue
        if postp: continue
        if comb: row_labels.append(str(sl) + "s")
        ds_name = "all_subjects_" + str(parameters[i][0]) + "s.dataset"
        dataset_path = os.path.join(base_result_path, "datasets")

        num_all = 1000
        real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
        real_data.train_data.X = real_data.train_data.X[:num_all]
        real_data.train_data.y = real_data.train_data.y[:num_all]
        real_data.train_data.y_onehot = real_data.train_data.y_onehot[:num_all]

        p1 = (bs, fade, comb, False)
        p2 = (bs, fade, comb, True)

        fake = result_dict[p1][sl]["samples"]
        fake.X = fake.X[:num_all]
        fake.y = fake.y[:num_all]
        fake_pp = result_dict[p2][sl]["samples"]
        fake_pp.X = fake_pp.X[:num_all]
        fake_pp.y = fake_pp.y[:num_all]

        # Absolute Values: Histograms & Boxplots
        print("plot absolute values")
        real_c0 = util.get_single_class(real_data.train_data, 0)
        real_c1 = util.get_single_class(real_data.train_data, 1)
        fake_c0 = util.get_single_class(fake, 0)
        fake_c1 = util.get_single_class(fake, 1)
        fake_pp_c0 = util.get_single_class(fake_pp, 0)
        fake_pp_c1 = util.get_single_class(fake_pp, 1)

        data = [[real_data.train_data.X, fake.X,    fake_pp.X],
                [real_c0.X,              fake_c0.X, fake_pp_c0.X],
                [real_c1.X,              fake_c1.X, fake_pp_c1.X]]

        labels =  [["real", "sim", "sim pp"],
                  ["real int", "sim int", "sim int pp"],
                  ["real ext", "sim ext", "sin ext pp"]]

        if boxplots:
            combstr = "separate"
            if comb: combstr = "combined"
            title = combstr + "_sl" + str(sl)
            plot_boxplot1(str(sl) + "s", "boxplot_" + title + "_without_fliers", data, labels, figure_path + "/", True, show,
                          show_outlier=False)
        if not comb:
            data_list_sep.append(data)
            label_list_sep.append(labels)
        else:
            data_list_comb.append(data)
            label_list_comb.append(labels)

    if histograms:
        plot_abs("histogram_abs_all_comb", data_list_comb, label_list_comb, figure_path + "/", True, show, "",
                 show_title=False, row_labels=row_labels, col_labels=col_labels)
        plot_abs("histogram_abs_all_sep", data_list_sep, label_list_sep, figure_path + "/", True, show, "",
                 show_title=False, row_labels=row_labels, col_labels=col_labels)

# --------------------------------------------------------------------------


def plot_metrics_paper_is(result_dict, figure_path, metric, title="Unfiltered"):
    """
    Plots the curves for the results of the distance metrics of the generated data from the GANs
    """

    """
     Dict structure:    
       gan_param (tuple to identify the correct gan)
         sample_length_in_s
           samples (Dataset)
           metrics (jsd, kld, is)
             classes (all, 0, 1)
                scores (e.g. [0.34, 0.28, 0.08, ...])    
     """

    cols = 1
    metrics = g_util.get_metrics_list(result_dict)
    print(metrics)

    metric_labels = {MetricType.JSD: "Jensen-Shannon-Distanz",
                     MetricType.SWD: "Sliced-Wasserstein-Distanz",
                     MetricType.IS: "Inception-Maß",
                     }

    title = metric_labels[metric]
    rows = 1

    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    # plt.suptitle(title)

    show = True
    use_comb = False
    use_pp = False

    res = []
    #res_c0 = []
    #res_c1 = []
    legend = []
    lengths = []

    for p in result_dict:
        print(">> ", p)
        base = True if p == "base" else False
        pts_all = []
        #pts_c0 = []
        #pts_c1 = []

        if not base:
            if not use_comb and p[2]:
                continue
            if not use_pp and p[3]:
                continue
            legend.append("sim")
        else:
            legend.append("real")

        for sl in result_dict[p]:
            if sl not in lengths:
                lengths.append(sl)

            pts_all.append(np.average(result_dict[p][sl][metric]["all"]))
            # pts_c0.append(np.average(result_dict[p][sl][metric]["0"]))
            # pts_c1.append(np.average(result_dict[p][sl][metric]["1"]))

        res.append(pts_all)
        #res_c0.append(pts_c0)
        #res_c1.append(pts_c1)

    ax_ = ax
    graph_to_subplot(ax_, "", res, legend, "", "Window size in seconds", lengths)
    ax_.set_ylabel("Inception Score")


    # ax_.set_xlabel("Window size")
    #ax_.set_title(None)


    title = str(metric.value)
    path = os.path.join(figure_path, title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    fig.set_size_inches(3.5, 2)

    print("path:", path + ".pdf")
    plt.savefig(path + ".pdf", bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)




def plot_metrics_paper_jsd(result_dict, figure_path, metric, title="Unfiltered"):
    """
    Plots the curves for the results of the distance metrics of the generated data from the GANs
    """

    """
     Dict structure:    
       gan_param (tuple to identify the correct gan)
         sample_length_in_s
           samples (Dataset)
           metrics (jsd, kld, is)
             classes (all, 0, 1)
                scores (e.g. [0.34, 0.28, 0.08, ...])    
     """

    cols = 2
    metrics = g_util.get_metrics_list(result_dict)
    print(metrics)

    metric_labels = {MetricType.JSD: "Jensen-Shannon-Distanz",
                     MetricType.SWD: "Sliced-Wasserstein-Distanz",
                     MetricType.IS: "Inception-Maß",
                     }

    title = metric_labels[metric]
    rows = 1

    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    # plt.suptitle(title)

    show = True
    use_comb = False
    use_pp = False

    res_c0 = []
    res_c1 = []
    legend = []
    lengths = []

    for p in result_dict:
        print(">> ", p)
        base = True if p == "base" else False
        # pts_all = []
        pts_c0 = []
        pts_c1 = []

        if not base:
            if not use_comb and p[2]:
                continue
            if not use_pp and p[3]:
                continue
            legend.append("sim")
        else:
            legend.append("real")

        for sl in result_dict[p]:
            if sl not in lengths:
                lengths.append(sl)

            pts_c0.append(np.average(result_dict[p][sl][metric]["0"]))
            pts_c1.append(np.average(result_dict[p][sl][metric]["1"]))

        # res.append(pts_all)
        res_c0.append(pts_c0)
        res_c1.append(pts_c1)


    ax_ = ax[0]
    graph_to_subplot(ax_, "internal", res_c0, legend, "", "Window size in seconds", lengths)
    ax_.set_ylabel("Jensen-Shannon Distance")

    plt.setp(ax[0].get_xticklabels()[-1], visible=False)

    ax_ = ax[1]
    graph_to_subplot(ax_, "external", res_c1, legend, "", "Window size in seconds", lengths)


    # ax_.set_xlabel("Window size")
    #ax_.set_title(None)


    title = str(metric.value)
    path = os.path.join(figure_path, title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    fig.set_size_inches(6, 2.5)

    print("path:", path + ".pdf")
    plt.savefig(path + ".pdf", bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)

def plot_metrics_paper1(result_dict, figure_path, metric, title="Unfiltered"):
    """
    Plots the curves for the results of the distance metrics of the generated data from the GANs
    """

    """
     Dict structure:    
       gan_param (tuple to identify the correct gan)
         sample_length_in_s
           samples (Dataset)
           metrics (jsd, kld, is)
             classes (all, 0, 1)
                scores (e.g. [0.34, 0.28, 0.08, ...])    
     """

    cols = 2
    metrics = g_util.get_metrics_list(result_dict)
    print(metrics)

    metric_labels = {MetricType.JSD: "Jensen-Shannon-Distanz",
                     MetricType.SWD: "Sliced-Wasserstein-Distanz",
                     MetricType.IS: "Inception-Maß",
                     }

    title = metric_labels[metric]
    rows = 2

    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    plt.suptitle(title)

    show = True
    use_comb = False
    use_pp = False

    c_all = "all"
    c0 = "0"
    c1 = "1"

    res = []
    legend = []
    lengths = []

    for p in result_dict:
        print(">> ", p)
        base = True if p == "base" else False
        pts_all = []
        pts_c0 = []
        pts_c1 = []

        if not base:
            if not use_comb and p[2]:
                continue
            if not use_pp and p[3]:
                continue
            legend.append(c_all)
            legend.append(c0)
            legend.append(c1)
        else:
            legend.append("baseline all")
            legend.append("baseline c0")
            legend.append("baseline c1")

        for sl in result_dict[p]:
            if sl not in lengths:
                lengths.append(sl)
            pts_all.append(np.average(result_dict[p][sl][metric][c_all]))
            pts_c0.append(np.average(result_dict[p][sl][metric][c0]))
            pts_c1.append(np.average(result_dict[p][sl][metric][c1]))

        res.append(pts_all)
        res.append(pts_c0)
        res.append(pts_c1)

    it = util.get_1d_as_n_m(0, cols)
    ax_ = ax[it[0]][it[1]]

    graph_to_subplot(ax_, title, res, legend, "", "Window length in seconds", lengths)

    if it[1] > 0:
        ax_.set_ylabel(None)
    if it[0] != rows-1:
        ax_.set_xlabel(None)
    if it[0] > 0:
        ax_.set_title(None)

    title = str(metric.value)
    path = os.path.join(figure_path, title)
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.set_size_inches(10, 10)
    plt.savefig(path, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------

def plot_metrics(result_dict, figure_path, metric):
    """
    Plots the curves for the results of the distance metrics of the generated data from the GANs
    """

    cols = 2
    classes = ["all", "0", "1"]
    metrics = g_util.get_metrics_list(result_dict)
    print(metrics)

    metric_labels = {MetricType.JSD: "Jensen-Shannon-Distanz",
                     MetricType.SWD: "Sliced-Wasserstein-Distanz",
                     MetricType.IS: "Inception-Maß",
                     }

    title = metric_labels[metric]
    rows = 3

    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    plt.suptitle(title)

    show = True
    index = 0

    show_comb = True
    show_sep = True

    for c in classes:
        res = []
        res_pp = []
        legend = []
        legend_pp = []
        lengths = []

        for p in result_dict:
            print(" ", p)
            base = True if p == "base" else False
            pts = []
            postprocessed = False

            if not show_comb and p[2]:
                continue
            if not show_sep and not p[2]:
                continue

            if base:
                legend.append("baseline")
                legend_pp.append("baseline")
            else:
                postprocessed = p[3]
                comb_str = "combined" if p[2] else "seperate"
                name = comb_str
                if postprocessed: legend_pp.append(name)
                else: legend.append(name)

            for sl in result_dict[p]:
                if sl not in lengths:
                    lengths.append(sl)
                pts.append(np.average(result_dict[p][sl][metric][c]))
            if base:
                res_pp.append(pts)
                res.append(pts)
            elif postprocessed:
                res_pp.append(pts)
            else:
                res.append(pts)

        it = util.get_1d_as_n_m(index, cols)
        ax_ = ax[it[0]][it[1]]

        if it[0] == 0:
            subplot_title = "alle"
        elif it[0] == 1:
            subplot_title = "internal"
        else:
            subplot_title = "external"

        graph_to_subplot(ax_, "Unfiltered", res, legend, subplot_title, "Window length in seconds", lengths)
        index += 1

        if it[1] > 0:
            ax_.set_ylabel(None)
        if it[0] != rows-1:
            ax_.set_xlabel(None)
        if it[0] > 0:
            ax_.set_title(None)

        it = util.get_1d_as_n_m(index, cols)
        ax_ = ax[it[0]][it[1]]
        graph_to_subplot(ax_, "Filtered", res_pp, legend_pp, subplot_title, "Window length in seconds",
                         lengths)

        if it[1] > 0:
            ax_.set_ylabel(None)
        if it[0] != rows - 1:
            ax_.set_xlabel(None)
        if it[0] > 0:
            ax_.set_title(None)

        index += 1

    title = str(metric.value)
    path = os.path.join(figure_path, title)
    # plt.subplots_adjust(wspace=0, hspace=0)

    # fig.set_size_inches(10, 10)
    plt.savefig(path, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------

def plot_gan_figures(result_dict, base_result_path, figure_path, num_samples, parameters, show):

    show_raw = True
    show_sorted = True
    metrics_list = [MetricType.SWD, MetricType.IS, MetricType.JSD]

    for i in range(len(parameters)):
        print("-" * 20)
        print(i, "/", len(parameters))
        sl = parameters[i][0]
        bs = parameters[i][1]
        fade = parameters[i][2]
        comb = parameters[i][3]
        postp = parameters[i][4]

        if sl != 4: continue
        if comb: continue
        if postp: continue
        if not fade: continue

        ds_name = "all_subjects_" + str(parameters[i][0]) + "s.dataset"
        dataset_path = os.path.join(base_result_path, "datasets")
        real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
        real_data.train_data.X = real_data.train_data.X
        real_data.train_data.y = real_data.train_data.y
        real_data.train_data.y_onehot = real_data.train_data.y_onehot

        p = (bs, fade, comb, postp)

        fake = result_dict[p][sl]["samples"]
        fake.X = fake.X
        fake.y = fake.y
        fake.y_onehot = fake.y_onehot

        print("real: ", real_data.train_data.X.shape, "fake: ", fake.X.shape)

        folder_name = "all_subjects_" + str(sl) + "s_bs" + str(bs)
        folder_name = folder_name + "_fade" if fade else folder_name
        folder_name = folder_name + "_combined" if comb else folder_name + "_seperate"
        path = os.path.join(figure_path, folder_name)

        # Raw
        print("SORT BY METRIC")

        fake_subset = Data(fake.X, fake.y, fake.y_onehot)
        scale = 4

        plot_raw_freq_sorted_by_metric(path, real_data.train_data.X, fake_subset, constants.FS,
                                       constants.CHNS, num_samples=20,
                                       sl=sl, comb=comb, scale=scale, metric_type=MetricType.JSD, clean=True, show=False,
                                       postprocessed=postp)

        for i in range(0, 20):
            X = real_data.train_data.X[i:i+1]
            title = str(i)

            info = mne.create_info(ch_names=constants.CHNS, sfreq=constants.FS, ch_types='eeg')
            scalings = dict(mag=1e-12, grad=4e-11, eeg=scale, eog=150e-6, ecg=5e-4,
                            emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                            resp=1, chpi=1e-4, whitened=1e2)

            raw = mne.io.RawArray(util.reduce_one_dim(np.copy(X)), info)



            fig = raw.plot(n_channels=len(constants.CHNS), scalings=scalings, title="Real",
                           show=False, block=True, duration=sl+0.5, show_options=False, show_scrollbars=False)

            plt.suptitle("Real")
            plt.tight_layout()

            path_raw = path + title + "_raw.pdf"
            fig.set_size_inches(6, 5)
            fig.savefig(path_raw)
            plt.close(fig)


def plot_gan_figures1(result_dict, base_result_path, figure_path, num_samples, parameters, show):
    """
    Goes through each GAN provided by the parameter-list and plot the samples sorted by a metric as well as the raw signal

    Parameters
    ----------
    result_dict: the dictionary with the results of the GAN evaluation
    base_result_path: the path where the GANs are located
    figure_path: the path where to store the images to
    num_samples: the number of samples to plot
    parameters: list of tuples with param that identify a GAN
    show: whether or not to show the plots
    """

    show_raw = True
    show_sorted = True
    metrics_list = [MetricType.SWD, MetricType.IS, MetricType.JSD]

    for i in range(len(parameters)):

        print("-" * 20)
        print(i, "/", len(parameters))
        sl = parameters[i][0]
        bs = parameters[i][1]
        fade = parameters[i][2]
        comb = parameters[i][3]
        postp = parameters[i][4]

        ds_name = "all_subjects_" + str(parameters[i][0]) + "s.dataset"
        dataset_path = os.path.join(base_result_path, "datasets")
        real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
        real_data.train_data.X = real_data.train_data.X
        real_data.train_data.y = real_data.train_data.y
        real_data.train_data.y_onehot = real_data.train_data.y_onehot

        p = (bs, fade, comb, postp)

        fake = result_dict[p][sl]["samples"]
        fake.X = fake.X
        fake.y = fake.y
        fake.y_onehot = fake.y_onehot

        print("real: ", real_data.train_data.X.shape, "fake: ", fake.X.shape)

        folder_name = "all_subjects_" + str(sl) + "s_bs" + str(bs)
        folder_name = folder_name + "_fade" if fade else folder_name
        folder_name = folder_name + "_combined" if comb else folder_name + "_seperate"
        path = os.path.join(figure_path, folder_name)

        # Raw
        print("SORT BY METRIC")

        fake_subset = Data(fake.X, fake.y, fake.y_onehot)
        scale = 4

        for m in metrics_list:
            if (m is MetricType.IS) or postp or (m is MetricType.SWD):
                continue

            plot_raw_freq_sorted_by_metric(path, real_data.train_data.X, fake_subset, constants.FS,
                                           constants.CHNS, num_samples = num_samples,
                                           sl=sl, comb=comb, scale=scale, metric_type=m, clean=True, show=False,
                                           postprocessed=postp)

# --------------------------------------------------------------------------

def graph_to_subplot(axis, title, res, legend, y_label, x_label, x_values):
    """
    plots a graph onto the given axis
    """

    print(len(legend))
    print(legend)
    print(len(res))

    x_ticks = [y for y in x_values]
    x = range(len(x_values)) #[0,1,2,3,4]

    # font_size = 8
    # matplotlib.rcParams.update({'font.size': 6})

    for i,r in enumerate(res):
        axis.xaxis.set_ticks(x)
        #axis.xaxis.set_ticklabels(x_ticks, size=font_size)
        axis.xaxis.set_ticklabels(x_ticks)
        #axis.yaxis.set_tick_params(labelsize=font_size)
        axis.yaxis.set_tick_params()

        if legend[i] == "baseline":
            axis.plot(x, r, label=legend[i], ls='--')
        else:
            axis.plot(x, r, label=legend[i], ls='-')

        axis.legend(loc='upper right')
        if title:
            axis.set_title(title, size=10)
        # axis.set_xlabel(x_label, size=font_size)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label, rotation=90, size=10, labelpad=25)
        print(">>>>>>>>>", y_label)
        axis.margins(x=0)

# --------------------------------------------------------------------------

def plot_boxplot1(title, filename, data, labels, figure_path, save, show, show_outlier=True):
    """
    Creates boxplots for the data
    """

    path = figure_path + "boxplots/"
    if not os.path.exists(path):
        os.makedirs(path)

    matplotlib.rcParams.update({'font.size': 6})

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    ax = fig.add_subplot(111)
    widths = 0.55

    ax.set_ylabel(title, rotation=0, size=10, labelpad=15)
    ax.yaxis.label.set_size(10)

    index = 0
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            offset = int(index/3) / 2
            vals = util.reduce_one_dim(util.reduce_one_dim(data[i][j]))

            bp = ax.boxplot(vals, positions=[index + offset], notch=False, widths=widths,
                             boxprops=dict(linewidth=1),
                             flierprops = dict(marker='o', markersize=6, linestyle='none'), labels=[labels[i][j]],
                             showfliers=show_outlier, showmeans=True)


            plt.legend([bp['medians'][0], bp['means'][0]], ['Median', 'Durchschnitt'], loc='upper right', framealpha=0.1)
            index += 1

    plt.ylim(-4, 4, 0.1)
    if save: plt.savefig(path + filename)
    if show: plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------

def plot_raw_freq_sorted_by_metric(figure_path, real_data, fake_data, fs, chns, sl, comb, scale, metric_type,
                                   clean=True, show=False, postprocessed=False, num_samples = 20):
    """
    Plots all the samples from fake_data ordered by the distance between fake_data and real_data measured by
    the given metric_type
    """

    path = os.path.join(figure_path, "sorted_by_" + metric_type.value)
    if postprocessed: path += "_pp/"
    else: path += "/"
    if not os.path.exists(path): os.makedirs(path)

    metric_dict = {}
    metric_dict[MetricType.IS] = metrics.inception_score
    metric_dict[MetricType.SWD] = metrics.sliced_wasserstein_dist
    metric_dict[MetricType.JSD] = metrics.jsd

    num = num_samples
    indices = np.arange(0, len(real_data))
    choice = np.random.choice(indices, num)
    real_data = np.copy(real_data[choice])

    fake = fake_data.X[:num]

    metric_list = []

    for i in range(0, len(fake)):
        sample = np.array([fake[i]])
        res = metric_dict[metric_type](sample, real_data, comb, sl)
        metric_list.append(res)

    metric_list = np.array(metric_list)

    A = fake_data.X
    B = metric_list

    inds = B.argsort()
    sorted_a = A[inds]
    sorted_b = B[inds]

    i = 0
    for t in zip(sorted_a, sorted_b):
        j = inds[i]
        i += 1

        X = fake_data.X[j:j+1]
        # title = str(i)
        title = "Simulated"
        info_txt = str(t[1])

        info = mne.create_info(ch_names=chns, sfreq=fs, ch_types='eeg')
        scalings = dict(mag=1e-12, grad=4e-11, eeg=scale, eog=150e-6, ecg=5e-4,
                        emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                        resp=1, chpi=1e-4, whitened=1e2)

        raw = mne.io.RawArray(util.reduce_one_dim(np.copy(X)), info)

        if clean:
            fig = raw.plot(n_channels=len(chns), scalings=scalings, title=title,
                           show=False, block=True, duration=sl+0.5, show_options=False, show_scrollbars=False)
        else:
            fig = raw.plot(n_channels=len(chns), scalings=scalings, title=title,
                       show=False, block=True, duration=sl + 0.5)

        plt.suptitle(title)
        plt.tight_layout()

        path_raw = path + title + f"_raw{i}.pdf"
        fig.set_size_inches(6, 5)
        fig.savefig(path_raw)
        plt.close(fig)

        """
        fig = raw.plot_psd(show=show, average=True, fmin=0, fmax=60)
        plt.suptitle(title + " " + info_txt)
        fig.savefig(path + title + "_freq")
        plt.close(fig)
        """

# --------------------------------------------------------------------------

def plot_raw(title, data, figure_path, save, show, fs, channels, scale, info_text, duration, clean=True):
    """
    Plots the raw eeg data
    """

    path = figure_path + "/raw/"
    if not os.path.exists(path):
        os.makedirs(path)

    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')

    if scale == "auto":
        scalings = scale
    else:
        scalings = dict(mag=1e-12, grad=4e-11, eeg=scale, eog=150e-6, ecg=5e-4,
                        emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                        resp=1, chpi=1e-4, whitened=1e2)

    X = util.reduce_one_dim(data)

    raw = mne.io.RawArray(X, info)

    if clean:
        fig = raw.plot(n_channels=len(channels), scalings=scalings, title=title + " " + info_text,
                       show=show, block=True, duration=duration, show_options=False, show_scrollbars=False)
    else:
        fig = raw.plot(n_channels=len(channels), scalings=scalings, title=title + " " + info_text,
                       show=show, block=True, duration=duration)
    if save:
        fig.tight_layout(pad=0)
        fig.set_size_inches(6, 5)
        fig.savefig(path + title, bbox_inches='tight')
    plt.close(fig)

# --------------------------------------------------------------------------

def plot_abs(title, data, titles, figure_path, save, show, info_text, row_labels, col_labels, show_title=True):
    """
    Plots histogramms of the absolute values of the data
    """
    path = os.path.join(figure_path, "histograms")
    if not os.path.exists(path):
        os.makedirs(path)

    # matplotlib.rcParams.update({'font.size': 6})

    n_bins = 120

    print(np.array(data).shape)

    rows = len(data)
    cols = len(data[0])

    fig, ax = plt.subplots(rows, cols, sharey=True, sharex=True)

    for j in range(0, rows):
        for i in range(0, cols):

            X_abs1 = np.abs(util.reduce_one_dim(util.reduce_one_dim(data[j][i][0])))
            X_abs2 = np.abs(util.reduce_one_dim(util.reduce_one_dim(data[j][i][1])))
            X_abs3 = np.abs(util.reduce_one_dim(util.reduce_one_dim(data[j][i][2])))

            try:
                ax[j][i].hist([X_abs1, X_abs2, X_abs3], n_bins, histtype="step", stacked=False, label=titles[j][i])
                ax[j][i].legend()
                ax[j][i].semilogy()
            except:
                pass

            if j == 0:
                ax[j][i].set_title(col_labels[i], size=10)

            if i != 0:
                ax[j][i].set_ylabel("")

        ax[j][0].set_ylabel(row_labels[j], rotation=0, size=10, labelpad=15)
        ax[j][0].yaxis.label.set_size(10)

    fig.set_size_inches(8, 8)
    plt.subplots_adjust(wspace=0, hspace=0)

    if show_title:
        plt.suptitle(title + " " + info_text)
    name = os.path.join(path, title)
    if save: plt.savefig(name, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------

def plot_freq(title, X, fs, channels, figure_path, show=True, save=False, frange=(0, 65)):
    """
    Plots the power spectral density of the given data
    """

    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
    X = mne.io.RawArray(util.reduce_one_dim(np.copy(X)), info)
    fig = X.plot_psd(show=show, average=True, fmin=frange[0], fmax=frange[1])

    path = os.path.join(figure_path, "psd")
    if not os.path.exists(path):
        os.makedirs(path)

    if save:
        fig.set_size_inches(4.7, 5)
        fig.savefig(os.path.join(path, title), dpi=100)

    plt.close(fig)

# --------------------------------------------------------------------------
