import os
import numpy as np
import eeggan.Bachelorarbeit.GAN.util as g_util
import eeggan.Bachelorarbeit.persistence as persistence
import eeggan.Bachelorarbeit.Classifier.plot as c_plot
import eeggan.Bachelorarbeit.metrics as metrics
from eeggan.Bachelorarbeit.Classifier.ClassifierTable import ClassifierTable
from eeggan.Bachelorarbeit.Classifier.TableType import TableType
import eeggan.Bachelorarbeit.config as config
from sys import platform

# --------------------------------------------------------------------------

def evaluate(parameters, fake_real_ratio, iterations, base_path, calc_metrics=False):
    """
    This function handles the evaluation of the classifiers, creates the tables and plots.

    Parameters
    ----------
    parameters: defines which classifiers should be evaluated
    fake_real_ratio: gives the upper limit of the ratio of fake to real on which classifiers were trained
    iterations: gives the number of iterations each combination of parameters and ratio was trained
    base_path: the base path of the results (and models)
    BA_path: the path to store the images and tables to
    calc_metrics: whether or not to calculate the metrics new
    """

    result_dict_path = os.path.join(base_path, "eval", "classifiers", "result_dict.npy")
    BA_path = os.path.join(base_path, "eval", "classifiers")
    BA_path_imgs = os.path.join(BA_path, "images")
    BA_path_tbls = os.path.join(BA_path, "tables")
    if not os.path.exists(BA_path_imgs): os.makedirs(BA_path_imgs)
    if not os.path.exists(BA_path_tbls): os.makedirs(BA_path_tbls)

    show = True
    make_tables = True
    plot_metrics = True
    plot_conf_mat = False

    show_title = False

    print(base_path)

    if calc_metrics:
        print("calc classifier result dict")
        result_dict = fill_classifier_result_dict(parameters, fake_real_ratio, iterations, base_path)
        print("Save the classifier result dict to ", result_dict_path)
        np.save(result_dict_path, result_dict)
    else:
        try:
            result_dict = load_result_dict(result_dict_path)
        except:
            print("\nNO RESULT-DICT FOUND.", "\nset 'c_calc_metrics = True' in config.ini\n"
                                             "or move an existing result-dict into the path:\n",
                                              result_dict_path)
            return

    tts = [TableType.BY_SAMPLE_LEN, TableType.BY_DIVISION, TableType.BY_RATIO]

    if make_tables:
        for tt in tts:
            tables = get_table_of_metrics(parameters, result_dict, tt, fake_real_ratio)

            for table in tables:
                latex = table.to_latex()
                filename = table.get_title()
                # print(latex)
                persistence.string_to_file(latex, BA_path_tbls, filename)

    # ---------------------------

    if plot_metrics:
        c_plot.plot_classifer_results(result_dict, BA_path_imgs, show=show, show_title=show_title)

    if plot_conf_mat:
        # plot confusion matrix across n iterations for specific div, sl, type (comb/sep/base), ratio
        if config.test_mode:
            param_for_selected = [
                (8, 4, "separate", 1),
                (1, 3, "separate", 1),
                (1, 3, "separate", 1),
                (1, 4, "separate", 1),
            ]

        title = "confusion_mat_"
        data = []
        row = []

        for p in param_for_selected:
            div = p[0]
            sl = p[1]
            type = p[2]
            ratio = p[3]
            subplot_title = "Fenstergröße: " + str(sl) + ", Verhältnis 1:" + str(ratio) + ", Datensatz 1/" + str(div)

            if len(row) < 2:
                row.append((result_dict[div][sl][type][ratio]["raw"], subplot_title))
            else:
                data.append(row)
                row = []
                row.append((result_dict[div][sl][type][ratio]["raw"], subplot_title))

        data.append(row)
        title = title[:-1]

        c_plot.plot_confusion_matrices(title, data, BA_path_imgs, save=True, show=show, show_title=False, info_text="")

# --------------------------------------------------------------------------

def get_results_from_result_dict(result_dict, sample_len, division, ratio, type, classification_metric, stat_metric):
    """
    a function to more conveniently get the data from the result dict
    """
    return result_dict[division][sample_len][type][ratio][classification_metric][stat_metric]

# --------------------------------------------------------------------------

def get_table_of_metrics(parameters, result_dict, table_type, max_ratio):
    """
    Takes the results from result_dict and puts them into the table given by the table_type and returns the table
    (ClassifierTable)
    """

    metrics = ["accuracy", "fscore", "precision", "recall"]
    tables = []
    max_ratio += 1
    twoDec = "{:.2f}"
    divisions = [1, 8]# [1, 2, 4, 8]
    sample_lens = [p[0] for p in parameters]
    ratios = [i for i in range(0, max_ratio)]

    for metric_ in metrics:
        if table_type == TableType.BY_SAMPLE_LEN:
            #sample len 1..n
            # ratio * div

            table = ClassifierTable(1, 1, table_type)
            table.array = []
            table.set_x_axis(divisions)
            table.y_axis_label = "Ratio"
            table.x_axis_label = "Division"
            table.title = "tt_" + metric_ + "_" + str(table_type.value) + "_ratio_x_div"

            table.caption = ""
            outer = sample_lens
            middle = ratios
            inner = divisions

        elif table_type == TableType.BY_DIVISION:
            #division 1..n
            #  ratio * sl

            table = ClassifierTable(1, 1, table_type)
            table.array = []
            table.set_x_axis(sample_lens)
            table.x_axis_label = "W.L."
            table.y_axis_label = "Ratio"
            table.title = "tt_" + metric_ + "_" +  str(table_type.value) + "_ratio_x_sl"
            table.caption = ""
            outer = divisions
            middle = ratios
            inner = sample_lens
        else:
            # ratio 0..n
            #   sl * div

            table = ClassifierTable(1, 1, table_type)
            table.array = []
            table.set_x_axis(divisions)
            table.x_axis_label = "Division"
            table.y_axis_label = "W.L."
            table.title = "tt_" + metric_ + "_" + str(table_type.value) + "_sl_x_div"
            table.caption = ""
            outer = ratios
            middle = sample_lens
            inner = divisions

        for j, outer_ in enumerate(outer):
            offset = 2

            if j!=0:
                table.array.append([])
                for i in range(0, len(inner)+offset):
                    table.array[len(table.array) - 1].append(" ")

            for k, middle_ in enumerate(middle):
                row = []

                if table_type == TableType.BY_SAMPLE_LEN:
                    if k==0: row.append(str(outer_) + "s")
                    else: row.append(" ")
                    row.append("1:" + str(middle_))
                elif table_type == TableType.BY_DIVISION:
                    if k == 0: row.append(str(outer_))
                    else: row.append(" ")
                    row.append("1:" + str(middle_))
                else:
                    if k == 0: row.append("1:" + str(outer_))
                    else: row.append(" ")
                    row.append(str(middle_) + "s")

                for i, inner_ in enumerate(inner):
                    if table_type == TableType.BY_SAMPLE_LEN:
                        s = outer_
                        d = inner_
                        r = middle_
                    elif table_type == TableType.BY_DIVISION:
                        d = outer_
                        s = inner_
                        r = middle_
                    else:
                        r = outer_
                        d = inner_
                        s = middle_

                    type = "base" if r == 0 else "separate"

                    avg_acc = get_results_from_result_dict(result_dict, s, d, r, type, metric_, "avg")
                    min_acc = get_results_from_result_dict(result_dict, s, d, r, type, metric_, "min")
                    max_acc = get_results_from_result_dict(result_dict, s, d, r, type, metric_, "max")
                    row.append(str(twoDec.format(avg_acc)) +  " (" +
                               str(twoDec.format(min_acc)) + ", " + str(twoDec.format(max_acc)) + ")")

                table.array.append(row)
            tables.append(table)
    return tables

# --------------------------------------------------------------------------

def load_result_dict(path):
    """
    load the result dicitionary from the disk and return it.
    """
    return np.load(path, allow_pickle=True).item()

# --------------------------------------------------------------------------

def create_classifier_result_dict(sample_lengths_in_sec, fake_real_ratio):
    """
    Creates a dictionary in which the results of all the classifiers can be stored,
    so that these dont have to be recalculated every time I want to change the way
    the results are plotted.

    Returns
    -------
    empty result dicitionary

    """

    type = ["base", "separate"]#, "combined"]
    metrics1 = ["accuracy", "precision", "recall", "fscore"]
    metrics2 = ["avg", "max", "min"]
    ratios = [(l + 1) for l in range(fake_real_ratio)]

    """
    Structure of the dictionary:
    
    data_divisions (/1, /2, /4 /8)
      sample_len_in_sec (1s, 2s, ...)
        base/comb/sep
          1/2/..  (ratios, 0 if base)
            acc/rec/prec..
              avg/min/max/..
                0.5
            raw (predictions across iterations) 
              [[1,0,1,0,1..],[1,0,1,0,1..], ...] 
    """
    result_dict = {}
    data_divisions = [1,2,4,8]

    for div in data_divisions:
        result_dict[div] = {}

        for sl in sample_lengths_in_sec:
            result_dict[div][sl] = {}
            for t in type:
                result_dict[div][sl][t] = {}

                if t == "base":
                    _ratios = [0]
                else:
                    _ratios = ratios

                for r in _ratios:
                    result_dict[div][sl][t][r] = {}
                    for m1 in metrics1:
                        result_dict[div][sl][t][r][m1] = {}
                        for m2 in metrics2:
                            result_dict[div][sl][t][r][m1][m2] = None
                    result_dict[div][sl][t][r]["raw"] = {}
    return result_dict

# --------------------------------------------------------------------------

def fill_classifier_result_dict(parameters, fake_real_ratio, iterations, base_path):
    """
    fills the result dictionary with the evaluation results of the classifiers

    Parameters
    ----------
    parameters: param to determine which classifiers to evaluate
    fake_real_ratio: passed on to create_classifier_result_dict(..)
    iterations: passed on to get the metrics from get_classifer_pred_metrics(..)
    base_path: base path of where to find the classifiers

    Returns
    -------
    filled result dictionary

    """

    sample_lengths = [t[0] for t in parameters]
    batch_size = 128
    fade = True

    result_dict = create_classifier_result_dict(sample_lengths, fake_real_ratio)
    dataset_path = os.path.join(base_path, "datasets")
    base_path = os.path.join(base_path, "classifiers")

    for div in result_dict:
        print("div", div)
        for sl in result_dict[div]:
            print(" sl", sl)
            ds_name = "all_subjects_" + str(sl) + "s.dataset"

            for type in result_dict[div][sl]:
                print("  type", type)
                #if type != "base":
                path = g_util.create_gan_path_from_params(base_path, sl, batch_size, fade)

                #else:
                #    path = base_path
                #    print(base_path)

                real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)

                for ratio in result_dict[div][sl][type]:
                    print("    ratio", ratio)

                    accs, precs, recs, fscs, raw_preds, labels = \
                        get_classifer_pred_metrics(real_data.test_data, sl, type, int(ratio), iterations, div, path)


                    for m1 in result_dict[div][sl][type][ratio]:
                        is_raw = False
                        preds = []

                        if m1 == "accuracy":
                            preds = accs
                        elif m1 == "recall":
                            preds = recs
                        elif m1 == "precision":
                            preds = precs
                        elif m1 == "fscore":
                            preds = fscs
                        elif m1 == "raw":
                            is_raw = True
                            result_dict[div][sl][type][ratio][m1] = (raw_preds, labels)




                        if not is_raw:
                            for m2 in result_dict[div][sl][type][ratio][m1]:
                                if m2 == "avg":
                                    result_dict[div][sl][type][ratio][m1][m2] = np.average(preds)

                                elif m2 == "min":
                                    result_dict[div][sl][type][ratio][m1][m2] = np.min(preds)
                                elif m2 == "max":
                                    result_dict[div][sl][type][ratio][m1][m2] = np.max(preds)
    return result_dict

# --------------------------------------------------------------------------

def get_classifer_pred_metrics(test_data, sample_len, type, fake_real_ratio, iterations, div, path):
    """
    Get the predictions of the classifiers given by the type, iterations, ratio and div and
    calculate metrics (accuracy, precision, recall, fscore) and store them in lists

    Parameters
    ----------
    test_data: data to predict on
    sample_len: only used for printing on the console
    type: type of the classifier (Shallow FBCSPNet or Deep4Net)
    fake_real_ratio, iterations, div: used to determine which classifier to load
    path: path of the classifiers

    Returns
    -------
    lists with metrics

    """

    prefix = "aug_" + ("separate" if type == "separate" or type == "base" else "combined") + "_" + str(
        fake_real_ratio) + "_div_" + str(div) + "_nr"

    suffix = ".FBCSPNet"

    accs = []
    precs = []
    recs = []
    fscs = []
    raw_preds = np.array([])
    real_labels = np.array([])

    for i in range(iterations):

        if div == 0:
            print(div, sample_len, type, fake_real_ratio, iterations)
            print(prefix)
            print("...............")

        c_name = prefix + str(i + 1) + suffix
        classifier = persistence.load_model(path, c_name)

        preds = classifier.predict_classes(test_data.X)

        raw_preds = np.hstack((raw_preds, preds))
        real_labels = np.hstack((real_labels, test_data.y))

        acc = metrics.accuracy(preds, test_data.y)
        prec = metrics.precision(preds, test_data.y, 0)
        rec = metrics.recall(preds, test_data.y, 0)
        fsc = metrics.fscore(preds, test_data.y, 0)

        print("\t", acc)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        fscs.append(fsc)

    return accs, precs, recs, fscs, raw_preds, real_labels

# --------------------------------------------------------------------------