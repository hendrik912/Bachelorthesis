import os
import numpy as np
import eeggan.Bachelorarbeit.persistence as persistence
import eeggan.Bachelorarbeit.constants as constants
from eeggan.Bachelorarbeit.GAN import generate, metrics
import eeggan.Bachelorarbeit.GAN.util as g_util
import eeggan.Bachelorarbeit.util as util
from eeggan.data.preprocess.util import create_onehot_vector
import eeggan.Bachelorarbeit.GAN.plot as g_plot
from numpy.random import random
from eeggan.Bachelorarbeit.GAN.MetricType import MetricType
from sys import platform
from eeggan.data.dataset import Data

# --------------------------------------------------------------------------

def evaluate(result_path, num_samples, parameters, eval_times, threshold, pred_th, calc_metrics, metric):
    """
    This function handles the evaluation of the GANs and creates tables, as well as plots

    Parameters
    ----------
    result_path: base path where the GANs are stored
    num_samples: number samples to calculate the metrics on
    parameters: list of tuples with param that identify a GAN
    eval_times: the number of times to repeat the process of calculating the metrics
    threshold: threshold for the rejection sampling of the generated samples (jsd)
    pred_th: classificationthreshold for the rejection sampling of the generated samples
    calc_metrics: whether or not to calculate the results, if false the result_dict will be loaded from the disk
    metric: function of JSD, SWD, ... (only jsd is actually used)
    metric_type: the type of the metric (MetricType)
    """

    result_dict_path = os.path.join(result_path, "eval", "gan")
    BA_path = result_dict_path
    BA_path_imgs = os.path.join(BA_path, "images")
    result_dict_name = "scores_dict.npy"

    if not os.path.exists(result_dict_path):
        os.makedirs(result_dict_path)
    if not os.path.exists(BA_path_imgs):
        os.makedirs(BA_path_imgs)

    result_dict_path = os.path.join(result_dict_path, result_dict_name)

    if calc_metrics:
        print("> Calc Metrics")
        gan_dict = create_gans_result_dict(result_path, num_samples, eval_times, parameters, threshold, pred_th, metric)
        np.save(result_dict_path, gan_dict)

    try:
        result_dict = load_scores_dict(result_dict_path)
    except:
        print("\nNO RESULT-DICT FOUND.", "\nset 'gan_calc_metrics = True' in config.ini\n"
                                         "or move an existing result-dict into the path:\n",
                                         result_dict_path)
        return

    print("> Plot Metrics")

    print("JSD")
    #g_plot.plot_metrics(result_dict, BA_path_imgs, MetricType.JSD)
    #g_plot.plot_metrics_paper_jsd(result_dict, BA_path_imgs, MetricType.JSD)

    #print("SWD")
    #g_plot.plot_metrics(result_dict, BA_path_imgs, MetricType.SWD)

    print("IS")
    ##g_plot.plot_metrics(result_dict, BA_path_imgs, MetricType.IS)
    #g_plot.plot_metrics_paper_is(result_dict, BA_path_imgs, MetricType.IS)

    print("psd")
    # g_plot.plot_psd(result_dict, result_path, BA_path_imgs, parameters)
    g_plot.plot_psd(result_dict, result_path, BA_path_imgs, parameters)

    #print("psd channels")
    #g_plot.plot_psd_channels(result_dict, result_path, BA_path_imgs, parameters)
    #g_plot.plot_psd_channels(result_dict, result_path, BA_path_imgs, parameters)

    print("Raw")
    # g_plot.plot_gan_figures(result_dict, result_path, BA_path_imgs, num_samples, parameters, show=False)

    print("histo & boxplots (may take some time)")
    plot_boxplots = True
    plot_histograms = True
    #g_plot.plot_absolute_values(result_dict, result_path, BA_path_imgs, parameters, boxplots=plot_boxplots,
    #                            histograms=plot_histograms, show=True)

# --------------------------------------------------------------------------

def load_scores_dict(path):
    """
    load the result dictionary from the disk and return it
    """
    return np.load(path, allow_pickle=True).item()

# --------------------------------------------------------------------------

def get_values(scores_dict, base, sample_len, comb, metric, postprocessed, class_):
    """
    This function serves as a more convenient way to retrieve data from the result dictionary
    -------

    """
    if base:
        return scores_dict["base"][sample_len][metric][class_]
    else:
        p = (128, True, comb, postprocessed)
        return scores_dict[p][sample_len][metric][class_]

# --------------------------------------------------------------------------

def create_gans_result_dict(base_result_path, num_samples, times, parameters, threshold, pred_th, metric):
    """
    Create the dictionary into which the results of the gan evaluation gets stored

    Parameters
    ----------
    base_result_path: path of the results (including the GANs)
    num_samples: number samples to calculate the metrics on
    times: the number of times to repeat the process of calculating the metrics
    parameters: list of tuples with param that identify a GAN
    threshold: threshold for the rejection sampling of the generated samples (jsd)
    pred_th: classificationthreshold for the rejection sampling of the generated samples
    metric: JSD, SWD, ...

    Returns
    -------
    empty results dict
    """

    metric_dict = {}
    metric_dict[MetricType.IS] = metrics.inception_score
    metric_dict[MetricType.JSD] = metrics.jsd
    metric_dict[MetricType.SWD] = metrics.sliced_wasserstein_dist

    scores_dict = {}

    metrics_list = [MetricType.IS, MetricType.JSD, MetricType.SWD]
    lengths = []

    gan_param = ["base"]
    num_samples_ = num_samples

    for p in parameters:
        # sl, bs, fade, comb, pp
        sl = p[0]
        bs = p[1]
        fade = p[2]
        comb = p[3]
        pp = p[4]
        gan_param.append((bs, fade, comb, pp))
        lengths.append(sl)
    lengths.sort()

    classes = ["all", "0", "1"]
    """
    Dict structure:    
      gan_param (tuple to identify the correct gan)
        sample_length_in_s
          samples (Dataset)
          metrics (jsd, kld, is)
            classes (all, 0, 1)
               scores (e.g. [0.34, 0.28, 0.08, ...])    
    """
    for p in gan_param:
        scores_dict[p] = {}
        for sl in lengths:
            scores_dict[p][sl] = {}
            scores_dict[p][sl]["samples"] = None
            for m in metrics_list:
                scores_dict[p][sl][m] = {}
                for c in classes:
                    scores_dict[p][sl][m][c] = []

    for i, p in enumerate(scores_dict):
        base = True if p == "base" else False

        print("-"*100)
        print(str(i+1), "/", len(scores_dict), p)

        bs = fade = comb = pp = None
        if not base:
            # (bs, fade, comb, pp)
            bs = p[0]
            fade = p[1]
            comb = p[2]
            pp = p[3]

        for sl in scores_dict[p]:
            print("\tsl:", sl, "comb:", comb, "pp:", pp)

            generator_c0_s5 = None
            generator_c1_s5 = None
            base_classifier = None

            if not base:
                path = g_util.create_gan_path_from_params(base_result_path, sl, bs, fade)
                gan_path_c0 = os.path.join(path, "gan_c0")
                gan_path_c1 = os.path.join(path, "gan_c1")
                gan_path_comb = os.path.join(path, "gan_combined")

                if comb:
                    generator_c0_s5, _ = persistence.load_gan(gan_path_comb + "/modules_stage_5.pt")
                    generator_c1_s5 = generator_c0_s5
                else:
                    generator_c0_s5, _ = persistence.load_gan(gan_path_c0 + "/modules_stage_5.pt")
                    generator_c1_s5, _ = persistence.load_gan(gan_path_c1 + "/modules_stage_5.pt")

                base_classifier = g_util.get_best_base_classifier(sl, base_result_path)

                if base_classifier is None or pred_th is None:
                    print("BASE CLASSIFIER IS NONE")

            ds_name = "all_subjects_" + str(sl) + "s.dataset"
            dataset_path = os.path.join(base_result_path, "datasets")
            real_data_all = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)

            if num_samples_ == "all":
                num_samples_ = len(real_data_all.train_data.X)
                print(num_samples_)

            for iter in range(0, times):
                print(3*"\t", iter, " / ", times)
                indices = np.arange(0, len(real_data_all.train_data.X))
                choice = np.random.choice(indices, num_samples_ * 2)

                real_train_all = Data(real_data_all.train_data.X[choice], real_data_all.train_data.y[choice],
                                      real_data_all.train_data.y_onehot[choice])

                real_train_0 = util.get_single_class(real_train_all, 0)
                real_train_1 = util.get_single_class(real_train_all, 1)
                train_dict = {"all" : real_train_all, "0" : real_train_0, "1" : real_train_1}

                indices = np.arange(0, len(real_data_all.test_data.X))
                choice = np.random.choice(indices, num_samples_ * 2)
                real_test_all = Data(real_data_all.test_data.X[choice], real_data_all.test_data.y[choice],
                                      real_data_all.test_data.y_onehot[choice])
                real_test_0 = util.get_single_class(real_test_all, 0)
                real_test_1 = util.get_single_class(real_test_all, 1)
                test_dict = {"all" : real_test_all, "0" : real_test_0, "1" : real_test_1}

                if not base:
                    if not pp:
                        fake_all = generate.generate_data_separate_gans(generator_c0_s5, generator_c1_s5, num_samples_ * 2,
                                                                    batch_size=50)
                        fake_0 = util.get_single_class(fake_all, 0)
                        fake_1 = util.get_single_class(fake_all, 1)
                    else:
                        fake_all = generate.generate_data_separate_gans_pp(generator_c0_s5, generator_c1_s5, 0, 1,
                                                                          num_samples_, base_classifier, metric,
                                                                          real_data_all, constants.CHNS, constants.FS,
                                                                          sl=sl, classification_threshold=pred_th,
                                                                          freq_filter_range=(0, 30), threshold=threshold,
                                                                          batch_size=50)
                        fake_0 = util.get_single_class(fake_all, 0)
                        fake_1 = util.get_single_class(fake_all, 1)

                    fake_dict = {"all": fake_all, "0": fake_0, "1": fake_1}

                    if scores_dict[p][sl]["samples"] is None:
                        scores_dict[p][sl]["samples"] = fake_all
                    else:
                        ds = scores_dict[p][sl]["samples"]
                        ds.X = np.vstack((ds.X, fake_all.X))
                        ds.y = np.hstack((ds.y, fake_all.y))
                        ds.y_onehot = create_onehot_vector(ds.y, 2)

                for m in scores_dict[p][sl]:
                    if m == "samples":
                        continue

                    for c in scores_dict[p][sl][m]:
                        if not base:
                            res = metric_dict[m](fake_dict[c].X, test_dict[c].X, comb, sl)
                            scores_dict[p][sl][m][c].append(res)
                        else:
                            res = metric_dict[m](train_dict[c].X, test_dict[c].X, comb, sl)
                            scores_dict[p][sl][m][c].append(res)
    return scores_dict

# --------------------------------------------------------------------------