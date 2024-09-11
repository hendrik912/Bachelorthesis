import os
import eeggan.Bachelorarbeit.persistence as persistence

# --------------------------------------------------------------------------

def get_metrics_list(scores_dict):
    """
    iterate through the scores-dictionary to find out which metrics were used
    and return them in a list.
    """

    metrics = []
    for p in scores_dict:
        for sl in scores_dict[p]:
            for m in scores_dict[p][sl]:
                if m != "samples":
                    metrics.append(m)
            break
        break
    return metrics

# --------------------------------------------------------------------------

def create_gan_path_from_params(base_result_path, sample_len_in_sec, batch_size, fade):
    """
    Takes some GAN parameters which identify a particular GAN and returns a path to it.
    """

    path = os.path.join(base_result_path, "GANs", "all_subjects_" + str(sample_len_in_sec) + "s_bs" + str(batch_size))
    path = path + "_fade" if fade else path
    return path

# --------------------------------------------------------------------------

def get_best_base_classifier(sample_len_in_sec, base_result_path):
    """
    Gets the best of the previously trained base classifiers (trained on not-augmented data)

    Parameters
    ----------
    sample_len_in_sec: The sample lengths in seconds
    base_result_path: The path to the where the overall results are stored

    Returns
    -------
    classifier

    """

    base_classifier_path = os.path.join(base_result_path, "classifiers/base")
    path_best_base = base_classifier_path + "/best_base_classifier.csv"
    columns, values = persistence.read_csv(path_best_base)

    index = 1
    if columns is not None:
        for k, col in enumerate(columns):
            if int(col) == sample_len_in_sec:
                index = values[0][k]
    else:
        print("error reading csv file: ", path_best_base)

    classifier_name = "all_subjects_" + str(sample_len_in_sec) + "s_nr" + str(index) + ".FBCSPNet"
    base_classifier = persistence.load_model(base_classifier_path, classifier_name)

    return base_classifier

# --------------------------------------------------------------------------
