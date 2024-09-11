import os
import numpy as np
import joblib
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.model_selection import train_test_split
from eeggan.Bachelorarbeit.Classifier import create
import eeggan.Bachelorarbeit.persistence as persistence
import eeggan.Bachelorarbeit.metrics as metrics
from eeggan.Bachelorarbeit.Classifier.ModelType import ModelType
from eeggan.data.dataset import Dataset, Data
import eeggan.Bachelorarbeit.GAN.generate as generate
import eeggan.Bachelorarbeit.util as util
import eeggan.Bachelorarbeit.constants as constants
from sklearn.utils import shuffle
from eeggan.data.preprocess.util import create_onehot_vector
import  eeggan.Bachelorarbeit.GAN.util as g_util
import eeggan.Bachelorarbeit.config as config

# --------------------------------------------------------------------------

def train_base_classifiers(sample_lengths, result_path, c_batch_size, c_iterations, c_epochs):
    """
    Train the base classifiers (on not augmented datasets)
    For each sample length this function notes which one of the iterations was the best
    and stores this result in a csv-file (best_base_classifier_.csv) and a txt-file.

    Parameters
    ----------
    sample_lengths: list of sample lengths for which to train the classifiers on
    result_path: Path to store the models to
    c_batch_size: the batch size for training
    c_iterations: the number of classifiers to train per sample length
    c_epochs: the number of epochs to train a single classifier
    """

    print("> training the base classifiers")
    print(sample_lengths)
    dataset_path = os.path.join(result_path, "datasets")

    output = "best base classifiers:\n"
    csv_header = ""
    csv_body = ""
    classifier_path = os.path.join(result_path, "classifiers/base")

    for i, sl in enumerate(sample_lengths):
        print(sl)

        ds_name = "all_subjects_" + str(sl) + "s.dataset"
        real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)

        best_cl_index = 0
        best_acc = 0

        for j in range(c_iterations):
            c_name = "all_subjects_" + str(sl) + "s_nr" + str(j+1)
            train_classifier(real_data, ModelType.SHALLOW, classifier_path, c_name, epochs=c_epochs,
                                   batch_size=c_batch_size)
            fbcspNet = persistence.load_model(classifier_path, c_name + ".FBCSPNet")
            preds = fbcspNet.predict_classes(real_data.test_data.X)
            acc = metrics.accuracy(preds, real_data.test_data.y)

            if acc > best_acc:
                best_acc = acc
                best_cl_index = j

        output += str(sl) + "s, Nr. " + str(best_cl_index+1) + ", acc: " + str(best_acc) + "\n"
        csv_header += str(sl) + ","
        csv_body += str(best_cl_index+1) + ","

    f = open(classifier_path + "/best_base_classifier.txt", "w+")
    f.write(output)

    f = open(classifier_path + "/best_base_classifier.csv", "w+")
    csv_header = csv_header[0:len(csv_header) - 1]
    csv_body = csv_body[0:len(csv_body) - 1]
    f.write(csv_header + "\n" + csv_body)

# --------------------------------------------------------------------------

def train_augm_classifiers(div, sl, bs, comb, result_path, fade, gan_stage, c_ratio_start, c_ratio_end,
                           filter_range, batch_size, epochs, use_clfr_th, pred_threshold, use_jsd_threshold,
                           jsd_threshold, iterations, metric):
    """
    Train the classifiers with the augmented datasets. The generated data is postprocessed.
    """

    print("iterations:", iterations)
    print("> training classifier on augmented data")

    dataset_path = os.path.join(result_path, "datasets")

    output = "\n\nepochs: " + str(epochs) + ", sl " + str(sl) + ", comb " + str(comb) + ", div " + str(div)

    ds_name = "all_subjects_" + str(sl) + "s.dataset"
    real_data = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)
    real_data_original = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)

    length = int(len(real_data.train_data.X)/div)

    if config.test_mode:
        length = 10

    X = real_data.train_data.X[:length]
    y = real_data.train_data.y[:length]
    y_onehot = real_data.train_data.y_onehot[:length]
    real_reduced = Dataset(Data(X, y, y_onehot), real_data.test_data)

    comb_str = "combined" if comb else "separate"

    base_classifier = g_util.get_best_base_classifier(sl, result_path)

    path = g_util.create_gan_path_from_params(result_path, sl, bs, fade)
    gan_path_c0 = os.path.join(path, "gan_c0")
    gan_path_c1 = os.path.join(path, "gan_c1")
    gan_path_comb = os.path.join(path, "gan_combined")
    base_classifier_path = os.path.join(result_path, "classifiers")
    augm_classifier_path = g_util.create_gan_path_from_params(base_classifier_path, sl, bs, fade)

    if comb:
        generator_c0, _ = persistence.load_gan(gan_path_comb + "/modules_stage_" + str(gan_stage - 1) + ".pt")
        generator_c1 = generator_c0
    else:
        generator_c0, _ = persistence.load_gan(gan_path_c0 + "/modules_stage_" + str(gan_stage - 1) + ".pt")
        generator_c1, _ = persistence.load_gan(gan_path_c1 + "/modules_stage_" + str(gan_stage - 1) + ".pt")

    generators = [generator_c0, generator_c1]

    if not use_jsd_threshold:
        jsd_threshold = None
        print("THRESHOLD IS NONE")
    else:
        print("threshold:", jsd_threshold)

    for ratio in range(0, c_ratio_end+1):
        output += "\nratio: " + str(ratio)
        num_samples = len(real_reduced.train_data.X)

        if not use_clfr_th:
            pred_th = None
            base_classifier = None
        else:
            pred_th = pred_threshold

        if ratio > 0:
            for label in range(0,2):

                real_c = util.get_single_class(real_data_original, label)

                data_batch = generate.generate_data_pp(generators[label], label, int(num_samples / 2), base_classifier, metric,
                                                real_c.train_data.X, constants.CHNS, constants.FS,
                                                classification_threshold=pred_th,
                                                freq_filter_range=filter_range,
                                                threshold=jsd_threshold,
                                                batch_size=batch_size,
                                                sl=sl)

                X = np.vstack((X, data_batch.X))
                y = np.hstack((y, data_batch.y))

        if ratio > c_ratio_start-1:
            X, y = shuffle(X, y)
            augm_data = Data(X, y, create_onehot_vector(y, 2))
            augm_dataset = Dataset(augm_data, real_data.test_data)

            print("div:", div, "("+str(length)+")", "total:", len(augm_data.X) ,", sl:", sl, ", comb:", comb, ", ratio", "1:" + str(ratio))

            if comb: output += "\ncombined: "
            else: output += "\nseperate: "

            for i in range(iterations):
                aug_c_name = "aug_" + comb_str + "_" + str(ratio) + "_div_" + str(div) + "_nr" + str(i + 1)
                output += train_classifier(augm_dataset, ModelType.SHALLOW, augm_classifier_path, aug_c_name, epochs=epochs)

    return output

# --------------------------------------------------------------------------

def train_classifier(dataset, model_type, model_path, filename, epochs=100, lr=0.015, shuffle=True, save=True, batch_size=50, return_model=False):
    """
    creates and trains the classifier of model_type

    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    X = dataset.train_data.X
    y = dataset.train_data.y
    X_test = dataset.test_data.X
    y_test = dataset.test_data.y

    if config.test_mode:
        print("\nTHE SIZE OF THE DATASET IS REDUCED BECAUSE THE TESTMODE IS ENABLED (config.ini)\n")
        X = X[:10]
        y = y[:10]

    average = []

    X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, shuffle=shuffle)

    train_set = SignalAndTarget(X, y)
    valid_set = SignalAndTarget(X_val, y_val)
    test_set = SignalAndTarget(X_test, y_test)

    n_classes = 2
    in_chans = train_set.X.shape[1]
    input_time_length = train_set.X.shape[2]

    wd = 0
    conv = 'auto'

    model = create.create_classifier(model_type=model_type, n_classes=n_classes, in_chans= in_chans,
                                     lr = lr, weight_decay= wd, final_conv_length=conv, input_time_length=input_time_length)
    model.fit(train_set.X, train_set.y, epochs=epochs, batch_size=batch_size, scheduler='cosine',
                input_time_length=input_time_length, validation_data=(valid_set.X, valid_set.y))

    result_train = model.evaluate(train_set.X, train_set.y)
    result_test = model.evaluate(test_set.X, test_set.y)
    average.append(1.0-result_train['misclass'])
    average.append(1.0-result_test['misclass'])

    if save:
        file_ext = get_file_ext(model_type)
        joblib.dump(model, os.path.join(model_path, filename + '.' + file_ext), compress=True)

    if not return_model:
        print("\ttrain_data:" + str(average[0]) + "| test_data:" + str(average[1]))
        return "\n\ttrain_data:" + str(average[0]) + "| test_data:" + str(average[1])
    else:
        return model

# --------------------------------------------------------------------------

def get_file_ext(model_type):
    """
    Returns the file extension for the model type
    """

    if model_type == ModelType.SHALLOW:
        return "FBCSPNet"
    elif model_type == ModelType.DEEP:
        return "Deep4Net"
    else:
        return ""

# --------------------------------------------------------------------------
