# This file contains the functions to load and store data, models and other files

import os
import mne
import joblib
import torch
import numpy as np
import pandas as pd
import eeggan.Bachelorarbeit.preprocessing as preprocessing
import eeggan.Bachelorarbeit.constants as constants
# import eeggan.Bachelorarbeit.config as config
from eeggan.data.preprocess.util import create_onehot_vector
from sklearn.model_selection import train_test_split
from eeggan.data.dataset import Dataset, Data
from configparser import ConfigParser

# --------------------------------------------------------------------------

def create_datasets(data_path, result_path, sample_lengths, data_filter, filter_range):
    """
    Goes through every sample lengths and creates and stores the Datasets

    Parameters
    ----------
    sample_lengths: list of sample lengths

    """

    dataset_path = os.path.join(result_path, "datasets")
    for sl in sample_lengths:
        print("  > " + str(sl) + "s")
        ds_name = "all_subjects_" + str(sl) + "s.dataset"
        create_attention_dataset_from_all_subjects(data_path, dataset_path, ds_name, filter=data_filter,
                                                               filter_range=filter_range, duration=sl)

# --------------------------------------------------------------------------


def create_attention_dataset_from_all_subjects(data_path, dataset_path, filename, duration, filter, filter_range=(0,30)):
    """
    Creates a dataset from multiple files from the attention data and stores it on the disk

    Parameters
    ----------
    data_path: Path to the fif-files
    dataset_path: Path to store the Datasets to
    filename: Name of the Dataset when stored
    duration: The sample length in seconds
    filter: Whether to filter the data or not
    filter_range: Range of bandpass filter to apply

    """

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    paths = [
        data_path + "iea_01/iea_01-epo.fif", data_path + "iea_16/iea_16-epo.fif", data_path + "iea_22/iea_22-epo.fif",
        data_path + "iea_34/iea_34-epo.fif", data_path + "iea_35/iea_35-epo.fif", data_path + "iea_36/iea_36-epo.fif",
        data_path + "iea_51/iea_51-epo.fif", data_path + "iea_63/iea_63-epo.fif", data_path + "iea_72/iea_72-epo.fif",
        data_path + "iea_77/iea_77-epo.fif", data_path + "iea_79/iea_79-epo.fif", data_path + "iea_87/iea_87-epo.fif",
        data_path + "iea_94/iea_94-epo.fif"
    ]

    dataset_all = None
    chns = ['CZ', 'FP2', 'F3', 'FT7', 'C3', 'C4', 'FT8', 'P3', 'PO7', 'PO8', 'OZ']

    for i, path in enumerate(paths):
        data, dataset = load_epoched_fif(path, chns, filter, filter_range, duration)
        if i == 0:
            dataset_all = dataset
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAa")
            print(len(dataset_all.train_data.X), len(dataset_all.test_data.X))

            dataset_all.train_data.X = np.vstack((dataset_all.train_data.X, dataset.train_data.X))
            dataset_all.train_data.y = np.hstack((dataset_all.train_data.y, dataset.train_data.y))
            dataset_all.test_data.X = np.vstack((dataset_all.test_data.X, dataset.test_data.X))
            dataset_all.test_data.y = np.hstack((dataset_all.test_data.y, dataset.test_data.y))

    dataset_all.train_data.y_onehot = create_onehot_vector(dataset_all.train_data.y, 2)
    dataset_all.test_data.y_onehot = create_onehot_vector(dataset_all.test_data.y, 2)

    joblib.dump(dataset_all, os.path.join(dataset_path, filename), compress=True)

# --------------------------------------------------------------------------

def string_to_file(string, path, title):
    """
    Stores a string into a file

    Parameters
    ----------
    string: the content
    path: location to store the file
    title: filename
    """
    if not os.path.exists(path):
        os.makedirs(path)

    f = open(path + "/" + title, "w+")
    f.write(string)

# --------------------------------------------------------------------------

def read_csv(path):
    """
    Read a csv file and return a DataFrame

    Parameters
    ----------
    path: path to csv-file

    Returns
    -------
    DataFrame

    """
    try:
        df = pd.read_csv(path, sep=',')
        return (df.columns.values, df.values)
    except:
        print("Failed to load ", path)
        return None, None

# --------------------------------------------------------------------------

def load_prepared_data_from_dataset(dataset_path, filename):
    """
    loads the data from a dataset which has been stored on the hard disk

    Parameters
    ----------
    dataset_path: path to dataset
    filename: name of file

    Returns
    -------
    the Dataset-object

    """
    path = os.path.join(dataset_path, filename)

    try:
        dataset = joblib.load(path)
        return dataset
    except:
        print(">> could not load dataset")
        print(path)
        return None

# --------------------------------------------------------------------------

def load_epoched_fif(data_path, channels, filter, bandpass_range, duration):
    """
    Load the epoched fif data from the hard disk

    Parameters
    ----------
    data_path: path to the data
    channels: the channels to select
    filter: whether to filter the data or not
    abs_threshold: not used anymore
    bandpass_range: the min and max for bandpass filtering
    duration: the sample length

    Returns
    -------
    MNE Epochs-Object and a Dataset-Object with the data

    """

    epoched = mne.read_epochs(data_path, preload=True)

    if channels != None:
        chns = epoched.info["ch_names"]
        indices = []
        for i, ch in enumerate(chns):
            if ch in channels:
                indices.append(i)

        epoched.pick_channels([epoched.ch_names[pick] for pick in indices])

    if filter:
        epoched.filter(l_freq=bandpass_range[0], h_freq=bandpass_range[1])

    X = preprocessing.standardize(epoched.get_data())
    y = (epoched.events[:, 2] - 2).astype(np.int64)

    for idx, i in enumerate(y):
        if i == 5:
            y[idx] = 1
        else:
            y[idx] = 0

    length = 6144  # 6501

    X = X[:, :, :length]
    y = y[:length]

    """
    1s
      6144 / 12 = 512
      512 / 500 ~ 1s

    2s 
      6144 / 6 = 1024
      1024 / 500 ~ 2s

    3s 
      6144 / 4 = 1536
      1536 / 500 ~ 3s

    4s 
      6144 / 3 = 2048
      2048 / 500 ~ 4s

    6s
      6144 / 2 = 3072
      3072 / 500 ~ 6s

    12s
      6144 / 500 ~ 12s

    """

    div = 1
    if duration == 1:
        div = 12
    elif duration == 2:
        div = 6
    elif duration == 3:
        div = 4
    elif duration == 4:
        div = 3
    elif duration == 6:
        div = 2
    elif duration == 12:
        div = 1
    else:
        print("> load.load_epoched_fif: only the durations 1s, 2s, 3s, 4s, ",
              "6s, 12s are valid, chose 12s as default")

    split = np.array(np.split(X, div, axis=2))

    X_new = split[0]
    y_new = np.copy(y)

    for i in range(1, len(split)):
        X_new = np.vstack((X_new, split[i]))
        y_new = np.hstack((y_new, np.copy(y)))

    X, X_test, y, y_test = train_test_split(X_new, y_new, test_size=0.25, shuffle=True)

    train_data = Data(X, y, create_onehot_vector(y, 2))
    test_data = Data(X_test, y_test, create_onehot_vector(y_test, 2))
    dataset = Dataset(train_data, test_data)

    return epoched, dataset

# --------------------------------------------------------------------------


def load_gan(model_path):
    """
    load the gan from the hard disk

    Parameters
    ----------
    model_path: path to model including the name

    Returns
    -------
    The generator and discriminator
    """

    model_modules = torch.load(model_path)
    generator = model_modules["generator"]
    discriminator = model_modules["discriminator"]
    return generator, discriminator

# --------------------------------------------------------------------------


def load_model(path, filename):
    """
    Loads the model from the hard disk

    Parameters
    ----------
    path: Path to the model
    filename: Name of the file

    Returns
    -------
    The model
    """
    return joblib.load(os.path.join(path, filename))

# --------------------------------------------------------------------------


def load_config():
    """
    Loads the config file and returns it

    """
    co = ConfigParser()
    co.read("../../config.ini")

    if len(co.sections()) == 0:
        co.read("config.ini")

    return co

# --------------------------------------------------------------------------