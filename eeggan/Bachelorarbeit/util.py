# This file contains utility functions that are used across the files

from mne.time_frequency import psd_welch
import random
import mne
import numpy as np
from eeggan.data.dataset import Dataset, Data
from eeggan.data.preprocess.util import create_onehot_vector
from PIL import Image


# --------------------------------------------------------------------------

def reduce_one_dim(data):
    """
    Reduces the dimension of a nd-array of data
    (n,m,k) -> (m,n*k)

    Parameters
    ----------
    data: nd-array

    Returns
    -------
    nd-array of reduced dimension
    """

    reduced = data[0]
    for i in range(1, len(data)):
        reduced = np.hstack((reduced, data[i]))
    return reduced

# --------------------------------------------------------------------------

def get_1d_as_n_m(index, cols):
    """
    Get a 1d coordinate (e.g. an index) and turn it into a 2d coordinate determined by the number of columns.
    Mostly used for calculated the coordinates for matplotlib-subplots

    Parameters
    ----------
    index: the 1d index to be converted
    cols: the number of columns

    Returns
    -------
    2d coordinates as tuple
    """
    a = int(index / cols)
    b = index % cols
    return (a, b)

# --------------------------------------------------------------------------

def get_index_of_prefixes(string: str, prefixes: list):
    """
    Get the index of the beginning of one of the prefixes

    Parameters
    ----------
    string: the string to be checked against
    prefixes: list of prefixes

    Returns
    -------
    whether or not it contains a prefix from prefixes and the index

    """
    for i in range(0,len(prefixes)):
        if string.startswith(prefixes[i]):
            return True, i
    return False, -1

# --------------------------------------------------------------------------

def data_to_freq_to_prob_dist(A, fs, channels):
    """
    Turn the data A into a probablity distribution by using MNE RawArray
    Parameters
    ----------
    A: The data
    fs: The sampling rate to use for the RawArray
    channels: The channels to use for the RawArray

    Returns
    -------
    Probability distribution

    """
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
    raw1 = mne.io.RawArray(A, info)
    A = reduce_one_dim(psd_welch(raw1)[0])
    A = A / np.linalg.norm(A)
    return A

# --------------------------------------------------------------------------

def get_single_class(dataset, label):
    """
    Get the data of a class from the dataset-object or data-object

    Parameters
    ----------
    dataset: either a Dataset-object or a Data-object
    label: the class

    Returns
    -------
    Data or Dataset with single class

    """

    if isinstance(dataset, Dataset):
        train_X = dataset.train_data.X
        train_y = dataset.train_data.y
        train_X_ = train_X[train_y == label]
        train_y_ = train_y[train_y == label]
        test_X = dataset.train_data.X
        test_y = dataset.train_data.y
        test_X_ = test_X[train_y == label]
        test_y_ = test_y[train_y == label]
        train = Data(train_X_, train_y_, create_onehot_vector(train_y_, 2))
        test = Data(test_X_, test_y_, create_onehot_vector(test_y_, 2))
        return Dataset(train, test)

    elif isinstance(dataset, Data):
        train_X = dataset.X
        train_y = dataset.y
        train_X_ = train_X[train_y == label]
        train_y_ = train_y[train_y == label]
        return Data(train_X_, train_y_, create_onehot_vector(train_y_, 2))

# --------------------------------------------------------------------------

def concat_images_horizontally(im1, im2, color=(255, 255, 255)):
    """
    Concatenates images horizontally

    Parameters
    ----------
    im1, im2: Images to concatenate
    color: Background color for the empty space in case the images dont have the same height

    Returns
    -------
    concatenated images
    """

    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# --------------------------------------------------------------------------

def get_random_sample(X, exclude_samples_list):
    """
    Get a random sample from X that is not in the exclude_samples_list

    Parameters
    ----------
    X: Data to sample from
    exclude_samples_list: Samples not to choose

    Returns
    -------
    Random sample and index

    """
    i = 0
    while True:
        n = random.randint(0, len(X))
        if not(n in exclude_samples_list):
            exclude_samples_list.append(n)
            break
        i += 1
        if i > 10*len(X) or len(X) == len(exclude_samples_list):
            break
    return X[n:n+1], n

# --------------------------------------------------------------------------

