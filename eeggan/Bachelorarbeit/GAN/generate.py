import eeggan.Bachelorarbeit.util as util
import torch
import numpy as np
import eeggan.Bachelorarbeit.postprocessing as postprocessing
from sklearn.utils import shuffle
from eeggan.cuda import to_device
from numpy.random.mtrand import RandomState
from eeggan.training.trainer.utils import detach_all
from eeggan.data.dataset import Data
from eeggan.data.preprocess.util import create_onehot_vector

# --------------------------------------------------------------------------

def generate_data_separate_gans_pp(generator_c0, generator_c1, label_c0, label_c1, num_samples, classifier, metric, real_data=None, channels=None, fs=None,
                                   classification_threshold=None, freq_filter_range=None, threshold=None, batch_size=50, sl=0):
    """
    Generate data where each class has its own GAN and process the data by applying a bandpassfilter and then filtering
    them with a threshold for the given metric, as well as a a classification threshold.

    Parameters
    ----------
    generator_c0, generator_c1: Generators for each class
    label_c0: label of class 0
    label_c1: label of class 1
    num_samples: number of samples to generate
    classifier: the classifier to use for the rejection sampling with the classification threshold
    metric: the metric to calcuate the distance on and filtering by the threshold
    real_data: data to calculate the metric on
    channels: the eeg-channels
    fs: the sampling rate
    classification_threshold: the classification threshold
    freq_filter_range: the range of the bandpassfilter
    threshold: the threshold for the distance metric (e.g. JSD)
    batch_size: the number of samples to generate in one step)
    sl: the sample length

    Returns
    -------
    The generated data

    """
    real_c0 = util.get_single_class(real_data, label_c0)
    data_fake_c0 = generate_data_pp(generator_c0, label_c0, int(num_samples / 2), classifier, metric,
                                    real_c0.train_data.X, channels, fs,
                                    classification_threshold=classification_threshold,
                                    freq_filter_range=freq_filter_range,
                                    threshold=threshold,
                                    batch_size=batch_size,
                                    sl=sl)

    real_c1 = util.get_single_class(real_data, label_c1)
    data_fake_c1 = generate_data_pp(generator_c1, label_c1, int(num_samples / 2), classifier, metric,
                                    real_c1.train_data.X, channels, fs,
                                    classification_threshold=classification_threshold,
                                    freq_filter_range=freq_filter_range,
                                    threshold=threshold,
                                    batch_size=batch_size,
                                    sl=sl)

    X = np.vstack((data_fake_c0.X, data_fake_c1.X))
    y = np.hstack((data_fake_c0.y.astype(dtype=np.int64), data_fake_c1.y.astype(dtype=np.int64)))

    X, y = shuffle(X, y)

    return Data(X, y, create_onehot_vector(y, 2))

# --------------------------------------------------------------------------

def generate_data_pp(generator, label, num_samples, classifier, metric, real_data=None, channels=None, fs=None,
                     classification_threshold=None, freq_filter_range=None, threshold=None, batch_size=200, sl=0):
    """
    Generate data, apply a bandpassfilter and filter the samples based on a classification threshold and
    a threshold for a distance metric.

    Parameters
    ----------
    generator: generator to use
    label: class of data to generate (and filter for)
    num_samples: number of samples to generate
    classifier: classifier to use for the classification threshold
    metric: distance metric to use
    real_data: data to calculate the metric on
   channels: the eeg-channels
    fs: the sampling rate
    classification_threshold: the classification threshold
    freq_filter_range: the range of the bandpassfilter
    threshold: the threshold for the distance metric (e.g. JSD)
    batch_size: the number of samples to generate in one step)
    sl: the sample length

    Returns
    -------
    generated data
    """
    rng = RandomState()
    device = torch.device('cuda:0')
    samples = []
    count = 0
    b = True

    while(b):
        if len(samples) == 0 and count > 100:
            print("generate_data_pp no samples under threshold")
        count += 1

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(device, *generator.create_latent_input(rng, batch_size))
            latent, y_fake, y_onehot_fake = detach_all(latent, y_fake, y_onehot_fake)

        X_fake = generator(latent.requires_grad_(False), y=y_fake.requires_grad_(False),
                           y_onehot=y_onehot_fake.requires_grad_(False))

        X_fake = X_fake.cpu().detach().numpy()

        if threshold != None:
            X_fake = postprocessing.rejection_sampling_and_freq_filter(real_data, X_fake, threshold, metric,
                                                                       fs, channels, freq_filter_range,
                                                                       False, sl)
        else:
            print("metric threshold is none")

        for i in range(0, len(X_fake)):

            if classification_threshold != None and classifier != None:
                preds = classifier.predict_outs(X_fake[i:i+1], individual_crops=False)
                preds = torch.exp(torch.tensor(preds))

                if preds[0][label] < classification_threshold:
                    continue

            if len(samples) < num_samples:
                samples.append(X_fake[i])
            else:
                b = False
                break

    X_fake = np.array(samples)

    if label == 0:
        y = np.zeros(len(X_fake))
    else:
        y = np.ones(len(X_fake))

    y = y.astype(dtype=np.int64)
    data = Data(X_fake, y, create_onehot_vector(y, 2))

    return data

# --------------------------------------------------------------------------

def generate_data_separate_gans(generator_c0, generator_c1, num_samples, batch_size=50):
    """
    Generate data where each class has its own GAN

    Parameters
    ----------
    generator_c0, generator_c1: The generators for each class
    num_samples: Number of samples to generate
    batch_size: The number of samples to generate each iteration

    Returns
    -------
    generated data
    """

    data_fake_c0 = generate_data(generator_c0, int(num_samples / 2), batch_size)
    data_fake_c1 = generate_data(generator_c1, int(num_samples / 2), batch_size)

    X = np.vstack((data_fake_c0.X, data_fake_c1.X))
    y = np.hstack((data_fake_c0.y.astype(dtype=np.int64), data_fake_c1.y.astype(dtype=np.int64)))
    X, y = shuffle(X, y)

    return Data(X, y, create_onehot_vector(y,2))

# --------------------------------------------------------------------------

def generate_data(generator, num_samples, batch_size):
    """
    Generate data

    Parameters
    ----------
    generator: The generator to use
    num_samples: The number of samples to generate
    batch_size: The number of samples to generate in each iteration (due to memory reasons they are not
                created all at the same time)

    Returns
    -------

    """

    rng = RandomState()
    device = torch.device('cuda:0')

    step = batch_size
    count = 0

    X_fake_all = None
    y_fake_all = None

    while(count < num_samples):

        if count + step > num_samples:
            step = num_samples - count

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(device, *generator.create_latent_input(rng, step))
            latent, y_fake, y_onehot_fake = detach_all(latent, y_fake, y_onehot_fake)

        X_fake = generator(latent.requires_grad_(False), y=y_fake.requires_grad_(False),
                           y_onehot=y_onehot_fake.requires_grad_(False))

        X_fake = X_fake.cpu().detach().numpy()
        y_fake = y_fake.cpu().detach().numpy()
        y_fake = y_fake.astype(dtype=np.int64)

        if not type(X_fake_all) is np.ndarray:
            X_fake_all = X_fake
            y_fake_all = y_fake
        else:
            X_fake_all = np.vstack((X_fake_all, X_fake))
            y_fake_all = np.hstack((y_fake_all, y_fake))

        count += step

    data = Data(X_fake_all, y_fake_all, create_onehot_vector(y_fake_all, 2))

    return data

# --------------------------------------------------------------------------
