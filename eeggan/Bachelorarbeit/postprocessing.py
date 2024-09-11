import mne
import eeggan.Bachelorarbeit.util as util
import numpy as np

# --------------------------------------------------------------------------

def rejection_sampling_and_freq_filter(real_data, fake_data, threshold, metric, fs, channels, frange, comb, sl):
    """
    Filters the fake-data by calculating the given metric between the fake data and the real data and only
    letting the samples pass which are under the threshold.
    Also applies a bandpass filter.
    (computationally it was at one point more efficient to combine these to operations than to separate them)

    Parameters
    ----------
    real_data: The data to calculate the metric against
    fake_data: The data which is to be filtered
    threshold: The threshold under which the metric has to be for a sample to pass
    metric: The metric-function (e.g. JSD or SWD)
    fs: The sampling rate for an MNE object
    channels: The channels for an MNE object
    frange: The bandpass filter min and max
    comb: To pass onto the metric-function
    sl: Sample length

    Returns
    -------
    Filtered data

    """

    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')

    num = len(fake_data)*2
    if num > len(real_data):
        num = len(real_data)
    real_data = np.copy(real_data[:num])

    X = []
    for i in range(0, len(fake_data)):
        sample = fake_data[i]
        raw = mne.io.RawArray(sample, info)

        if frange != None:
            raw.filter(frange[0], frange[1])
            sample, _ = raw[:]

        sample = np.array([sample])
        res = metric(sample, real_data, comb, sl)

        if res < threshold:
            X.append(util.reduce_one_dim(sample))

    return np.array(X)

# --------------------------------------------------------------------------
