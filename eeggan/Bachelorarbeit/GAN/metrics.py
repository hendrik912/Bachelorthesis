import gc
import torch
import numpy as np
import eeggan.Bachelorarbeit.util as util
import eeggan.Bachelorarbeit.constants as constants
import eeggan.Bachelorarbeit.GAN.util as g_util
from eeggan.Bachelorarbeit.main import RESULT_PATH
from scipy.spatial.distance import jensenshannon
from eeggan.validation.metrics import wasserstein
from eeggan.cuda import to_cuda, init_cuda
from eeggan.cuda import to_device
from braindecode.torch_ext.modules import IntermediateOutputWrapper
from eeggan.validation.metrics.inception import calculate_inception_score
from eeggan.validation.validation_helper import logsoftmax_act_to_softmax

# --------------------------------------------------------------------------


def jsd(X_a, X_b, comb, sl):
    """
    Calculates the jensen shannon distance between X_a and X_b

    Parameters
    ----------
    X_a, X_b: Data
    comb, sl: Not used, only necessary because this function is stored in a list with other metric-functions which may
              need those parameters.
    Returns
    -------
    the distance
    """

    X_a_reduced = util.reduce_one_dim(np.copy(X_a))
    X_b_reduced = util.reduce_one_dim(np.copy(X_b))
    X_a_pb = util.data_to_freq_to_prob_dist(X_a_reduced, constants.FS, constants.CHNS)
    X_b_pb = util.data_to_freq_to_prob_dist(X_b_reduced, constants.FS, constants.CHNS)
    return jensenshannon(X_a_pb, X_b_pb)

# --------------------------------------------------------------------------


def inception_score(X_a, X_b, comb, sl):
    """
    Calculates the inception score of X_a
    Parameters
    ----------
    X_a: data
    X_b: Not used, only necessary because this function is stored in a list with other metric-functions which may
         need those parameters.
    comb, sl: needed to find the best base classifier, which is used in the calculation of the inception score
              (instead of the inception-model)
    Returns
    -------
    inception score
    """

    init_cuda()

    x1 = torch.from_numpy(X_a).float()
    x1, = to_device(x1.device, x1)
    x1 = x1[:, :, :, None]
    x1 = to_cuda(x1)

    select_modules = ['softmax']
    base_classifier = g_util.get_best_base_classifier(sl, RESULT_PATH)
    shallow = to_cuda(IntermediateOutputWrapper(select_modules, base_classifier.__dict__["network"]))

    with torch.no_grad():
        preds = shallow(x1)[0]
        preds = logsoftmax_act_to_softmax(preds)
        mean, score = calculate_inception_score(preds, 1, 1)

    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    return mean

# --------------------------------------------------------------------------


def sliced_wasserstein_dist(X_a, X_b, comb, sl):
    """
    Calculates the sliced wasserstein distance between X_a and X_b

    Parameters
    ----------
    X_a, X_b: data
    comb, sl: Not used, only necessary because this function is stored in a list with other metric-functions which may
              need those parameters.
    Returns
    -------
    The distance
    """

    n_projections = 100
    n_features = np.prod(X_a.shape[1:]).item()
    w_transform = wasserstein.create_wasserstein_transform_matrix(n_projections, n_features)
    res = wasserstein.calculate_sliced_wasserstein_distance(X_a, X_b, w_transform)
    return res

# --------------------------------------------------------------------------
