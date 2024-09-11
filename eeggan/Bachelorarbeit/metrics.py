import numpy as np

# --------------------------------------------------------------------------

def precision(prediction, reference, label):
    """
    Calculates the precision score

    Parameters
    ----------
    prediction: The predictions of the classifer
    reference: The ground truths
    label: For which class the score is to be calculated

    Returns
    -------
    the precision score as a float
    """

    tp = np.sum(prediction[reference == label] == reference[reference == label])
    fp = np.sum(prediction[reference != label] == label)
    return float(tp)/(tp+fp)

# --------------------------------------------------------------------------

def recall(prediction, reference, label):
    """
    Calculates the recall score

    Parameters
    ----------
    prediction: The predictions of the classifer
    reference: The ground truths
    label: For which class the score is to be calculated

    Returns
    -------
    the recall score as a float
    """

    tp = np.sum(prediction[reference == label] == reference[reference == label])
    fn = len(reference[reference == label]) - tp
    return float(tp)/(tp+fn)

# --------------------------------------------------------------------------

def fscore(prediction, reference, label):
    """
    Calculates the f1-score

    Parameters
    ----------
    prediction: The predictions of the classifer
    reference: The ground truths
    label: For which class the score is to be calculated

    Returns
    -------
    the f1-score as a float
    """

    prec = precision(prediction, reference, label)
    rec = recall(prediction, reference, label)
    if prec*rec == 0: return 0
    else: return 2*(prec*rec)/(prec+rec)

# --------------------------------------------------------------------------

def accuracy(prediction, reference):
    """
    Calculates the accuracy score

    Parameters
    ----------
    prediction: The predictions of the classifer
    reference: The ground truths
    label: For which class the score is to be calculated

    Returns
    -------
    the accuracy as a float
    """

    correct = np.sum(prediction == reference)
    return float(correct) / len(reference)

# --------------------------------------------------------------------------