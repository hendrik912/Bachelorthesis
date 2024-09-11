import numpy as np

# --------------------------------------------------------------------------

def standardize(X):
    """
    Standardize the data
    Parameters
    ----------
    X: The data

    Returns
    -------
    Standardized data
    """
    return (X - X.mean()) / X.std()

# --------------------------------------------------------------------------
