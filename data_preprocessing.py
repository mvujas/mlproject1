import numpy as np

# TODO: Adjust one_hot to work with multicolumn data and decide on output format in such case
def one_hot(x):
    """Transforms input into one hot format.
    All values of the input array are expected to be >= 0.
    Number of classes is equal by the value of maximum element + 1.
    Note that number of classes does not depend on number of distinctive values,
        so it is recommended to map interval to [0, N] such that every value of 
        interval is present at least once before calling this function.

    Parameters
    ----------
    x : np.ndarray
        Either 1D or (N, 1) array of positive integers
    """
    if x.ndim == 2:
        x = x[:, 0]
    depth = np.max(x) + 1
    return np.eye(depth)[x]

def standardize(x):
    """Standardizes values of each column in the given matrix (2D array)

    Parameters
    ----------
    x : np.ndarray
        Matrix to be standardized
    """
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)
    # In order to avoid division by 0, columns whose value is 0 are set to 1.
    #   This will result in standardized value of all cells in the given column being 0
    std_dev = np.where(std_dev != 0, std_dev, np.ones(std_dev.shape))
    return (x - mean) / std_dev

def normalize(x):
    """Scales values of each column in the given matrix (2D array) 
        to range [0, 1] using Min-Max feature scaling

    Parameters
    ----------
    x : np.ndarray
        Matrix whose features are to be scaled
    """
    min_els = np.min(x, axis=0)
    max_els = np.max(x, axis=0)
    dif = max_els - min_els
    # In order to avoid division by 0, columns whose value is 0 are set to 1.
    #   This will result in standardized value of all cells in the given column being 0
    dif = np.where(dif != 0, dif, np.ones(dif.shape))
    return (x - min_els) / dif

def prepend_bias_column(x):
    """Adds an additional column whose all values are 1 as 
        the first column of the input matrix (2D array).
        The number of columns of the result is 1 higher than of the input matrix.

    Parameters
    ----------
    x : np.ndarray
    """
    data_size = x.shape[0]
    bias_column = np.ones((data_size, 1))
    return np.append(bias_column, x, axis=1)