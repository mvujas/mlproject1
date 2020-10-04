import numpy as np

# TODO: Adjust one_hot to work with multicolumn data and decide on output format in such case
def one_hot(x, depth=None):
    """Transforms input into one hot format.
    All values of the input array are expected to be >= 0.
    Note that number of classes does not depend on number of distinctive values,
        so it is recommended to map interval to [0, N] such that every value of 
        interval is present at least once before calling this function.

    Parameters
    ----------
    x : np.ndarray
        Either 1D or (N, 1) array of positive integers
    depth : int or None
        Indicates minimal number of classes. 
            If value is None, the number of classes will be equal to the value of maximum element of x + 1.
            Otherwise it will be maximum between the value of maximum element of x + 1 and depth
    """
    if type(x) != np.int32:
        x = x.astype(np.int32)
    if x.ndim == 2:
        x = x[:, 0]
    min_depth = np.max(x) + 1
    if depth == None:
        depth = min_depth
    else:
        depth = max(min_depth, depth)
    return np.eye(depth)[x]


# TODO: Implement more efficient way to do mapping if this proves too slow
def map_values(x, mapping, default=0):
    """Maps values from the given array using mapping rules given in the dictionary
        
    Parameters
    ----------
    x : np.ndarray
        Array whose values are to be mapped
    mapping : dict
        Dictionary containing value to value mapping rules
    default
        Default mapping value for cells whose value is not specified in the mapping rule
    """
    condlist = []
    choicelist = []
    for key, value in mapping.items():
        condlist.append(x == key)
        choicelist.append(value)
    return np.select(condlist, choicelist, default=default)


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
    std_dev[np.where(std_dev == 0)] = 1
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
    dif[np.where(dif == 0)] = 1
    return (x - min_els) / dif


def prepend_bias_column(x):
    """Adds an additional column whose all values are 1 as 
        the first column of the input matrix (2D array).
        The number of columns of the result is 1 higher than of the input matrix.

    Parameters
    ----------
    x : np.ndarray
        Matrix to which a bias column is to be added
    """
    data_size = x.shape[0]
    bias_column = np.ones((data_size, 1))
    return np.append(bias_column, x, axis=1)


def nullify_missing_values(x, missing_field_matrix):
    """Set value of all fields in the given ndarray 
        whose corresponding value in missing_field_matrix is True to 0 

    Parameters
    ----------
    x : np.ndarray
        Ndarray whose missing fields are to be nulllified
    missing_field_matrix : np.ndarray
        Ndarray of bools of shape equal to the shape of x 
        whose values corresponds to whether a field in the same position in x is missing  
    """
    return np.where(missing_field_matrix, x, 0)


def apply_transformation(x, column_idx, transformation):
    columns = x[:, column_idx]
    x_without_columns = np.delete(x, column_idx, axis=1)
    columns = transformation(columns)
    return np.append(x_without_columns, columns, axis=1)


def shuffle(y, x):
    """Randomly shuffles data.
        All the provided data have to be of same size

    Parameters
    ----------
    y : np.ndarray
        Labels
    x : np.ndarray
        Features
    """
    data_size = y.size
    if data_size != x.size:
        raise ValueError(
            'Features and labels have to have the same size')
    
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    return shuffled_y, shuffled_x


def stratify_sampling(y, x, number_of_folds, shuffle=False):
    """Splits data points in folds of relatively equal size 
        (can sligtly differ if number of data points is not dividable by number of folds)
        such that each fold has equal number of instances of same class

    Parameters
    ----------
    y : np.ndarray
        Labels
    x : np.ndarray
        Features
    number_of_folds : int
        Number of folds data point should be split into
    shuffle : bool
        Indicate whether data should be shuffled before being stratified
    """
    def sizes_of_folds(size, number_of_folds):
        """Helper function that returns where data should be split and whose output
            can be fed directly into np.split function as the second argument

        Parameters
        ----------
        size : int
            Size of the collection
        number_of_folds : int
            Number of folds collection should be split into
        """
        min_fold_size = round(size / number_of_folds)
        folds = []
        last = 0
        for i in range(number_of_folds - 1):
            last = last + min_fold_size
            folds.append(last)
        return folds

    if shuffle:
        y, x = shuffle(y, x)

    classes = np.unique(y)
    # Divide indexes of instances of each class in respective arrays
    class_idx = [np.where(y == class_val) for class_val in classes]
    class_idx_fold = []
    for single_class_idx in class_idx:
        # For each class split their indexes into smaller folds
        idx = single_class_idx[0]
        fold_slices = sizes_of_folds(idx.size, number_of_folds)
        idx_folds = np.split(idx, fold_slices)
        class_idx_fold.append(idx_folds)

    y_folds = []
    x_folds = []
    for i in range(number_of_folds):
        # Take instances of each class
        new_x_fold = np.concatenate([x[class_folds[i]] for class_folds in class_idx_fold])
        new_y_fold = np.concatenate([y[class_folds[i]] for class_folds in class_idx_fold])
        x_folds.append(new_x_fold)
        y_folds.append(new_y_fold)

    return y_folds, x_folds