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

def one_hot_(x, column_to_index_mapping, column_name, depth=None):
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
    
    column_to_index_mapping_upd = {}
    for k, v in column_to_index_mapping.items():
        if v > column_to_index_mapping[column_name]:
            column_to_index_mapping_upd[k] = v - 1
        elif v < column_to_index_mapping[column_name]:
            column_to_index_mapping_upd[k] = v
    
    for d in range(depth):
        column_to_index_mapping_upd[column_name + '_' + str(d)] = len(column_to_index_mapping_upd)
    return np.eye(depth)[x], column_to_index_mapping_upd


def one_hot_transformation(x, column_name, column_to_index_mapping):
    column_idx = column_to_index_mapping[column_name]
    columns = x[:, column_idx]
    new_columns, column_to_index_mapping_upd = one_hot_(columns, column_to_index_mapping, column_name)
    if columns.shape != new_columns.shape:
        x_without_columns = np.delete(x, column_idx, axis=1)
        x = np.append(x_without_columns, new_columns, axis=1)
    else:
        x[:, column_idx] = new_columns
    return x, column_to_index_mapping_upd


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


def standardize_with_nans(x):
    """Standardizes values of each column in the given matrix (2D array) handling nans

    Parameters
    ----------
    x : np.ndarray
        Matrix to be standardized
    """

    for i in range(x.shape[1]):
        mask = ~np.isnan(x[:, i])
        std = max(1e-6, np.std(x[mask, i], ddof=1))
        x[mask, i] = (x[mask, i] - np.mean(x[mask, i])) / std

    return x


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
    return np.where(missing_field_matrix, 0, x)

def mean_missing_values(x, missing_field_matrix):
    """Set value of all fields in the given ndarray 
        whose corresponding value in missing_field_matrix is True to the mean value of 
        the corresponding column

    Parameters
    ----------
    x : np.ndarray
        Ndarray whose missing fields are to be nulllified
    missing_field_matrix : np.ndarray
        Ndarray of bools of shape equal to the shape of x 
        whose values corresponds to whether a field in the same position in x is missing
    """
    for col in range(missing_field_matrix.shape[1]):
        tofill = x[:, col][~missing_field_matrix[:, col]].mean()
        x[:, col] = np.where(missing_field_matrix[:, col], tofill, x[:, col])
    return x

def median_missing_values(x, missing_field_matrix):
    """Set value of all fields in the given ndarray 
        whose corresponding value in missing_field_matrix is True to the median value of 
        the corresponding column

    Parameters
    ----------
    x : np.ndarray
        Ndarray whose missing fields are to be nulllified
    missing_field_matrix : np.ndarray
        Ndarray of bools of shape equal to the shape of x 
        whose values corresponds to whether a field in the same position in x is missing
    """
    for col in range(missing_field_matrix.shape[1]):
        tofill = np.median(x[:, col][~missing_field_matrix[:, col]])
        x[:, col] = np.where(missing_field_matrix[:, col], tofill, x[:, col])
    return x

def onehot_missing_values(x, missing_field_matrix, col_to_index_mapping):
    # TODO: return list of one hotted columns (required for the test submission)
    # REMARK: add col_to_index_mapping into documentation and why it is even used in this function ?!
    """add onehot columns for each feature which indicates that there was nan value

    Parameters
    ----------
    x : np.ndarray
        Ndarray whose missing fields are to be nulllified
    missing_field_matrix : np.ndarray
        Ndarray of bools of shape equal to the shape of x 
        whose values corresponds to whether a field in the same position in x is missing
    """
    col_to_index_mapping_inverse = {v: k for k, v in col_to_index_mapping.items()}
    
    new_cols = []
    for col in range(missing_field_matrix.shape[1]):
        nan_col = np.sum(missing_field_matrix[:, col]) > 0
        if nan_col:
            new_col = np.zeros_like(x[:, 0])
            new_col[np.where(missing_field_matrix[:, col])] = 1
            new_cols.append(new_col)
            
            col_name = col_to_index_mapping_inverse[col]
            col_to_index_mapping[col_name + '_isnan'] = len(col_to_index_mapping)
    new_cols = np.array(new_cols).T
    return np.concatenate((x, new_cols), 1), col_to_index_mapping


# DELETE IF NOT NEEDED
# def onehot_missing_values_transform(x, missing_field_matrix, col_to_index_mapping):
#     # TODO: return list of one hotted columns (required for the test submission)
#     """add onehot columns for each feature which indicates that there was nan value

#     Parameters
#     ----------
#     x : np.ndarray
#         Ndarray whose missing fields are to be nulllified
#     missing_field_matrix : np.ndarray
#         Ndarray of bools of shape equal to the shape of x 
#         whose values corresponds to whether a field in the same position in x is missing
#     """
#     name_set = set()
#     for k in col_to_index_mapping.keys():
#         if 'nan' in k:
#             name_set.update(k[:-6])
        
#     new_cols = []
#     for name in name_set:
#         col = col_to_index_mapping[name]
#         new_col = np.zeros_like(x[:, 0])
#         new_col[np.where(missing_field_matrix[:, col])] = 1
#         new_cols.append(new_col)

#         col_name = col_to_index_mapping_inverse[col]
#         col_to_index_mapping[col_name + '_isnan'] = len(col_to_index_mapping)
#     new_cols = np.array(new_cols).T
#     return np.concatenate((x, new_cols), 1), col_to_index_mapping


def build_poly(x, column_idx, degree):
    """Takes the given columns from the features matrix and builds polynomial basis functions out of them.
        Degree is specified by a parameter degree which can either be a positive integer of an iterable structure of positive integers.
        If it is an iterable structure the resulting polymial will only have degrees of the specified columns given in the structure.
        If degree is a positive integer the polynomial will take all degrees from 1 to the given positive integer.

    Parameters
    ----------
    x : np.ndarray
        Features
    column_idx : [int]
        Columns from the feature matrix to be augmented 
    degree : int or iterable of int
        Degrees the polynoms should contain (if it is an iterable it will take only 
        degrees from it; while if it is an int it will take degree from 1 to the given int)
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
        
    if isinstance(degree, int):
        degree = list(range(1, degree + 1))

    columns = x[:, column_idx]
    column_len = columns.shape[1]

    result = np.empty((columns.shape[0], columns.shape[1] * len(degree)), dtype=float)
    for degree_ind, degree_val in enumerate(degree):
        result[:, degree_ind * column_len:(degree_ind + 1) * column_len] = columns ** degree_val

    return result


def build_pairwise(x, column_idx):
    """Takes specified columns from the given matrix and multiply each 2 with each other.
        The result is the original matrix together with the given columns multiplied with each other as new columns.

    Parameters
    ----------
    x : np.ndarray
        2D matrix
    column_idx : [int]
        Indexes of columns of x that should be multiplied with each other
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
        
    columns = np.copy(x[:, column_idx])
    pairwise = []
    for i in range(columns.shape[1] - 1):
        for j in range(i + 1, columns.shape[1] - 1):
            pairwise.append(columns[:, i] * columns[:, j])
    pairwise = np.array(pairwise).T
    return np.concatenate([np.copy(x), pairwise], 1)


def __two_combinations(n):
        """A helper function that calculates the number of possible combinations 
            with two elements over a set with n elements.
        """
        return n * (n - 1) // 2


def build_pairwise_alt(x, column_idx):
    """Takes specified columns from the given matrix and multiply each 2 with each other.
        The result is the original matrix together with the given columns multiplied with each other as new columns.

    Parameters
    ----------
    x : np.ndarray
        2D matrix
    column_idx : [int]
        Indexes of columns of x that should be multiplied with each other
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]

    columns_num = len(column_idx)
    columns = x[:, column_idx]

    D = x.shape[1] # total number of columns of x
    new_columns = __two_combinations(columns_num)

    result = np.empty((x.shape[0], new_columns + D), dtype=x.dtype)
    result[:, :D] = x # First few columns are original values of x

    index_counter = 0
    for i in range(columns_num - 1):
        for j in range(i + 1, columns_num):
            el_index = D + index_counter
            result[:, el_index] = columns[:, i] * columns[:, j]
            index_counter += 1

    return result


def build_pairwise_and_select_indexes(x, column_idx, selected_idx):
    """Function that works similarly to build_pairwise, but only returns the result
        elements that would end up on indexes specified with selected_idx. While this
        may seem redundant (e.g. build_pairwise(x, column_idx)[:, selected_idx]) 
        it is really helpful in order to avoid RAM shortage which happened during testing 
        with really high number of columns (> 1500) as it is lazy and don't calculate
        columns that are not needed.

    Parameters
    ----------
    x : np.ndarray
        2D matrix
    column_idx : [int]
        Indexes of columns of x that should be multiplied with each other
    selected_idx : [int]
        Indexes of columns of result that should be selected
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]

    selected_idx = sorted(list(set(selected_idx)))

    columns_num = len(column_idx)
    columns = x[:, column_idx]

    D = x.shape[1] # total number of columns of x

    num_selected_idx = len(selected_idx)
    result = np.empty((x.shape[0], num_selected_idx), dtype=x.dtype)

    selected_idx_from_x = list(filter(lambda n: n < D, selected_idx))
    result[:, :len(selected_idx_from_x)] = x[:, selected_idx_from_x] # First few columns are original values of x

    current_selected_idx_index = len(selected_idx_from_x)
    
    index_counter = D
    selected_all_indexes = False
    for i in range(columns_num - 1):
        if selected_all_indexes:
            break
        for j in range(i + 1, columns_num):
            el_index = index_counter
            if selected_idx[current_selected_idx_index] == el_index:
                result[:, current_selected_idx_index] = columns[:, i] * columns[:, j]
                current_selected_idx_index += 1
                selected_all_indexes = current_selected_idx_index >= num_selected_idx
                if selected_all_indexes:
                    break
            index_counter += 1
    
    return result

def apply_transformation(x, column_idx, transformation, column_to_index_mapping=None):
    """Takes column with specified indexes from the given matrix, applies transformation
        function on them and return them on their old places if no additional columns are
        added after applying the transformation function, otherwise appends them as
        new columns at the end of the matrix while deleting the original columns.

    Parameters
    ----------
    x : np.ndarray
        2D Matrix
    column_idx : [int] 
        Indexes of columns of x that should be transformed
    transformation : np.ndarray -> np.ndarray
        A function that transforms the specified columns of x
    column_to_index_mapping : {str : int}
        A dictionary that maps column names to indexes.
        As some columns might be moved to the end after applying the transformation 
        function, it is important to update the mappings for columns whose mapping is 
        affected by applying the transformation function (it is supposed that if
        a transformation result of the column is pushed at the end it can't be kept
        track of anymore, so their mappings are updated and programmer is 
        expected to ignore mappings of these columns if this occurs [programmer is 
        supposed to know whether the transformation function will introduce new columns]) 
        The given parameter is treated as a mutable object.
    """
    columns = x[:, column_idx]
    new_columns = transformation(columns)
    if columns.shape != new_columns.shape:
        x_without_columns = np.delete(x, column_idx, axis=1)
        x = np.append(x_without_columns, new_columns, axis=1)
        if column_to_index_mapping is not None:
            for key in column_to_index_mapping:
                index = column_to_index_mapping[key]
                lower_indexes = np.sum(np.less(column_idx, index))
                column_to_index_mapping[key] -= lower_indexes
    else:
        x[:, column_idx] = new_columns
    return x

def shuffle_samples(y, x):
    """Randomly shuffles data.
        All the provided data have to be of same size

    Parameters
    ----------
    y : np.ndarray
        Labels
    x : np.ndarray
        Features
    """
    data_size = y.shape[0]
    if data_size != x.shape[0]:
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
        y, x = shuffle_samples(y, x)

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
