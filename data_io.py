import numpy as np

def load_data(file_path):
    """Loads data from the file under the given file path and returns it in form
        of a tuple (columns, ids, labels, features) 
        where labels are mapped to numerical value

    Parameters
    ----------
    file_path : str
        Path to the file which data should be read from
    """
    with open(file_path, 'r') as f:
        # retrieving names of columns and calculating number of them
        column_names = f.readline().strip().split(',')
        num_cols = len(column_names)
        # starts reading the file from beginning
        f.seek(0)
        labels = np.genfromtxt(f, delimiter=",", 
            skip_header=1, usecols=[1], converters={1: lambda x: 0 if x == b"b" else 1})
        f.seek(0)
        # IDs and other columns except Predictions are loaded 
        #   together as they are all numerical values
        data_columns = np.genfromtxt(f, delimiter=",", 
            skip_header=1, usecols=[0, *range(2, num_cols)])
    ids = data_columns[:, 0].astype(np.int32)
    # while features start from column 2 in the file, there is no prediction column
    #   in data_columns array and therefore they start from index 1 in the given array 
    features = data_columns[:, 1:]
    return column_names, ids, labels, features

def save_predictions_as_csv(file_path, ids, predictions):
    """Save predictions in form of a csv file in the required format

    Parameters
    ----------
    file_path : str
        Path to the file in which data should be saved 
        (.csv suffix will to the file name be added if not present in the path)
    ids : np.ndarray
        Array of ids for each data point (expected to be integer array)
    predictions : np.ndarray
        Array of predicted classes for each data point (expected to be string array)
    """
    # Making sure file path is good and corresponds to csv format
    file_path = file_path.strip()
    if not file_path.endswith('.csv'):
        file_path = file_path + '.csv'
    # File header
    id_header = 'Id'
    predictions_header = 'Prediction'
    header = f'{id_header},{predictions_header}'
    # Workaround saving data of different types (id is int and predictions are strings)
    data = np.zeros(ids.size, 
        dtype=[(id_header, ids.dtype), (predictions_header, predictions.dtype)])
    data[id_header] = ids
    data[predictions_header] = predictions
    # Saving data. Header is considered as a comment, so we have to remove 
    #   default prefix for comments ('# ') with argument comments = ''
    np.savetxt(file_path, data, header=header, fmt='%d,%s', delimiter=',', comments='')