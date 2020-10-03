import numpy as np

def load_data(file_path):
    """Loads data from the file under the given file path and returns it in form
        of a tuple (ids, labels, features) where labels are mapped to numerical value

    Parameters
    ----------
    file_path : str
        Path to the file which data should be read from
    """
    with open(file_path, 'r') as f:
        num_cols = len(f.readline().split(','))
        # starts reading the file from beginning
        f.seek(0)
        labels = np.genfromtxt(f, delimiter=",", 
            skip_header=1, usecols=[1], converters={1: lambda x: 0 if x == "b" else 1})
        f.seek(0)
        # IDs and other columns except Predictions are loaded 
        #   together as they are all numerical values
        data_columns = np.genfromtxt(f, delimiter=",", 
            skip_header=1, usecols=[0, *range(2, num_cols)])
    ids = data_columns[:, 0]
    # while features start from column 2 in the file, there are no prediction column
    #   in data_columns array and therefore they start from index 1 in the given array 
    features = data_columns[:, 1:]
    return ids, labels, features