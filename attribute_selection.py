import numpy as np


def forward_attribute_selector(y, x, evaluator, 
        attribute_subset_size, verbose=False):
    """Runs forward attribute selection algorithm on given data.
        It starts with an empty set of selected attributes 
        and in each iteration tries all not yet selected attributes and chooses 
        one with the highest value of evaluation function,
        the given attribute is added to the set of selected attributes.
        This is repeated until the size of selected attributes set reaches
        size specified by attribute_subset_size parameter.
        Subsets are evaluated using evaluator function that should accept
        output and input data and return a numeric value.
        Attribute subsets with higher value of evaluator function are considered
        as better.

    Parameters
    ----------
    y : np.ndarray
        Labels (Output array)
    x : np.ndarray
        Features (Input matrix)
    evaluator : (np.ndarray, np.ndarray) -> float
        A function that accept output and input data and return a number
        indicating how good can input be used to reproduce answer
    attribute_subset_size : int
        Size of attribute subset that should be obtained using the algorithm.
        If it is higher than the number of attributes of input matrix it will
        stop when the set of all attributes of input matrix is reached.
    verbose : bool
        Specifies whether function should log progress after each iteration
    """
    num_instances, num_attributes = x.shape
    # Starts with empty set of indexes of selected attributes
    avaliable_attributes = list(range(num_attributes))
    choosen_attributes = []
    x_subset = np.empty((num_instances, 0), float)
    while len(avaliable_attributes) > 0 and \
            len(choosen_attributes) < attribute_subset_size:
        current_best = {
            'index' : None,
            'score' : None
        }
        # Tries adding each of the available attributes
        # to data and checks which will produce best results
        for attr_index in avaliable_attributes:
            x_subset = np.append(x_subset, x[:, [attr_index]], axis=1)
            score = evaluator(y, x_subset)
            if current_best['score'] is None or \
                    score > current_best['score']:
                current_best = {
                    'index' : attr_index,
                    'score' : score
                }
            x_subset = np.delete(x_subset, -1, axis=1)

        # The attribute that produced the best result is added
        # to the set of selected attributes
        choosen_attributes.append(current_best['index'])
        avaliable_attributes.remove(current_best['index'])
        x_subset = np.append(x_subset, 
                x[:, [current_best['index']]],
                axis = 1)

        # Logs progress if verbose is set to True
        if verbose:
            best_score = current_best['score']
            best_attributes = ', '.join(map(str, choosen_attributes))
            print(f' --- FORWARD ATTRIBUTE SELECTION: ' + 
                f'Best attribute indexes for size {len(choosen_attributes)}: ' +
                f'{best_attributes} (score {best_score:.6})')
    
    choosen_attributes.sort()
    return choosen_attributes


def backward_attribute_selector(y, x, evaluator, 
        attribute_subset_size, verbose=False):
    """Runs backward attribute selection algorithm on given data.
        It starts with a set containing all attributes of input matrix 
        and in each iteration tries to exclude each of the features 
        still in the set and checks how it affects the value
        of evaluation function, the attribute whose exclusion
        produces the highest value of evaluation function in the 
        given itteration is excluded. This is repeated until the 
        This is repeated until the size of selected attributes set reaches
        size specified by attribute_subset_size parameter or until
        the function runs out of attributes to exclude.
        Subsets are evaluated using evaluator function that should accept
        output and input data and return a numeric value.
        Attribute subsets with higher value of evaluator function are considered
        as better.

    Parameters
    ----------
    y : np.ndarray
        Labels (Output array)
    x : np.ndarray
        Features (Input matrix)
    evaluator : (np.ndarray, np.ndarray) -> float
        A function that accept output and input data and return a number
        indicating how good can input be used to reproduce answer
    attribute_subset_size : int
        Size of attribute subset that should be obtained using the algorithm.
        If it is a negative number the algorithm will stop when the 
        set of still available attributes is empty.
    verbose : bool
        Specifies whether function should log progress after each iteration
    """
    num_instances, num_attributes = x.shape
    # Set of selected attributes at the start is full set of attributes of input matrix
    choosen_attributes = list(range(num_attributes))
    while len(choosen_attributes) > 0 and \
            len(choosen_attributes) > attribute_subset_size:
        current_best = {
            'index' : None,
            'score' : None
        }
        # Tries excluding each of the attributes still in the set of selected attributes
        for local_index, attr_index in enumerate(choosen_attributes):
            choosen_attributes.remove(attr_index)
            x_subset = x[:, choosen_attributes]
            score = evaluator(y, x_subset)
            if current_best['score'] is None or \
                    score > current_best['score']:
                current_best = {
                    'index' : attr_index,
                    'score' : score
                }
            choosen_attributes.insert(local_index, attr_index)

        # Permanently removes the attribute whose exclusion affected result 
        # of evaluation function least negatively
        choosen_attributes.remove(current_best['index'])
        
        # Logs progress if verbose is set to True
        if verbose:
            best_score = current_best['score']
            best_attributes = ', '.join(map(str, choosen_attributes))
            print(f' --- BACKWARD ATTRIBUTE SELECTION: ' + 
                f'Best attribute indexes for size {len(choosen_attributes)}: ' +
                f'{best_attributes} (score {best_score:.6})')
    
    choosen_attributes.sort()
    return choosen_attributes