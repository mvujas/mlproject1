import validation
import numpy as np

# TODO: document

def cross_validation_evaluator(train_model, 
        cross_validation_score_reducer, number_of_folds=10):
    '''Evaluation function that performs cross validation on the given model and
        uses the given reduction function to map the result of cross validation (accuracy, F_1 score for fold)
        to a number.

    Parameters
    ----------
    train_model : (np.ndarray, np.ndarray) -> ((np.ndarray) -> np.ndarray)
        A function that accepts data samples, trains a model and afterwards returns its predicton function
    cross_validation_score_reducer : ([float], [float]) -> float
        Function that maps the result of crossvalidation (accuracy, F_1 score for fold) to a number
    number_of_folds : int
        Number of subsets the data should be split into when performing cross valdiation
    '''
    def evaluation_function(y, x):
        '''Evaluation function accepts data samples and is expected to return a number as an output.

        Parameters
        ----------
        y : ndarray
            Labels
        x : ndarray
            Features
        '''
        accs, f_scores = validation.cross_validation(
            y, x, train_model, number_of_folds)
        return cross_validation_score_reducer(accs, f_scores)
    return evaluation_function 


def cross_validation_mean_acc_evaluator(train_model, 
        number_of_folds=10):
    '''"Extends" cross validation evaluator by reducing the result of cross validation
        to the mean of accuracies over folds.
    
    Parameters
    ----------
    train_model : (np.ndarray, np.ndarray) -> ((np.ndarray) -> np.ndarray)
        A function that accepts data samples, trains a model and afterwards returns its predicton function
    number_of_folds : int
        Number of subsets the data should be split into when performing cross valdiation
    '''
    def mean_accuracy_reduction(accs, _):
        return np.mean(accs)
    return cross_validation_evaluator(
        train_model, 
        mean_accuracy_reduction, 
        number_of_folds)