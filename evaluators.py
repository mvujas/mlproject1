import validation
import numpy as np

# TODO: document

def cross_validation_evaluator(train_model, 
        cross_validation_score_reducer, number_of_folds=10):
    def evaluation_function(y, x):
        accs, f_scores = validation.cross_validation(
            y, x, train_model, number_of_folds)
        return cross_validation_score_reducer(accs, f_scores)
    return evaluation_function 


def cross_validation_mean_acc_evaluator(train_model, 
        number_of_folds=10):
    def mean_accuracy_reduction(accs, _):
        return np.mean(accs)
    return cross_validation_evaluator(
        train_model, 
        mean_accuracy_reduction, 
        number_of_folds)