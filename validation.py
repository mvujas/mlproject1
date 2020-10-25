import numpy as np
import metrics
from data_preprocessing import stratify_sampling

def cross_validation(y, x, train_model, number_of_folds, verbose=False):
    """Implements k-fold cross validation.
        The result of the function is a 2 element tuple whose first element is array of 
        acurracies for each step while the second is array of f1-scores for each step.
        It performes stratification before splitting data into subsets.

    Parameters
    ----------
    y : np.ndarray
        Labels
    x : np.ndarray
        Features
    train_model : (np.ndarray, np.ndarray) -> ((np.ndarray) -> np.ndarray)
        A function that train model and returns function that should accept
        unlabeled features and return their classes
    number_of_folds : int
        Positive integer that indicates number of folds the data should be split into
    verbose : bool
        Indicate whether results should be summarized in the human readable form
        at the end of the function
    """
    stratified_idx = stratify_sampling(y, number_of_folds, True)
    accuracies = np.zeros((number_of_folds,))
    fbeta_scores = np.zeros((number_of_folds,))
    for i in range(number_of_folds):
        training_instances = np.concatenate(
            list(map(lambda x: x[1], filter(lambda x: x[0] != 1, enumerate(stratified_idx))))    )
        
        y_train = y[training_instances]
        x_train = x[training_instances]
        y_test = y[stratified_idx[i]]
        x_test = x[stratified_idx[i]]

        model = train_model(y_train, x_train)
        predictions = model(x_test)

        acc = metrics.accuracy(y_test, predictions)
        f_score = metrics.fbeta_score(y_test, predictions)
        accuracies[i] = acc
        fbeta_scores[i] = f_score

    if verbose:
        def print_metric_stats(metric_name, metric_values):
            print(f'    {metric_name}: avg {np.mean(metric_values):.5}, ' +
                f'max {np.max(metric_values):.5}, min {np.min(metric_values):.5}, ' +
                f'stddev {np.std(metric_values):.5}')

        print(f'------ {number_of_folds}-fold cross validation results ------')
        print_metric_stats('Accuracy', accuracies)
        print_metric_stats('Fbeta score', fbeta_scores)

    return accuracies, fbeta_scores
