import numpy as np
import class_config
import math

def accuracy(labels, predictions):
    """Calculates accuracy of prediction.
        labels and predictions should have the same size

    Parameters
    ----------
    labels : np.ndarray
        Annotated labels
    predictions: np.array
        Labels predicted by model
    """
    correctly_predicted = np.sum(labels == predictions)
    instances_num = labels.size
    return correctly_predicted / instances_num


def calculate_true_positives(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED):
    """Calculates number of true positives for given prediction"""
    true_positive_count = np.sum(
        np.logical_and(predictions == class_to_retrieve, predictions == labels))
    return true_positive_count


def calculate_false_positives(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED):
    """Calculates number of false positives for given prediction"""
    false_positive_count = np.sum(
        np.logical_and(predictions == class_to_retrieve, predictions != labels))
    return false_positive_count


def calculate_true_negatives(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED):
    """Calculates number of true negatives for given prediction"""
    true_negative_count = np.sum(
        np.logical_and(predictions != class_to_retrieve, predictions == labels))
    return true_negative_count


def calculate_false_negatives(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED):
    """Calculates number of false negatives for given prediction"""
    false_negative_count = np.sum(
        np.logical_and(predictions != class_to_retrieve, predictions != labels))
    return false_negative_count


def __recall_using_confusion_matrix(true_positive_count, false_negative_count):
    return true_positive_count / (true_positive_count + false_negative_count)


def __precision_using_confusion_matrix(true_positive_count, false_positive_count):
    return true_positive_count / (true_positive_count + false_positive_count)

def precision(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED):
    """Calculates precision of given prediction.
        labels and predictions should have the same size

    Parameters
    ----------
    labels : np.ndarray
        Annotated labels
    predictions : np.array
        Labels predicted by model
    class_to_retrieve : labels.dtype
        Class to which instances that should be retrieved belong
    """
    tp = calculate_true_positives(labels, predictions, class_to_retrieve)
    fp = calculate_false_positives(labels, predictions, class_to_retrieve)
    
    precision = __precision_using_confusion_matrix(tp, fp)
    # Sometimes division by 0 may occur, in that cases we will consider precision is zero
    if math.isnan(precision):
        precision = 0.0
    return precision


def recall(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED):
    """Calculates recall of given prediction.
        labels and predictions should have the same size

    Parameters
    ----------
    labels : np.ndarray
        Annotated labels
    predictions : np.array
        Labels predicted by model
    class_to_retrieve : labels.dtype
        Class to which instances that should be retrieved belong
    """
    tp = calculate_true_positives(labels, predictions, class_to_retrieve)
    fn = calculate_false_negatives(labels, predictions, class_to_retrieve)

    recall = __recall_using_confusion_matrix(tp, fn)
    # Sometimes division by 0 may occur, in that cases we will consider recall is zero
    if math.isnan(recall):
        recall = 0.0
    return recall


def fbeta_score(labels, predictions, 
        class_to_retrieve=class_config.CLASS_TO_BE_RETRIEVED, beta=1):
    """Calculates F_{beta} score of given prediction.
        labels and predictions should have the same size

    Parameters
    ----------
    labels : np.ndarray
        Annotated labels
    predictions : np.array
        Labels predicted by model
    class_to_retrieve : labels.dtype
        Class to which instances that should be retrieved belong
    beta : float
        Positive real factor
    """
    tp = calculate_true_positives(labels, predictions, class_to_retrieve)
    fp = calculate_false_positives(labels, predictions, class_to_retrieve)
    fn = calculate_false_negatives(labels, predictions, class_to_retrieve)

    recall = __recall_using_confusion_matrix(tp, fn)
    precision = __precision_using_confusion_matrix(tp, fp)

    dividend = (1 + beta ** 2) * precision * recall
    divisor = (beta ** 2) * precision + recall 
    f_score = dividend / divisor
    # Sometimes division by 0 may occur, in that cases we will consider fbeta score is zero
    if math.isnan(f_score):
        f_score = 0.0
    return f_score