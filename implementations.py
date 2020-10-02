import numpy as np

def least_squares(y, tx):
    """
    Calculates the least squares solution using normal equations.
    
    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    """
    a = tx.T @ tx
    b = tx.T @ y
    return np.linalg.solve(a, b)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Calculates the least squares solution using gradient descent.
    
    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    initial_w : numpy.ndarray
         Initial parameters
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO : check whether we can add and delete arguments
    raise NotImplementedError()

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Calculates the least squares solution using stochastic gradient descent.
    
    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    initial_w : numpy.ndarray
         Initial parameters
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO : check whether we can add and delete arguments
    raise NotImplementedError()

def ridge_regression(y, tx, lambda_):
    """
    Implements ridge regression.
    
    Parameters
    ----------
    y :  numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    lambda_ : float
         Trade-off parameter
    """
    N = y.shape[0]
    D = tx.shape[1]
    lambda_prim = 2 * N * lambda_
    a = tx.T @ tx + lambda_prim * np.eye(D)
    b = tx.T @ y
    return np.linalg.solve(a, b)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    ...

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    initial_w : numpy.ndarray
         Initial parameters
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO: implement and document what the function does
    raise NotImplementedError()

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    ...

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    lambda_ : float
         Trade-off parameter
    initial_w : numpy.ndarray
         Initial parameters
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO: implement and document what the function does
    raise NotImplementedError()