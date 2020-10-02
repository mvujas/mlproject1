import numpy as np
from losses import MeanSquaredError

# TODO : all functions should return (w, loss) pair

def least_squares(y, tx):
    """Calculates the least squares solution using normal equations.
    Returns tuple (parameters, loss)

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    """
    # Calculate parameters
    a = tx.T @ tx
    b = tx.T @ y
    weights = np.linalg.solve(a, b)
    # Calculate loss
    loss = MeanSquaredError.calculate(y, tx, weights)
    return (weights, loss)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Calculates the least squares solution using gradient descent.
    Returns tuple (parameters, loss)

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    initial_w : numpy.ndarray
         Initial parameters of the model
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    weights = initial_w
    for iteration in range(max_iters):
        # Gradient Descent step
        gradient = MeanSquaredError.gradient(y, tx, weights)
        weights = weights - gamma * gradient
    # Calculate loss
    loss = MeanSquaredError.calculate(y, tx, weights)
    return (weights, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Calculates the least squares solution using stochastic gradient descent.
    Returns tuple (parameters, loss)

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    initial_w : numpy.ndarray
         Initial parameters of the model
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO : check whether we can add and delete arguments
    raise NotImplementedError()

def ridge_regression(y, tx, lambda_):
    """Implements ridge regression.
    Returns tuple (parameters, loss)

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
    # Calculate parameters
    lambda_prim = 2 * N * lambda_
    a = tx.T @ tx + lambda_prim * np.eye(D)
    b = tx.T @ y
    weights = np.linalg.solve(a, b)
    # Calculate loss
    mse = MeanSquaredError.calculate(y, tx, weights)
    loss = mse + lambda_ * np.sum(weights ** 2)
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """...
    Returns tuple (parameters, loss)

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    initial_w : numpy.ndarray
         Initial parameters of the model
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO: implement and document what the function does
    raise NotImplementedError()

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """...
    Returns tuple (parameters, loss)

    Parameters
    ----------
    y : numpy.ndarray
         Labels
    tx : numpy.ndarray
         Features
    lambda_ : float
         Trade-off parameter
    initial_w : numpy.ndarray
         Initial parameters of the model
    max_iters : int
         Maximum number of iterations
    gamma : float
         Learning rate 
    """
    # TODO: implement and document what the function does
    raise NotImplementedError()