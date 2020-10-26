import numpy as np
import warnings


class MeanSquaredError:
    """Class that implements static methods that calculate
    value and gradient of mean squared error
    """
    @staticmethod
    def calculate(y, tx, w):
        """Calculates value of mean squared error

        Parameters
        ----------
        y : np.ndarray
            Labels
        tx : np.ndarray
            Features
        w : np.ndarray
            Parameters of the model
        """
        e = y - tx @ w
        return .5 * np.mean(e ** 2)

    @staticmethod
    def gradient(y, tx, w):
        """Calculates gradient of mean squared error

        Parameters
        ----------
        y : np.ndarray
            Labels
        tx : np.ndarray
            Features
        w : np.ndarray
            Parameters of the model
        """
        e = y - tx @ w
        N = y.shape[0]
        return -1 / N * (tx.T @ e)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    Parameters
    ----------
    y : np.ndarray
          Labels
    tx : np.ndarray
          Features
    batch_size : int
          Number of datapoints in each batch
    num_batches : int
          Positive integer indicating a number of batches the data to be split into
          [effective value: min(num_batches, math.ceil(len(y) / batch_size))]
    shuffle : bool
          Should data be randomly shuffled before being split into batches
    """
    data_size = len(y)

    shuffled_y = y
    shuffled_tx = tx

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index >= end_index:
            break
        yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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
    weights = initial_w

    for iteration in range(max_iters):
        # Stochastic Gradient Descent step
        batches = batch_iter(y, tx, batch_size=1, num_batches=1)
        for y_batch, tx_batch in batches:
            gradient = MeanSquaredError.gradient(y_batch, tx_batch, weights)
            weights = weights - gamma * gradient
    # Calculate loss
    loss = MeanSquaredError.calculate(y, tx, weights)
    return (weights, loss)


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
    # Calculate loss (regularization is 
    #     excluded from loss, as it's purpose is for training)
    loss = MeanSquaredError.calculate(y, tx, weights)
    return (weights, loss)


def expit(x):
    """Implements sigmoid function that should not produce overflows.
          However, numpy may return warning that this is happening as the function
          is not lazy, but this warning is a false alarm.
    """
    exp_x = np.exp(x)
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    exp_x / (1 + exp_x))


def logistic_regression_grad(y, tx, weights):
    """Calculates gradient of logistic regression

    Parameters
    ----------
    y : np.ndarray
          Labels 
    tx : np.ndarray
          Features
    weights : np.ndarray
          Parameters of logistic regression model
    """
    p = expit(tx @ weights)
    g = (p - y)[:, None] * tx
    return g.mean(0)


def softplus(x):
    """Softplus function with stability optimization"""
    return np.where(x >= 30, x, np.log1p(np.exp(x)))


def logistic_regression_loss(y, tx, weights):
    """Calculates loss of logistic regression

    Parameters
    ----------
    y : np.ndarray
          Labels 
    tx : np.ndarray
          Features
    weights : np.ndarray
          Parameters of logistic regression model
    """
    t = 2 * y - 1
    loss = softplus(-t * (tx @ weights))
    return loss.mean()


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """...
    Train weights minimizing logistic loss function using GD:
    L(w) = \sum_{i=1}^N [y_i \log s(tx_i^T w) + (1 - y_i) \log(1 - s(tx_i^T w))] -> min_w

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
    for i in range(max_iters):
        # calculate gradient
        g = logistic_regression_grad(y, tx, weights) * tx.shape[0]
        # make a GD step
        weights -= gamma * g
    # Calculate loss
    loss = logistic_regression_loss(y, tx, weights) * tx.shape[0]
    return (weights, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """...
    Train weights minimizing logistic loss function with L2 regularizer using GD:
    L(w) = \sum_{i=1}^N [y_i \log s(tx_i^T w) + (1 - y_i) \log(1 - s(tx_i^T w))] + .5 * lambda_ * w^Tw -> min_w

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
    weights = initial_w

    for i in range(max_iters):
        # calculate gradient
        g = logistic_regression_grad(y, tx, weights) * tx.shape[0]
        # add L2 regularizer gradient (lambda_ * w^Tw)
        g += lambda_ * weights
        # make a GD step
        weights -= gamma * g

    # calculate loss (regularization part is excluded from 
    #     loss as it's purpose is mainly for training)
    loss = logistic_regression_loss(y, tx, weights) * tx.shape[0]

    return (weights, loss)


def reg_logistic_regression_history(y, tx, lambda_, initial_w, max_iters, gamma):
    """...
    Train weights minimizing logistic loss function with L2 regularizer using GD:
    L(w) = 1/N \sum_{i=1}^N [y_i \log s(tx_i^T w) + (1 - y_i) \log(1 - s(tx_i^T w))] + lambda_ * w^Tw -> min_w

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
    weights = initial_w
    loss_h = [logistic_regression_loss(y, tx, weights)]

    for i in range(max_iters):
        # calculate gradient
        g = logistic_regression_grad(y, tx, weights)
        # add L2 regularizer gradient (lambda_ * w^Tw)
        g += 2 * lambda_ * weights
        # make a GD step
        weights -= gamma * g

        loss_h.append(logistic_regression_loss(y, tx, weights))

    return (weights, loss_h)


def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, n_epochs, batch_size, gamma, history=False):
    """...
    Train weights minimizing logistic loss function with L2 regularizer using SGD:
    L(w) = 1/N \sum_{i=1}^N [y_i \log s(tx_i^T w) + (1 - y_i) \log(1 - s(tx_i^T w))] + lambda_ * w^Tw -> min_w

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
    n_epochs : int
        number of passes over the data
    batch_size : int
        number of instances in each batch
    gamma : float
         Learning rate
    history : bool
         return weights history if set to True
    """
    weights = initial_w
    loss_h = [logistic_regression_loss(y, tx, weights)]

    for iteration in range(n_epochs):
        # Stochastic Gradient Descent step
        batches = batch_iter(y, tx, batch_size=batch_size, num_batches=len(y) // batch_size)
        for y_batch, tx_batch in batches:
            # calculate gradient
            g = logistic_regression_grad(y_batch, tx_batch, weights)
            # add L2 regularizer gradient (lambda_ * w^Tw)
            g += 2 * lambda_ * weights
            # make a GD step
            weights -= gamma * g

        loss_h.append(logistic_regression_loss(y, tx, weights))

    if not (1e-3 > 1 - loss_h[-1] / loss_h[-2] > 0):
        warnings.warn("Logistic regression didn't converge!")

    return (weights, np.array(loss_h) if history else loss_h[-1])


def lasso_logistic_regression_sgd(y, tx, lambda_, initial_w, n_epochs, batch_size, gamma, history=False):
    """...
    Train weights minimizing logistic loss function with L1 regularizer using GD:
    L(w) = 1/N \sum_{i=1}^N [y_i \log s(tx_i^T w) + (1 - y_i) \log(1 - s(tx_i^T w))] + lambda_ * \sum_{i=1}^D |w_i| -> min_w

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
    n_epochs : int
        number of passes over the data
    batch_size : int
        number of instances in each batch
    gamma : float
         Learning rate
    history : bool
         return weights history if set to True
    """
    weights = initial_w
    loss_h = [logistic_regression_loss(y, tx, weights)]

    for iteration in range(n_epochs):
        # Stochastic Gradient Descent step
        batches = batch_iter(y, tx, batch_size=batch_size, num_batches=len(y) // batch_size)
        for y_batch, tx_batch in batches:
            # calculate gradient
            g = logistic_regression_grad(y_batch, tx_batch, weights)
            # add L1 regularizer gradient (lambda_ * ||w||_1)
            g += lambda_ * np.sign(weights)
            # make a GD step
            weights -= gamma * g

        loss_h.append(logistic_regression_loss(y, tx, weights))

    return (weights, loss_h if history else loss_h[-1])
