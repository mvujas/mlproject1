import numpy as np

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

