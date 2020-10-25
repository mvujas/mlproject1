from implementations import *
import numpy as np

def reg_logistic_neuralnet_sgd(y, tx, lambda_, n_epochs, batch_size, gamma, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = []
    assert(y.ndim == 1)
    layers_size = [tx.shape[-1]] + hidden_layers + [1]
    layers_num = len(layers_size)

    weights = [np.random.uniform(size=(layers_size[i], layers_size[i + 1])) for i in range(layers_num - 1)]

    forward_pass = [None] * layers_num
    
    weight_gradients = [None] * len(weights)
    layer_gradients = [None] * (layers_num - 1)
    
    # Forward pass
    forward_pass[0] = tx
    for i in range(layers_num - 1):
        forward_pass[i + 1] = forward_pass[i] @ weights[i]
        if i == layers_num - 2:
            forward_pass[i + 1] = expit(forward_pass[i + 1])[:, 0]



    loss = logistic_regression_loss(
        forward_pass[-1], 
        forward_pass[-2], 
        weights[-1][:, 0])

    weight_gradients[-1] = logistic_regression_grad(
        forward_pass[-1], 
        forward_pass[-2], 
        weights[-1][:, 0]
    )
    layer_gradients[-1] = logistic_regression_grad(
        forward_pass[-1], 
        weights[-1],
        forward_pass[-2].T 
    )

    print(layer_gradients)

    '''
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

        loss_h.append(logistic_regression_loss(y, tx, weights) + lambda_ * weights @ weights)

    if not (1e-3 > 1 - loss_h[-1] / loss_h[-2] > 0):
        warnings.warn("Logistic regression didn't converge!")

    return (weights, np.array(loss_h) if history else loss_h[-1])
    '''

y = np.array([
    0, 1, 1, 0, 0
])

x = np.array([
    [1, 2, 3],
    [1, 4, 5],
    [1, 6, 7],
    [1, 8, 9],
    [1, 10, 11]
])

reg_logistic_neuralnet_sgd(y, x, 1e-5, 5, 10, 0.1, hidden_layers=[1, 2, 3])
