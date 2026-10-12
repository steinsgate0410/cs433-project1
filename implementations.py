import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Batch Gradient Descent for Mean Squared Error

    args:
        y: numpy array of shape (N, ) containing the target values
        tx: numpy array of shape (N, D) containing the input features
        initial_w: numpy array of shape (D, ) containing the initial weights
        max_iters: int, number of iterations for gradient descent
        gamma: float, learning rate

    returns:
        w: numpy array of shape (D, ) containing the final weights after gradient descent
        loss: float, the final loss value after gradient descent
    """
    # 1. Initialize parameters (weight, number of samples, loss)
    w = initial_w
    N = len(y)
    loss = 0.0

    # 2. Start the gradient descent loop
    for i in range(max_iters):
        # 2.1 Compute the predictions
        y_pred = tx @ w

        # 2.2 Compute the error
        e = y - y_pred

        # 2.3 Compute the loss
        loss = (1 / (2 * N)) * (e.T @ e)

        # 2.4 Compute the gradient
        grad = -(1 / N) * (tx.T @ e)

        # 2.5 Update the weights
        w = w - gamma * grad

    # 3. Compute the final loss after the last iteration
    y_pred = tx @ w
    e = y - y_pred
    loss = (1 / (2 * N)) * (e.T @ e)

    return (w, loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic Gradient Descent for Mean Squared Error

    args:
        y: numpy array of shape (N, ) containing the target values
        tx: numpy array of shape (N, D) containing the input features
        initial_w: numpy array of shape (D, ) containing the initial weights
        max_iters: int, number of iterations for gradient descent
        gamma: float, learning rate

    returns:
        w: numpy array of shape (D, ) containing the final weights after gradient descent
        loss: float, the final loss value after gradient descent
    """

    # 1. Initialize parameters (weight, number of samples, loss)
    w = initial_w
    N = len(y)
    loss = 0.0

    # 2. Start the gradient descent loop
    for i in range(max_iters):
        # 2.1 Randomly select an index for stochastic gradient descent
        rand_idx = np.random.randint(N)
        y_i = y[rand_idx]
        tx_i = tx[rand_idx]

        # 2.2 Compute the prediction
        y_pred_i = tx_i @ w

        # 2.3 Compute the error
        e_i = y_i - y_pred_i

        # 2.4 Compute the loss
        loss = (1 / 2) * (e_i**2)

        # 2.5 Compute the gradient
        grad = -tx_i * e_i

        # 2.6 Update the weights
        w = w - gamma * grad

    # 3. Compute the final loss after the last iteration
    # This is done on the entire dataset, not just a single sample
    y_pred = tx @ w
    e = y - y_pred
    loss = (1 / (2 * N)) * (e.T @ e)

    return (w, loss)


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse, scalar.
    """

    # 0. get the number of samples
    N = len(y)

    # 1. Directly solve w
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    # 2. Compute the MSE
    e = y - tx @ w
    mse = 1 / (2 * N) * (e.T @ e)
    loss = mse

    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: regularization parameter.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse, scalar.
    """

    # 0. get the number of samples and features
    N = tx.shape[0]
    D = tx.shape[1]

    # 1. get the real lambda for ridge regression
    lambda_real = 2 * N * lambda_

    # 2. Solve for w using the normal equations with regularization
    w = np.linalg.solve(tx.T @ tx + lambda_real * np.eye(D), tx.T @ y)

    # 3. Compute the MSE
    e = y - tx @ w
    mse = 1 / (2 * N) * (e.T @ e)
    loss = mse

    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    y is 0 or 1.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial weights, numpy array of shape(D,), D is the number of features.
        max_iters: number of iterations for gradient descent.
        gamma: learning rate.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: final loss value after gradient descent.
    """

    # 0. Initialize parameters
    w = initial_w
    N = len(y)
    loss = 0.0

    # 1. Start the gradient descent loop
    for i in range(max_iters):
        # 1.1 Compute the predictions using the sigmoid function
        y_pred = 1 / (1 + np.exp(-tx @ w))

        # 1.2 Compute the error
        e = y_pred - y

        # 1.3 Compute the loss (negative log-likelihood)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # 1.4 Compute the gradient
        grad = (1 / N) * (tx.T @ e)

        # 1.5 Update the weights
        w = w - gamma * grad

    # 2. Compute the final loss after the last iteration
    y_pred = 1 / (1 + np.exp(-tx @ w))
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    L2 Regularization
    y is 0 or 1.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: regularization parameter.
        initial_w: initial weights, numpy array of shape(D,), D is the number of features.
        max_iters: number of iterations for gradient descent.
        gamma: learning rate.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: final loss value after gradient descent.
    """

    # 0. Initialize parameters
    w = initial_w
    N = len(y)
    loss = 0.0

    # 1. Start the gradient descent loop
    for i in range(max_iters):
        # 1.1 Compute the predictions using the sigmoid function
        y_pred = 1 / (1 + np.exp(-tx @ w))

        # 1.2 Compute the error
        e = y_pred - y

        # 1.3 Compute the loss (negative log-likelihood with L2 regularization)
        # As required in the note, I do not include the regularization term in the loss computation for this implementation.
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # 1.4 Compute the gradient with L2 regularization
        grad = (1 / N) * (tx.T @ e) + 2 * lambda_ * w

        # 1.5 Update the weights
        w = w - gamma * grad

    # 2. Compute the final loss after the last iteration
    y_pred = 1 / (1 + np.exp(-tx @ w))
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    return (w, loss)
