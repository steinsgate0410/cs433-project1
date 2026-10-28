"""The six CS-433 estimators, implemented with NumPy only.

All vectors are one-dimensional. Every method returns the final weights and
unpenalized mean loss; MSE includes the course's factor of one half.
"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Minimize mean squared error by full-batch gradient descent."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * (tx.T @ (tx @ w - y)) / len(y)
    return w, np.mean((y - tx @ w) ** 2) / 2


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Use one randomly sampled observation per SGD step.

    Set np.random.seed before calling this function to reproduce the draws.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        w -= gamma * tx[i] * (tx[i] @ w - y[i])
    return w, np.mean((y - tx @ w) ** 2) / 2


def least_squares(y, tx):
    """Solve the normal equations by a pseudoinverse, allowing redundant columns."""
    gram, rhs = tx.T @ tx, tx.T @ y
    w = np.linalg.pinv(gram, hermitian=True) @ rhs
    return w, np.mean((y - tx @ w) ** 2) / 2


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_ * ||w||^2; return MSE without the penalty."""
    if lambda_ == 0:
        return least_squares(y, tx)
    gram = tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w = np.linalg.solve(gram, tx.T @ y)
    return w, np.mean((y - tx @ w) ** 2) / 2


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit labels in {0, 1} by gradient descent on mean log loss."""
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Minimize mean log loss + lambda_ * ||w||^2 by gradient descent.

    The penalty applies to every coordinate, as specified by the course.
    logaddexp keeps both the sigmoid and the returned loss numerically stable.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        probability = np.exp(-np.logaddexp(0.0, -(tx @ w)))
        gradient = tx.T @ (probability - y) / len(y) + 2 * lambda_ * w
        w -= gamma * gradient
    logits = tx @ w
    return w, np.mean(np.logaddexp(0.0, logits) - y * logits)
