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


# Additional project models. The six required functions above are unchanged.

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def balanced_sample_weights(y, class_weight=None):
    """Per-FIT-split weights, always mean-normalized to 1.

    Equalize total class contribution for 'balanced': weight_c=N/(2*N_c).
    No oversampling; weights never use validation/threshold/test labels.
    """
    y = np.asarray(y, dtype=np.int8)
    if class_weight is None:
        return np.ones(len(y), dtype=np.float32)
    if class_weight != 'balanced':
        raise ValueError("class_weight must be None or 'balanced'")
    n_pos = int(np.sum(y==1)); n_neg = len(y)-n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError('Balanced training needs both classes in fit split')
    return np.where(y==1, len(y)/(2*n_pos), len(y)/(2*n_neg)).astype(np.float32)


def fit_logistic(X, y, model='l1', lam=0.001, iterations=800, tol=1e-6,
                 class_weight=None, l1_ratio=0.5):
    """Proximal FISTA: weighted BCE + lam*L1 (or lam/2*L2).

    The intercept is not penalized. For weighted loss, class weights are
    computed from the CURRENT training fold only. The weighting changes the
    fitted score distribution; tune cutoff on a separate tuning split.
    """
    n, d = X.shape
    if n == 0: raise ValueError('Empty training data')
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sw = balanced_sample_weights(y, class_weight)
    # Normalized weights keep the regularization lambda on a comparable scale.
    weighted_prevalence = np.clip(float(np.dot(sw, y)/sw.sum()),1e-6,1-1e-6)
    w = np.zeros(d, dtype=np.float32)
    b = float(np.log(weighted_prevalence/(1-weighted_prevalence)))
    yw, yb = w.copy(), b
    t = 1.0
    # Logistic Hessian <= (1/4N) [X,1]^T diag(sw) [X,1]. Include intercept!
    # Form its (d+1)x(d+1) Gram rather than concatenating a huge matrix.
    weighted_x = X * np.sqrt(sw)[:,None]
    gram = weighted_x.T @ weighted_x / n
    cross = (X.T @ sw)/n
    augmented = np.empty((d+1,d+1),dtype=np.float64)
    augmented[:d,:d] = gram
    augmented[:d,d] = augmented[d,:d] = cross
    augmented[d,d] = float(sw.mean())
    max_eigen = float(np.linalg.eigvalsh(augmented)[-1])
    step = 1.0 / (.25*max(max_eigen,1e-8))
    for it in range(1, iterations + 1):
        err = (sigmoid(X @ yw + yb) - y) * sw
        v = yw - step * (X.T @ err / n)
        new_b = yb - step * float(np.mean(err))
        if model == 'l1':
            new_w = np.sign(v) * np.maximum(np.abs(v) - step*lam, 0.)
        elif model == 'l2':
            new_w = v / (1 + step*lam)
        elif model == 'elasticnet':
            new_w = (np.sign(v) * np.maximum(np.abs(v)-step*lam*l1_ratio,0)
                     / (1+step*lam*(1-l1_ratio)))
        else:
            raise ValueError('model must be l1, l2 or elasticnet')
        movement = (np.linalg.norm(new_w-w)+abs(new_b-b))/(1+np.linalg.norm(w)+abs(b))
        t_next = (1+np.sqrt(1+4*t*t))/2
        a = (t-1)/t_next
        yw, yb = new_w + a*(new_w-w), new_b + a*(new_b-b)
        w, b, t = new_w.astype(np.float32), float(new_b), t_next
        if it >= 30 and movement < tol:
            break
    z = X @ w + b
    bce_individual = np.logaddexp(0, z) - y * z
    weighted_bce = float(np.dot(sw, bce_individual) / n)
    reg = lam*(np.abs(w).sum() if model == 'l1' else .5*np.square(w).sum()) if model != 'elasticnet' else lam*(l1_ratio*np.abs(w).sum() + .5*(1-l1_ratio)*np.square(w).sum())
    err = (sigmoid(z) - y) * sw
    v = w - step*(X.T @ err/n)
    prox = (np.sign(v)*np.maximum(np.abs(v)-step*lam, 0) if model == 'l1'
            else (np.sign(v)*np.maximum(np.abs(v)-step*lam*l1_ratio,0)/(1+step*lam*(1-l1_ratio))
                  if model=='elasticnet' else v/(1+step*lam)))
    stationarity = np.sqrt(np.sum(((w-prox)/step)**2)
                           + float(np.mean(err))**2)
    return {'w':w, 'b':b, 'iterations':it, 'objective':float(weighted_bce+reg),
            'weighted_bce':weighted_bce,
            'prox_gradient_norm':float(stationarity)}


def fit_candidate(X, y, candidate, iterations=800):
    model = candidate['model']
    if model != 'l1_l2':
        return {'estimator':fit_logistic(X,y,model,candidate['lam'],iterations,
                                          class_weight=candidate.get('class_weight'),
                                          l1_ratio=candidate.get('l1_ratio',0.5)),
                'mask':np.ones(X.shape[1], dtype=bool)}
    screen = fit_logistic(X,y,'l1',candidate['lam'],iterations,
                          class_weight=candidate.get('class_weight'))
    mask = np.abs(screen['w']) > 1e-7
    if not np.any(mask):
        mask = np.zeros(X.shape[1], dtype=bool)  # intercept-only fallback
    refit = fit_logistic(X[:,mask],y,'l2',candidate['refit_lam'],iterations,
                         class_weight=candidate.get('class_weight'))
    return {'estimator':refit,'mask':mask}


def probabilities(X, fitted):
    state = fitted['estimator']
    return sigmoid(X[:,fitted['mask']] @ state['w'] + state['b'])

