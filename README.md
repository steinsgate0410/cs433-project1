# EPFL CS433 Project 1

Shared workspace for the Fall 2026 heart disease classification project.
The current contribution implements the six required algorithms using NumPy.
Data preparation, model comparison, validation, and the final prediction entry
point still need to be implemented by the team.

## Implemented functions

The authoritative module for imports and grading is `implementations.py`.
`implementations.ipynb` contains the corresponding notebook version.

| Function | Method |
| --- | --- |
| `mean_squared_error_gd` | Linear regression with batch gradient descent |
| `mean_squared_error_sgd` | Linear regression with stochastic gradient descent, batch size 1 |
| `least_squares` | Least squares with normal equations |
| `ridge_regression` | Ridge regression with normal equations |
| `logistic_regression` | Logistic regression with gradient descent |
| `reg_logistic_regression` | Logistic regression with L2 regularization |

Every function returns `(w, loss)`. Vectors are one-dimensional NumPy arrays:
`y.shape == (N,)`, `w.shape == (D,)`, and `tx.shape == (N, D)`.
The returned squared-error loss is `sum((y - tx @ w) ** 2) / (2 * N)`.
Regularized methods return the data loss without the regularization penalty.
Iterative methods return the loss evaluated at their final weights.

## Using the functions

From another Python file in the repository root:

```python
import numpy as np
from implementations import ridge_regression, logistic_regression

# x_train and x_valid must already be cleaned and transformed consistently.
# Add an intercept column in preprocessing if your model should use a bias.
tx_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
tx_valid = np.column_stack((np.ones(x_valid.shape[0]), x_valid))

# The provided dataset labels are -1 and 1.
w, loss = ridge_regression(y_train, tx_train, lambda_=0.01)
predictions = np.where(tx_valid @ w > 0, 1, -1)

# Logistic regression requires labels 0 and 1.
y_binary = (y_train == 1).astype(float)
w, loss = logistic_regression(
    y_binary,
    tx_train,
    initial_w=np.zeros(tx_train.shape[1]),
    max_iters=1000,
    gamma=0.01,
)
predictions = np.where(tx_valid @ w > 0, 1, -1)
```

This is an integration example, not a complete training script. The numerical
hyperparameters are examples, not values selected through validation.
The functions do not impute missing values, scale features, add an intercept,
split the data, or choose classification thresholds.
For reproducible SGD experiments, set a NumPy random seed before training.

`helpers.py` provides `load_csv_data` and `create_csv_submission`. Download the
course dataset separately and place its CSV files in `dataset/`. Dataset files
are excluded from Git.

## Running the official public tests

Download the course's `grading_tests` directory from:
https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests

Place it in your local project root. It is excluded from Git, following the
course instructions. The grading environment is separate from the libraries
permitted in submitted project code.

If the grading environment does not exist yet:

```bash
conda env create --file=grading_tests/environment.yml --name=project1-grading
```

Activate it, then run these commands from the project root:

```bash
conda activate project1-grading

# Check the six functions: existence, docstrings, and numerical results.
python -m pytest grading_tests/test_project1_public.py --github_link . -v \
  -k 'function_exists or function_has_docstring or mean_squared_error or least_squares or ridge_regression or logistic_regression'

# Run every public check against the local project.
python -m pytest grading_tests/test_project1_public.py --github_link . -v
```

The function selection contains 22 tests, including 10 numerical tests.
With a local directory, `test_github_link_format` is expected to fail: it requires
a GitHub URL identifying a precise commit. The full suite also requires a
`run.py` or `run.ipynb`; the team must implement that entry point to reproduce
the final predictions. Passing public tests does not guarantee hidden-test
coverage or a complete project submission.

To format the project modules with the grader's Black version:

```bash
python -m black implementations.py helpers.py
```

For the final GitHub check, replace `YOUR_USERNAME`, `YOUR_REPOSITORY`, and
`COMMIT_SHA` with the actual values. Run from the grading tools directory so the
temporary clone is outside the submitted repository:

```bash
cd grading_tests
python -m pytest test_project1_public.py \
  --github_link 'https://github.com/YOUR_USERNAME/YOUR_REPOSITORY/tree/COMMIT_SHA' -v
```

## Team workflow

Keep working versions on branches and merge reviewed changes into `main`.
Before starting a new task:

```bash
git switch main
git pull --ff-only
git switch -c feature/your-task
```

After implementing and checking your changes, stage the relevant files,
commit them, and push your branch. For example:

```bash
git add preprocessing.py
git commit -m "Add missing-value preprocessing"
git push -u origin feature/your-task
```

Open a pull request on GitHub to review and merge the branch. Teammates can
import the six functions from `implementations.py` and add preprocessing,
experiments, the prediction entry point, and the report in separate files.
