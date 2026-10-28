"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.loadtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skiprows=1,
        dtype=int,
        usecols=1,
    )

    # Preallocate once rather than retaining millions of Python objects.
    # The distributed numeric CSVs use empty fields for missing values.
    def read_features(filename):
        path = os.path.join(data_path, filename)
        with open(path) as stream:
            columns = len(next(stream).strip().split(","))
            rows = sum(1 for _ in stream)
        values = np.empty((rows, columns))
        with open(path) as stream:
            next(stream)
            for i, line in enumerate(stream):
                line = line.strip().replace(",,", ",nan,").replace(",,", ",nan,")
                if line.endswith(","):
                    line += "nan"
                values[i] = np.fromstring(line, sep=",")
        return values

    x_train = read_features("x_train.csv")
    x_test = read_features("x_test.csv")

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
