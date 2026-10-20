import numpy as np


def subsetting_data(dataset, featurelist):
    """Return a subset of the dataset containing only the requested features.

    Args:
        dataset (dict): Mapping from feature names to arrays.
        featurelist (list[str]): Feature names to keep.

    Returns:
        dict: A dictionary containing only the requested features that exist in the dataset.
    """
    return {feature: dataset[feature] for feature in featurelist if feature in dataset}


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the given split ratio using a single permutation.

    Parameters:
    - x: numpy array of shape (N, D) or (N,)
    - y: numpy array of shape (N,)
    - ratio: float, fraction of data for training (e.g., 0.8)
    - seed: int, random seed for reproducibility

    Returns:
    - x_tr, y_tr: training split
    - x_te, y_te: test split
    """
    np.random.seed(seed)
    num_rows = len(y)

    # Generate a single random permutation of indices
    indices = np.random.permutation(num_rows)

    # Determine the split index
    split_idx = int(np.floor(ratio * num_rows))

    # Partition indices into training and testing sets
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # Select data using the permuted index arrays
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    x_te = x[test_idx]
    y_te = y[test_idx]

    return x_tr, y_tr, x_te, y_te


def shape(dataset):
    num_cols = len(dataset)
    num_rows = dataset[list(dataset.keys())[0]].shape[0] if num_cols > 0 else 0
    return (num_rows, num_cols)

def subsetting_data (dataset, featurelist):
    '''
    expect dataset in column major order as a dictionary of np.arrays, while featureset should be a list of strings that spans dataset.keys()
    '''
    sub_dataset = {}
    for column_name in featurelist:
        if column_name in dataset:
            sub_dataset[column_name] = dataset[column_name]

    return sub_dataset

def parse_dict_to_matrix(data_dict):
    """
    Parses a column-major dictionary of string arrays into a numeric (N, D) matrix.
    Handles empty strings, non-numeric values, and performs mean imputation.
    
    Args:
        data_dict (dict): A dictionary where keys are feature names and 
                          values are 1D NumPy string/object arrays of length N.
                          
    Returns:
        tx (np.ndarray): A numeric matrix of shape (N, D) where D = len(data_dict).
        feature_names (list): A list of the keys in the exact order of the columns.
    """
    feature_columns = []
    feature_names = list(data_dict.keys())
    
    # Check if dict is empty
    if not feature_names:
        return np.empty((0, 0)), []
        
    for key in feature_names:
        # 1. Clean whitespaces
        str_arr = np.char.strip(data_dict[key].astype(str))
        
        # 2. Identify missing or empty records
        # Catches completely empty strings or typical text placeholders
        missing_mask = (str_arr == '') | (str_arr == 'NA') | (str_arr == 'NaN') | (str_arr == '?')
        
        # 3. Safely initialize a float array
        float_arr = np.zeros(str_arr.shape, dtype=float)
        
        # 4. Parse the valid string numbers into the float array
        try:
            float_arr[~missing_mask] = str_arr[~missing_mask].astype(float)
        except ValueError:
            # Fallback if there are unexpected characters (e.g., mixing characters and numbers)
            # This loops only through unique entries to keep it highly optimized
            for unique_val in np.unique(str_arr[~missing_mask]):
                try:
                    float(unique_val)
                except ValueError:
                    # Treat unexpected strings as missing data
                    missing_mask |= (str_arr == unique_val)
            # Retry population with newly expanded missing mask
            float_arr[~missing_mask] = str_arr[~missing_mask].astype(float)
            
        # 5. Populate missing positions with NaN for tracking
        float_arr[missing_mask] = np.nan
        
        # 6. Mean Imputation
        nan_mask = np.isnan(float_arr)
        if np.any(nan_mask):
            if np.any(~nan_mask):
                valid_mean = np.mean(float_arr[~nan_mask])
            else:
                valid_mean = 0.0  # Safe fallback if an entire column is empty
            float_arr[nan_mask] = valid_mean
            
        feature_columns.append(float_arr)
        
    # 7. Stack horizontally into an (N, D) matrix
    tx = np.column_stack(feature_columns)
    
    return tx, feature_names

def compute_covariance_matrix(tx):
    """
    Computes the (D, D) covariance matrix for an (N, D) feature matrix.
    """
    # 1. Center the features by subtracting the mean of each column
    mean_vector = np.mean(tx, axis=0)
    tx_centered = tx - mean_vector
    
    # 2. Compute the sample covariance matrix: (tx_centered.T @ tx_centered) / (N - 1)
    N = tx.shape[0]
    covariance_matrix = (tx_centered.T @ tx_centered) / (N - 1)
    
    return covariance_matrix

def compute_correlation_matrix(tx):
    """
    Computes the (D, D) Pearson correlation matrix from the covariance matrix.
    """
    cov_matrix = compute_covariance_matrix(tx)
    
    # Extract the standard deviations (square root of the diagonal variances)
    std_devs = np.sqrt(np.diag(cov_matrix))
    
    # Handle any constant features (variance = 0) to avoid division by zero
    std_devs[std_devs == 0] = 1e-9
    
    # Correlation = Cov(X, Y) / (Std(X) * Std(Y))
    # Using outer product to create a (D, D) denominator matrix
    correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    return correlation_matrix