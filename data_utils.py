import numpy as np

# Ref: cs231 assignment 2 template https://cs231n.github.io/assignments2017/assignment2/

def get_data(num_training=49000, num_validation=1000, num_test=10000, num_dev=500, mode='standard'):
    """
        mode: Mode of data preprocessing, value can be "mean_sub" or 'None' or "standard" (the default)
    """

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    # Load the raw data
    X_train = np.load("datasets/train_data.npy")
    y_train = np.load("datasets/train_label.npy")
    X_test = np.load("datasets/test_data.npy")
    y_test = np.load("datasets/test_label.npy")
    
    # convert column vector to row vector for convenience
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    if mode == 'standard':
        mean_image = np.mean(X_train, axis=0)
        std_image = np.std(X_train, axis=0)
        X_train = (X_train - mean_image) / std_image
        X_val = (X_val - mean_image) / std_image
        X_test = (X_test - mean_image) / std_image
        X_dev = (X_dev - mean_image) / std_image
    elif mode == 'mean_sub':
        mean_image = np.mean(X_train, axis=0)
        X_train = X_train - mean_image
        X_val = X_val - mean_image
        X_test = X_test - mean_image
        X_dev = X_dev - mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

