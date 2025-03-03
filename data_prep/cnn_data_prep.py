import numpy as np

def reshape_data():
    X_train = np.loadtxt("./data/scaled_data_max_abs/X_train.txt")
    y_train = np.loadtxt("./data/scaled_data_max_abs/y_train.txt")
    X_test = np.loadtxt("./data/scaled_data_max_abs/X_test.txt")
    y_test= np.loadtxt("./data/scaled_data_max_abs/y_test.txt")

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], 2)
    y_test = y_test.reshape(y_test.shape[0], 2)

    np.save("./data/cnn_data/X_train.npy", X_train)
    np.save("./data/cnn_data/X_test.npy", X_test)
    np.save("./data/cnn_data/y_train_dx.npy", y_train)
    np.save("./data/cnn_data/y_test_dx.npy", y_test)


def split_sequences_with_targets(X, y, n_steps):
    """
    Split sequences into overlapping windows and align them with target values.
    
    Args:
        X (np.array): 2D array (samples, features)
        y (np.array): 2D array (samples, 2) - target variables
        n_steps (int): Number of time steps per window
    
    Returns:
        X_windows (np.array): 3D array (samples - n_steps + 1, n_steps, features)
        y_windows (np.array): 2D array (samples - n_steps + 1, 2)
    """
    X_windows, y_windows = [], []
    
    for i in range(len(X) - n_steps + 1):
        # Extract a window of X
        window = X[i:i + n_steps, :]
        X_windows.append(window)
        
        # Take the last corresponding y value
        y_windows.append(y[i + n_steps - 1])
    
    return np.array(X_windows), np.array(y_windows)

X_train = np.loadtxt("./data/data_split/X_train.txt")
y_train= np.loadtxt("./data/data_split/y_train.txt")

X_test = np.loadtxt("./data/data_split/X_test.txt")
y_test= np.loadtxt("./data/data_split/y_test.txt")

# choose a number of time steps.
n_steps = 20
# convert into input/output
X_test_windows,y_test_windows = split_sequences_with_targets(X_test, y_test, n_steps)
X_train_windows,y_train_windows = split_sequences_with_targets(X_train, y_train, n_steps)

np.save("./data/cnn_data/X_train_windows.npy", X_train_windows)
np.save("./data/cnn_data/y_train_windows.npy", y_train_windows)
np.save("./data/cnn_data/X_test_windows.npy", X_test_windows)
np.save("./data/cnn_data/y_test_windows.npy", y_test_windows)