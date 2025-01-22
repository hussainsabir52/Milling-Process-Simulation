import numpy as np

def reshape_data():
    X_train = np.loadtxt("./data/scaled_data/X_train.txt")
    y_train = np.loadtxt("./data/scaled_data/y_train.txt")
    X_test = np.loadtxt("./data/scaled_data/X_test.txt")
    y_test= np.loadtxt("./data/scaled_data/y_test.txt")

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train_dx = y_train[:,0].reshape(y_train[:,0].shape[0], 1)
    y_test_dx = y_test[:,0].reshape(y_test[:,0].shape[0], 1)
    y_train_dy = y_train[:,1].reshape(y_train[:,1].shape[0], 1)
    y_test_dy = y_test[:,1].reshape(y_test[:,1].shape[0], 1)

    np.save("./data/cnn_data/X_train.npy", X_train)
    np.save("./data/cnn_data/X_test.npy", X_test)
    np.save("./data/cnn_data/y_train_dx.npy", y_train_dx)
    np.save("./data/cnn_data/y_test_dx.npy", y_test_dx)
    np.save("./data/cnn_data/y_train_dy.npy", y_train_dy)
    np.save("./data/cnn_data/y_test_dy.npy", y_test_dy)



def split_sequences_with_targets(X, y1, y2, n_steps):
    """
    Split sequences into overlapping windows and align them with target values.
    
    Args:
        X (np.array): 2D array (samples, features)
        y (np.array): 1D array (samples,)
        n_steps (int): Number of time steps per window
    
    Returns:
        X_windows (np.array): 3D array (samples - n_steps + 1, n_steps, features)
        y_windows (np.array): 1D array (samples - n_steps + 1,)
    """
    X_windows, y_windows_dx, y_windows_dy = [], [], []
    
    for i in range(len(X) - n_steps + 1):
        # Extract a window of X
        window = X[i:i + n_steps, :]
        X_windows.append(window)
        
        # Take the last corresponding y value
        y_windows_dx.append(y1[i + n_steps - 1])
        y_windows_dy.append(y2[i + n_steps - 1])
    
    return np.array(X_windows), np.array(y_windows_dx), np.array(y_windows_dy)

X_train = np.loadtxt("./data/scaled_data/X_train.txt")
y_train= np.loadtxt("./data/scaled_data/y_train.txt")
y_train_dx= y_train[:,0]
y_train_dy= y_train[:,1]

X_test = np.loadtxt("./data/scaled_data/X_test.txt")
y_test= np.loadtxt("./data/scaled_data/y_test.txt")
y_test_dx= y_test[:,0]
y_test_dy= y_test[:,1]

# choose a number of time steps.
n_steps = 20
# convert into input/output
X_test_windows,y_test_windows_dx, y_test_windows_dy = split_sequences_with_targets(X_test, y_test_dx, y_test_dy, n_steps)
X_train_windows,y_train_windows_dx, y_train_windows_dy = split_sequences_with_targets(X_train,y_train_dx, y_train_dy, n_steps)

np.save("./data/cnn_data/X_train_windows.npy", X_train_windows)
np.save("./data/cnn_data/y_train_windows_dx.npy", y_train_windows_dx)
np.save("./data/cnn_data/y_train_windows_dy.npy", y_train_windows_dy)
np.save("./data/cnn_data/X_test_windows.npy", X_test_windows)
np.save("./data/cnn_data/y_test_windows_dx.npy", y_test_windows_dx)
np.save("./data/cnn_data/y_test_windows_dy.npy", y_test_windows_dy)