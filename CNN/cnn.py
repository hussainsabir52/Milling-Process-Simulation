import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping


n_steps = 20
n_features = 11

X_train = np.load("./data/cnn_data/X_train_windows.npy")
y_train_dx = np.load("./data/cnn_data/y_train_windows_dx.npy")
X_test = np.load("./data/cnn_data/X_test_windows.npy")
y_test_dx = np.load("./data/cnn_data/y_test_windows_dx.npy")

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train_dx, batch_size=64, epochs=25, callbacks=[early_stopping])

model.save("model1_cnn_base.keras")

y_pred_dx = model.predict(X_test)

np.save("./data/cnn_data/y_pred_model1_dx.npy", y_pred_dx)