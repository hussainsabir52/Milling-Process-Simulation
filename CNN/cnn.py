import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


n_steps = 20
n_features = 11

X_train = np.load("./data/cnn_data/X_train_windows.npy")
y_train = np.load("./data/cnn_data/y_train_windows.npy")
X_test = np.load("./data/cnn_data/X_test_windows.npy")
y_test = np.load("./data/cnn_data/y_test_windows.npy")

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(n_steps, n_features, 1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2))


optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

model.fit(X_train, y_train, batch_size=64, epochs=10, callbacks=[early_stopping])

y_pred = model.predict(X_test)

np.save("./data/cnn_data/y_pred_model1.npy", y_pred)

model.save("model1_cnn_base.keras")
