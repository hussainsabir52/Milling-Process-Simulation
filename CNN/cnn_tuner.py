import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from keras_tuner import Hyperband
import json

# Sample data generation for demonstration
import numpy as np
X_train = np.load("./data/cnn_data/X_train_windows.npy")
y_train_dx = np.load("./data/cnn_data/y_train_windows_dx.npy")
X_test = np.load("./data/cnn_data/X_test_windows.npy")
y_test_dx = np.load("./data/cnn_data/y_test_windows_dx.npy")

# Define a function to build the model
def build_model(hp):
    model = Sequential()

    # Add Conv1D layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(layers.Conv1D(
            filters=hp.Choice(f'filters_{i}', [16, 32, 64]),
            kernel_size=hp.Choice(f'kernel_size_{i}', [3, 5, 7]),
            activation='relu',
            padding='same'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(
            pool_size=hp.Choice(f'pool_size_{i}', [2, 3])
        ))
    model.add(layers.Attention())
    
    model.add(layers.Flatten())
    
    # Add Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Choice(f'units_{i}', [32, 64, 128]),
            activation='relu'
        ))
        model.add(layers.BatchNormalization())

    
    # Output layer
    model.add(layers.Dense(2))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

# Initialize the Keras Tuner
tuner = Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=20,
    directory='/tmp/tb_logs',
    project_name='cnn_regression'
)

# Perform hyperparameter search
tuner.search(
    X_train, y_train_dx,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")
    
# Save best parameters to a JSON file
with open('best_params_cnn.json', 'w') as f:
    json.dump(best_hps.values, f, indent=4)

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
best_model.fit(
    X_train, y_train_dx,
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)
best_model.save("best_cnn_regression_model.h5")

y_pred_dx=best_model.predict(X_test)

np.save("./data/cnn_data/y_pred_best_dx.npy", y_pred_dx)
