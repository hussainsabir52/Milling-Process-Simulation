from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import matplotlib.pyplot as plt



def model_train():
    # Generate a random regression problem
    X_train = np.loadtxt("../data/data_split/X_train.txt")
    y_train = np.loadtxt("../data/data_split/y_train.txt")
    X_test = np.loadtxt("../data/data_split/X_test.txt")
    y_test = np.loadtxt("../data/data_split/y_test.txt")

    # Define the RandomForestRegressor with the specified parameters
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        min_samples_split=24,
        min_samples_leaf=3,
        max_features='log2',
        max_depth=10,
        bootstrap=True,
        random_state=42
    )

    # Fit the model to the training data
    rf_regressor.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(rf_regressor, 'best_model.pkl')
    # Predict on the test data
    y_pred = rf_regressor.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
def model_test():
    # Load the model from the file
    rf_regressor = joblib.load('best_model.pkl')

    for i in range(52, 64):
        # Load the test data
        X_test = np.loadtxt(f"../data/rf_test_data/{i}/X_test.txt")
        y_test = np.loadtxt(f"../data/rf_test_data/{i}/y_test.txt")

        # Predict on the test data
        y_pred = rf_regressor.predict(X_test)
        
        # Plot the actual vs predicted values for Dx
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(y_test[:, 0], label='Actual Dx')
        plt.plot(y_pred[:, 0], label='Predicted Dx')
        plt.xlabel('Sample Index')
        plt.ylabel('Dx Value')
        plt.title(f'Actual vs Predicted Dx for Test Set {i}')
        plt.legend()
        
        # Plot the actual vs predicted values for Dy
        plt.subplot(1, 2, 2)
        plt.plot(y_test[:, 1], label='Actual Dy')
        plt.plot(y_pred[:, 1], label='Predicted Dy')
        plt.xlabel('Sample Index')
        plt.ylabel('Dy Value')
        plt.title(f'Actual vs Predicted Dy for Test Set {i}')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot as an image file
        plt.savefig(f'../data/rf_test_data/{i}/test_set_{i}_comparison.png')

        # Calculate the mean squared error
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        
model_test()