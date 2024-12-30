from sklearn.ensemble import RandomForestRegressor
import numpy as np

X_train = np.loadtxt("../data/scaled_data/X_train.txt")
y_train = np.loadtxt("../data/scaled_data/y_train.txt")
X_test = np.loadtxt("../data/scaled_data/X_test.txt")
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train[:,0])
y_pred = regressor.predict(X_test)

np.savetxt("../data/scaled_data/y_pred.txt", y_pred)