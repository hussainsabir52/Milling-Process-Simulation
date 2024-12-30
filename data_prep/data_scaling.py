from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as py

files = ['X_test.txt','X_train.txt','y_test.txt','y_train.txt']
minMaxScaler = MinMaxScaler()

for filename in files:
    data = np.loadtxt(f"../data/train_test/{filename}")
    minMaxScaler.fit(data)
    scaled_data = minMaxScaler.transform(data)
    np.savetxt(f"../data/scaled_data/{filename}", scaled_data)





