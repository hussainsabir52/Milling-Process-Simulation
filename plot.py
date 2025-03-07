import numpy as np
import matplotlib.pyplot as plt


data1=np.load('./data/cnn_data/y_test_windows.npy')
# data2=np.load('./data/cnn_data/y_pred_model1.npy')

data2 = np.load('./data/cnn_data/y_pred_pytorch_cnn.npy')

# Plot the first columns of both data sets
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(data1[:, 0], label='Dx Data')
plt.plot(data2[:, 0], label='Prediction', color='orange')
plt.title('Dx')
plt.xlabel('Index')
plt.ylabel('Value')

# Plot the second columns of both data sets
plt.subplot(1, 2, 2)
plt.plot(data1[:, 1], label='Dy Data')
plt.plot(data2[:, 1], label='Prediction', color='orange')
plt.title('Dy')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()