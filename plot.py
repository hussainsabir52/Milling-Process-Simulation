import numpy as np
import matplotlib.pyplot as plt


data1=np.loadtxt('./data/scaled_data/y_test.txt')
data2=np.loadtxt('./RandomForest/ypred2.txt')


# Plot the first columns of both data sets
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(data1[:, 0], label='Test Data', color='blue')
plt.plot(data2[:, 0], label='Prediction', color='orange')
plt.title('Dx')
plt.xlabel('Index')
plt.ylabel('Value')

# Plot the second columns of both data sets
plt.subplot(1, 2, 2)
plt.plot(data1[:, 1], label='Test Data', color='blue')
plt.plot(data2[:, 1], label='Prediction', color='orange')
plt.title('Dy')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()