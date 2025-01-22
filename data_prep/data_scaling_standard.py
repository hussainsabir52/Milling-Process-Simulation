from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as py
import numpy as np
import os

data_folder = "../data/data_trimmed"
target_folder = "../data/scaled_data_max_abs"

def scale_data_max_abs(data):
    scaler = MaxAbsScaler()
    return scaler.fit_transform(data)

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        file_data = np.loadtxt(file_path)
        scaled_data = scale_data_max_abs(file_data)
        np.savetxt(target_folder+"/"+filename, scaled_data)

