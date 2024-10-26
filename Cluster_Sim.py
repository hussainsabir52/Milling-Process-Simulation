import numpy as np
from scipy import signal
from scipy.signal import resample
import matplotlib.pyplot as py
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()


class Cluster_Sim:
    def __init__(self):
        self.path_data={}
        self.vibration_data={}
        self.decimated_combined_data={}
        self.clustersim_lhs= pd.read_excel(os.getenv('PATH_TO_EXCEL'), sheet_name=os.getenv('EXCEL_SHEET_NAME'))

    def populate_data(self):
        for i in range(len(self.clustersim_lhs)):
            Messdatei=self.clustersim_lhs['Messdatei'][i]
            data,vibration = self.load_file_data(Messdatei)
            self.path_data[Messdatei]=data
            self.vibration_data[Messdatei]=vibration

    def load_file_data(self, filename):
        try:
            print(filename)
            data = np.loadtxt(f'{os.getenv('PATH_TO_RAW_DATA')}/Cluster_Sim_{filename}.txt', skiprows=8, usecols=[1, 2, 3, 4, 5, 6, 7])
            vibrations = np.load(f'{os.getenv('PATH_TO_FILTERED_DATA')}/Cluster_Sim_{filename}_filtered.npz')
        except FileNotFoundError:
            return [],[]
        else:
            return data,vibrations  

    def downsample_resample(self, filename, downsample_factor):
        data = np.loadtxt(f'{os.getenv('PATH_TO_RAW_DATA')}/Cluster_Sim_{filename}.txt', skiprows=8, usecols=[1, 2, 3, 4, 5, 6, 7])
        num_samples_downsampled = len(data) // downsample_factor  # New number of samples
        data_downsampled = resample(data, num_samples_downsampled)
        return data, data_downsampled


    def downsample_decimate(self,downsample_factor):
        for i in range(len(self.clustersim_lhs)):
            Messdatei=self.clustersim_lhs['Messdatei'][i]
            if(len(self.path_data[Messdatei])>27):
                data_downsampled = signal.decimate(self.path_data[Messdatei], downsample_factor, axis=0)
                vibrations_dx = signal.decimate(self.vibration_data[Messdatei]['dx'], downsample_factor, axis=0)
                vibrations_dy = signal.decimate(self.vibration_data[Messdatei]['dy'], downsample_factor, axis=0)
                vibrations_dx=vibrations_dx.reshape(vibrations_dx.shape[0],1)
                vibrations_dy=vibrations_dy.reshape(vibrations_dy.shape[0],1)
                data_combined = np.append(data_downsampled, vibrations_dx, axis=1)
                data_combined = np.append(data_downsampled, vibrations_dy, axis=1)
                np.savetxt(f'{os.getenv('PATH_TO_DECIMATED_DATA')}/{Messdatei}.txt', data_combined, delimiter=" ")
                self.decimated_combined_data[Messdatei]=data_combined

    def load_decimated_data(self):
        for filename in os.listdir(os.getenv('PATH_TO_DECIMATED_DATA')):
            data=np.loadtxt(os.path.join(os.getenv('PATH_TO_DECIMATED_DATA'), filename))
            self.decimated_combined_data[filename]=data
        
    def remove_initial_idle_period(self,data, window_size=100, threshold=0.1):
        rolling_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data) - window_size + 1)])
        rolling_std = np.pad(rolling_std, (window_size-1, 0), mode='constant', constant_values=np.nan)
        start_idx = np.argmax(rolling_std > threshold)
        if rolling_std[start_idx] > threshold:
            trimmed_data = data[start_idx-100:]
            return trimmed_data, start_idx
        else:
            return data, None
        

    def readDataAndDecimate(self):
        for i in range(len(self.clustersim_lhs)):
            Messdatei=self.clustersim_lhs['Messdatei'][i]
            self.downsample_decimate(Messdatei,5)

if __name__ == "__main__":
    cluster_sim_obj=Cluster_Sim()
    cluster_sim_obj.load_decimated_data()
