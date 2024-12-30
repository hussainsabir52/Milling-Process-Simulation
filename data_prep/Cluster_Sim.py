import numpy as np
from scipy import signal
from scipy.signal import resample
import matplotlib.pyplot as py
import pandas as pd
from dotenv import load_dotenv
import os
from matplotlib.widgets import Button, Slider

load_dotenv()


class Cluster_Sim:
    def __init__(self):
        self.path_data={}
        self.vibration_data={}
        self.decimated_combined_data={}
        self.start_index=0
        self.end_index=0
        self.columns=['Fx','Fy','Fz','Mic','Spindel_Wi','Dx','Dy']
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
            data = np.loadtxt(f'{os.getenv('PATH_TO_RAW_DATA')}/Cluster_Sim_{filename}.txt', skiprows=8, usecols=[1, 2, 3, 4, 7])
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
                data_combined = np.append(data_combined, vibrations_dy, axis=1)
                np.savetxt(f'{os.getenv('PATH_TO_DECIMATED_DATA')}/{Messdatei}.txt', data_combined, delimiter=" ")
                self.decimated_combined_data[Messdatei]=data_combined

    def load_decimated_data(self):
        for filename in os.listdir(os.getenv('PATH_TO_DECIMATED_DATA')):
            data=np.loadtxt(os.path.join(os.getenv('PATH_TO_DECIMATED_DATA'), filename))
            self.decimated_combined_data[filename.split(".")[0]]=data

    def load_single_decimated_data(self, filename):
        data=np.loadtxt(os.path.join(os.getenv('PATH_TO_DECIMATED_DATA'), filename))
        return data

    def remove_initial_idle_period(self,data, threshold=0.1):
            new_data=[]
            rolling_std = np.array([np.std(data[i:i+100]) for i in range(len(data) - 100 + 1)])
            rolling_std = np.pad(rolling_std, (100-1, 0), mode='constant', constant_values=np.nan)
            start_idx = np.argmax(rolling_std > threshold)
            if rolling_std[start_idx] > threshold:
                new_data = data[start_idx-100:]
            else:
                new_data= data
            data=new_data[::-1]
            rolling_std = np.array([np.std(data[i:i+100]) for i in range(len(data) - 100 + 1)])
            rolling_std = np.pad(rolling_std, (100-1, 0), mode='constant', constant_values=np.nan)
            end_idx = np.argmax(rolling_std > threshold)
            if rolling_std[end_idx] > threshold:
                new_data = data[end_idx-100:]
                new_data = new_data[::-1]
            else:
                new_data= data[::-1]
            return new_data,start_idx,end_idx
    
    def trim_data(self, data2D, start_index, end_index,filename):
        data=data2D[start_index:end_index]
        np.savetxt(f'{os.getenv('PATH_TO_TRIMMED_DATA')}/{filename}.txt', data, delimiter=" ")
        
    def plot_decimated_data(self, data):
        data=np.delete(data,4,1)
        data=np.delete(data,5,1)
        fig, ax = py.subplots(data.shape[1])
        for i in range(data.shape[1]):
            ax[i].plot(data[:,i])
            ax[i].set_ylabel(self.columns[i])
        py.show()

    def readDataAndDecimate(self):
        for i in range(len(self.clustersim_lhs)):
            Messdatei=self.clustersim_lhs['Messdatei'][i]
            self.downsample_decimate(Messdatei,5)    

# def update(val):
#     data,cluster_sim_obj.start_index, cluster_sim_obj.end_index = cluster_sim_obj.remove_initial_idle_period(fx, freq_slider.val)
#     if data.shape != line.get_ydata().shape:
#         line.set_data(np.arange(len(data)), data)
#     else:
#         line.set_ydata(data)
#     fig.canvas.draw_idle()
        
# def save(event):
#     cluster_sim_obj.trim_data(data,cluster_sim_obj.start_index,data.shape[0]-cluster_sim_obj.end_index-1,"V0_"+str(int(value)))

# cluster_sim_obj=Cluster_Sim()
# #cluster_sim_obj.load_decimated_data()
# for value in np.loadtxt("missed.txt"):
#     data=cluster_sim_obj.load_single_decimated_data("V0_"+str(int(value))+".txt")
#     fx=data[:,0]
#     fig, ax = py.subplots()
#     fig.subplots_adjust(left=0.25, bottom=0.25)    
#     line,=ax.plot(fx)
#     axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
#     freq_slider = Slider(
#         ax=axfreq,
#         label='Frequency [Hz]',
#         valmin=2,
#         valmax=5,
#         valinit=2,
#         valstep=1
#     )
#     freq_slider.on_changed(update)
    
#     resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
#     button = Button(resetax, 'Save', hovercolor='0.975')
#     button.on_clicked(save)
#     py.show()
