import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as py
import pandas as pd

clustersim_lhs = pd.read_excel('clustersim_lhs.xlsx', sheet_name='Zuordnung_Messdaten')

clustersim_lhs_train, clustersim_lhs_test= train_test_split(clustersim_lhs.index, test_size=0.3, train_size=0.7)

X_train=[]
y_train=[]
i=0
for index in clustersim_lhs_train:
    try:
        data=np.loadtxt(f"data_trimmed/{clustersim_lhs.iloc[index]['Messdatei']}.txt")
        n=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['n'],
            dtype=np.float32
            )
        fz=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['fz'],
            dtype=np.float32
            )
        ae=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['ae'],
            dtype=np.float32
            )
        eingabe_ae=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['Eingabe ae'],
            dtype=np.float32
            )
        R1=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['R1'],
            dtype=np.float32
            )
        R2=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['R2'],
            dtype=np.float32
            )
       
        presets=np.vstack((n,fz,ae,eingabe_ae,R1,R2)).T
        data_X=np.concatenate((presets, data[:, :5]), axis=1)
        if i==0:
            print(i) 
            X_train=data_X
            y_train=data[:,[-2,-1]]
        else:
            X_train = np.vstack((X_train, data_X))
            y_train = np.vstack((y_train,data[:,[-2,-1]]))
        i=i+1
        
    except FileNotFoundError:
        pass

np.savetxt("X_train.txt", X_train, delimiter=" ")
np.savetxt("y_train.txt", y_train, delimiter=" ")




X_test=[]
y_test=[]
i=0
for index in clustersim_lhs_test:
    try:
        data=np.loadtxt(f"data_trimmed/{clustersim_lhs.iloc[index]['Messdatei']}.txt")
        n=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['n'],
            dtype=np.float32
            )
        fz=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['fz'],
            dtype=np.float32
            )
        ae=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['ae'],
            dtype=np.float32
            )
        eingabe_ae=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['Eingabe ae'],
            dtype=np.float32
            )
        R1=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['R1'],
            dtype=np.float32
            )
        R2=np.full(
            shape=data.shape[0],
            fill_value=clustersim_lhs.iloc[index]['R2'],
            dtype=np.float32
            )
       
        presets=np.vstack((n,fz,ae,eingabe_ae,R1,R2)).T
        data_X=np.concatenate((presets, data[:, :5]), axis=1)
        if i==0:
            print(i) 
            X_test=data_X
            y_test=data[:, [-2,-1]]
        else:
            X_test = np.vstack((X_test, data_X))
            y_test = np.vstack((y_test,data[:, [-2,-1]]))
        i=i+1
        
    except FileNotFoundError:
        pass


np.savetxt("X_test.txt", X_test, delimiter=" ")
np.savetxt("y_test.txt", y_test, delimiter=" ")
