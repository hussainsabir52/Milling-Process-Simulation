import pandas as pd
import numpy as np
import os


clustersim_lhs = pd.read_excel('../clustersim_lhs.xlsx', sheet_name='Zuordnung_Messdaten')
source_folder = "../data/scaled_data_max_abs"

for index in clustersim_lhs.index:
    target_folder = f"../data/rf_test_data/{index}"
    try:
        data=np.loadtxt(f"{source_folder}/{clustersim_lhs.iloc[index]['Messdatei']}.txt")
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
        
        X_test=data_X
        y_test=data[:, [-2,-1]]
        
        os.makedirs(target_folder, exist_ok=True)
        np.savetxt(f"{target_folder}/X_test.txt", X_test)
        np.savetxt(f"{target_folder}/y_test.txt", y_test)
            
    except FileNotFoundError:
        pass