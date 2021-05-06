import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import transform_helper as Transform 
from tqdm import tqdm

if __name__ == "__main__":
    data_path = f'data{os.sep}'
    train_path = data_path + "train.csv"
    
    dtypes = Transform.get_dtypes()
    
    df = pd.read_csv(train_path, dtype=dtypes)
    
    df_columns = list(df.columns)
    
    unique_df = pd.DataFrame(columns=df_columns) 
    
    skip_columns = [
        'DefaultBrowsersIdentifier',
        'PuaMode',
        'Census_ProcessorClass',
        'Census_InternalBatteryType',
        'Census_IsFlightingInternal',
        'Census_ThresholdOptIn',
        'Census_IsWIMBootEnabled',
        'OsVer',
        'EngineVersion',
        'Census_OSVersion',
        'AvSigVersion',
        'OsBuildLab',
        'AppVersion'
    ]
    
    print('Starting...\n')
  
    for i in tqdm(range(1, len(df_columns), 1)):
        col = df_columns[i]
        if col in skip_columns or str(df.dtypes[col]) != 'category':
            continue 
        else:
            print(col)
            tmp_df = df.drop_duplicates(subset=col)
            unique_df = unique_df.append(tmp_df, ignore_index=True)
        
    unique_df.to_csv("all_values_training.csv", index=False)