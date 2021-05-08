import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
import sys, math, joblib, gc
from sklearn import preprocessing

from sklearn import metrics
import transform_helper as Transform 
from Model import Model


if __name__ == "__main__":
    
    MODEL_NUM = 1 # 1 - softmax, 2 - neural network
    
    model_dict = {
        1: 'softmax',
        2: 'neural_network'
    }
    
    if len(sys.argv) > 1:
        MODEL_NUM = int(sys.argv[1])
        
    data_path = f'data{os.sep}'
    useSubset = False
    train_path = ""
    
    if not useSubset:
        train_path = data_path + 'train.csv'
    else:
        train_path = "train_subset.csv"
    
    dtypes = Transform.get_dtypes()
    
    print('Reading from csv...')
    
    train_data = pd.read_csv(train_path, nrows=100000, dtype=dtypes)
    
    print('Done\n')

    ytr = train_data["HasDetections"].to_numpy()
    
    print('Transforming Dataframe...')
    
    train_data = Transform.transform_dataframe(train_data)
    
    train_data = Transform.transform_categorical(train_data)    # perform one-hot encoding on categorical columns
    
    train_data.to_csv('final_train.csv', index=False)
    
    train_data = train_data.drop(['MachineIdentifier', 'HasDetections'], axis=1)  # drop unnecessary columns
    
    labels = list(train_data.columns)
    
    Xtr = train_data.to_numpy(dtype='float64')
    Xtr = np.nan_to_num(Xtr)
    
    model = Model(Xtr, ytr, labels, MODEL_NUM)
    
    print('Done\n')
    
    print('Training model...')
    
    selection, model = model.train_model()
    
    print('Done\n')
    
    # save the model to disk
    model_name = model_dict.get(MODEL_NUM)
    filename = 'model.sav'
    
    print('Saving....\n')
    joblib.dump(model, filename)
    
    print(f'{model_name} model saved')
    
    
    
    
    
    