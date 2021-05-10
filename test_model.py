import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
import sys, math, joblib, gc


from sklearn import preprocessing

from sklearn import metrics
from tensorflow import keras

import transform_helper as Transform 

import pickle


if __name__ == "__main__":    
    f = open('model_num.pckl', 'rb')
    MODEL_NUM = pickle.load(f)
    f.close()
    
    data_path = ""
    useSubset = False
    test_path = ""
    
    test_path = "test_submission2.csv"
    
    print('Reading from csv...')
    
    test_data = pd.read_csv(test_path)
    
    df = test_data[['MachineIdentifier']].copy()
    print(df)
    test_data = test_data.drop(["MachineIdentifier"], axis=1)

    dtypes = Transform.get_dtypes()
    train_data = pd.read_csv("train.csv", nrows=100000, dtype=dtypes)
    train_data = Transform.transform_dataframe(train_data)

    train_data = Transform.transform_categorical(train_data)
    print(train_data.shape)
    test_data = Transform.make_matching(train_data, test_data)

    print('Done')
    
    Xte = test_data.to_numpy(dtype='float64')
    Xte = np.nan_to_num(Xte)

    # load the model from disk
    model = None 
    if MODEL_NUM == 2:
        model = keras.models.load_model('model')
    
    else:
        model = joblib.load('model.sav')

    
    model = keras.models.load_model('model')
    
    # normalize testing data
    Xte = preprocessing.StandardScaler().fit_transform(Xte)

    yte = model.predict_proba(Xte)
    print('ml')
    print(yte)
    results = yte if MODEL_NUM == 1 else yte
    
    df.insert(1, "HasDetections", results)
    print(df)
    df.to_csv("submission.csv", index=False)
    
    
    
    
    