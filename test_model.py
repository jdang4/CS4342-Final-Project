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
from keras.models import model_from_json

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
    
    test_data = pd.read_csv(test_path, nrows=10000)
    print(test_data.shape)
    df = test_data[['MachineIdentifier']].copy()
    
    test_data = test_data.drop(["MachineIdentifier"], axis=1)

    # dtypes = Transform.get_dtypes()
    # train_data = pd.read_csv("train.csv", nrows=100000, dtype=dtypes)
    # train_data = Transform.transform_dataframe(train_data)
    #
    # train_data = Transform.transform_categorical(train_data)
    # print(train_data.shape)
    # test_data = Transform.make_matching(train_data, test_data)

    print('Done')
    
    Xte = test_data.to_numpy(dtype='float64')
    Xte = np.nan_to_num(Xte)
    
    # load the model from disk
    model = None 
    if MODEL_NUM == 2:
        json_file = open('model.json', 'r')
        json_model = json_file.read()
        json_file.close()
        model = model_from_json(json_model)
        model.load_weights('model.h5')
    else:
        model = joblib.load('model.sav')
    
    # normalize testing data
    Xte = preprocessing.StandardScaler().fit_transform(Xte)
    
    yte = model.predict_proba(Xte)
    
    results = yte[:,1] if MODEL_NUM == 1 else yte
    
    df.insert(1, "HasDetections", results)
    df.to_csv("submission.csv", index=False)
    
    
    
    
    