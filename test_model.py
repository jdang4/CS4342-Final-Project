import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
import sys, math, joblib, gc
from sklearn import preprocessing

from sklearn import metrics
import transform_helper as Transform 


if __name__ == "__main__":    
    data_path = f'data{os.sep}'
    useSubset = False
    test_path = ""
    
    test_path = "test_submission.csv"
    
    print('Reading from csv...')
    
    test_data = pd.read_csv(test_path, dtype='float64')
    
    df = test_data[['MachineIdentifier']].copy()
    
    test_data = test_data.drop(["MachineIdentifier"], axis=1)
    
    print('Done\n')
    
    Xte = test_data.to_numpy(dtype='float64')
    Xte = np.nan_to_num(Xte)
    
    # load the model from disk
    model = joblib.load('model.sav')
    
    # normalize testing data
    Xte = preprocessing.StandardScaler().fit_transform(Xte)
    
    yte = model.predict_proba(Xte)

    results = yte[:,1]
    
    df.insert(1, "HasDetections", results)
    df.to_csv("submission.csv", index=False)
    
    
    
    
    