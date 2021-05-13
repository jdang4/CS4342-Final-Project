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

if __name__ == "__main__":
    data_path = f'data{os.sep}'
    train_path = "final_train.csv"
    test_path = data_path + "test.csv"
    
    dtypes = Transform.get_dtypes()
    
    print('Reading from csv...')
    
    train_data = pd.read_csv(train_path, dtype=dtypes)
    #test_data = pd.read_csv(test_path, nrows=7853253, dtype=dtypes)
    test_data = pd.read_csv(test_path, nrows=100000, dtype=dtypes)
    
    test_data = Transform.transform_dataframe(test_data)
    
    test_chunks = Transform.split_dataframe(test_data, chunk_size=1000)
    
    train_cols = list(train_data.columns)
    
    del test_data
    gc.collect()
    
    # print(len(test_chunks))
    num_of_chunks = len(test_chunks)
    print(f'Total Number of Chunks: {num_of_chunks}\n')
    
    list_of_chunks = []
    print('One-Hot Encoding All Chunks...')
    for chunk in test_chunks:
        chunk = Transform.transform_categorical(chunk)
    
        chunk = Transform.make_matching(train_data, chunk)
        chunk = Transform.add_missing_columns(train_data, chunk)
    
        list_of_chunks.append(chunk)
    
    
    del test_chunks
    gc.collect()
    
    print('Done\n')
    
    f = open('test_submission2.csv', "w")
    f.write('')
    f.close() 
    
    print('Writing to file....')
    with open("test_submission2.csv", 'a') as f:
    
        for i,chunk in enumerate(list_of_chunks):
            print(f'Chunk #{i}')
            if i == 0:
                chunk.to_csv(f, header=True, index=False)
            else:
                chunk.to_csv(f, header=False, index=False)

    
    print('Done\n')
