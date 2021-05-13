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


def get_total_rows(filepath):
    count = 0
    with open(filepath) as fp:
        for _ in fp:
            count += 1

    return count

def update_skip(cur_list, start, end):
    new_skip = [i for i in range(start, end+1, 1)]

    cur_list += new_skip 

    return cur_list

if __name__ == "__main__":    
    f = open('model_num.pckl', 'rb')
    MODEL_NUM = pickle.load(f)
    f.close()
    
    data_path = f'data{os.sep}'
    useSubset = False
    test_path = ""
    
    test_path = "test_submission2.csv"

    # load the model from disk
    model = None 
    if MODEL_NUM == 2:
        json_file = open(f'saved_models{os.sep}model.json', 'r')
        json_model = json_file.read()
        json_file.close()
        model = model_from_json(json_model)
        model.load_weights(f'saved_models{os.sep}model.h5')
    
    else:
        model = joblib.load(f'saved_models{os.sep}model.sav')

    num_of_rows = get_total_rows(test_path) - 1

    chunk_size = 10000

    num_chunks = num_of_rows // chunk_size + 1

    list_of_chunks = []

    for i in range(0, num_chunks):
        start = i * chunk_size + 1 if i != 0 else 1
        end = (i + 1) * chunk_size 
        row_range = (start, end)

        list_of_chunks.append(row_range)
    
    rows_to_skip = []

    print(list_of_chunks)
    input('stop')

    print('Starting...\n')
    for i in range(len(list_of_chunks)):
        print(f'Chunk #{i}')
        start, end = list_of_chunks[i]
        row_range = end - start + 1

        if start >= num_of_rows:
            break

        test_data = pd.read_csv(test_path, skiprows=rows_to_skip, nrows=row_range)
        rows_to_skip = update_skip(rows_to_skip, start, end)

        df = test_data[['MachineIdentifier']].copy()
        test_data = test_data.drop(["MachineIdentifier"], axis=1)

        Xte = test_data.to_numpy(dtype='float64')
        Xte = np.nan_to_num(Xte)

        del test_data
        gc.collect()
    
        # normalize testing data
        Xte = preprocessing.StandardScaler().fit_transform(Xte)

        yte = model.predict_proba(Xte)
    
        results = yte[:,1] if MODEL_NUM == 1 else yte

        del Xte
        gc.collect()

        df.insert(1, "HasDetections", results)
        
        if i == 0:
            df.to_csv("submission_chunks.csv", index=False)

        else:
            df.to_csv("submission_chunks.csv", mode='a', index=False, header=False)
    
    
    
    
    