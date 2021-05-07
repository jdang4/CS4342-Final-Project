import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
from datetime import datetime
from functools import cmp_to_key
import sys
import math
from sklearn import preprocessing
import joblib, gc

from sklearn import metrics
import transform_helper as Transform 

#Number of features as a command line argument
NUM_FEATURES = 5

if len(sys.argv) > 1:
    NUM_FEATURES = int(sys.argv[1])

def perform_softmax(X, y, labels, train=True, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    trees = ExtraTreesClassifier(random_state=1)
    trees.fit(X_train, y_train)

    selector = SelectFromModel(trees, prefit=True, threshold=-np.inf)

    print(X_train.shape)

    #NEW X_TRAIN FROM SELECTED FEATURES:
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    #standardize data
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)

    print(X_train.shape)
#{'C': 0.040980805223454236, 'tol': 0.0037189066625450827, 'max_iter': 61, 'solver': 'newton-cg', 'penalty': 'l2'}
    model = LogisticRegression(C=0.040980805223454236, tol=0.0037189066625450827, penalty='l2', max_iter=1000, solver='newton-cg')
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_test)
    
    print(yhat)
    yhat = yhat[:, 1]
    
    score = metrics.roc_auc_score(y_test, yhat, average=None)
    
    print(f"\nAUC Score: {score}\n")
    
    if plot:
        metrics.plot_roc_curve(model, X_test, y_test)
        plt.show()
        
    return selector, model


def feature_importance(X, y, labels, selection):
    
    model = ExtraTreesClassifier(random_state=1) 
    model.fit(X, y)
    
    mask = selection.get_support()

    importances = model.feature_importances_

    labels = np.array(labels)[mask]
    
    feature_list = []
    # scores for the features
    for i in range(len(importances)):
        feature_list.append( (labels[i], importances[i]) )
        
    feature_list = Transform.sort_tuple(feature_list)

    print("\n################# FEATURE IMPORTANCES: #################\n")
    
    for i,v in feature_list:
        print(f"Feature {i}: {v}")
        
        
    print("\n########################################################\n")
    
    
if __name__ == "__main__":
    data_path = f'data{os.sep}'
    useSubset = False
    train_path = ""
    if not useSubset:
        train_path = data_path + 'train.csv'
    else:
        train_path = "train_subset.csv"
    test_path = 'test_subset.csv'
    # unique_train_path = 'all_values_training.csv'
    
    dtypes = Transform.get_dtypes()
    
    #unique_train = pd.read_csv(unique_train_path, dtype=dtypes)
    #train_data = pd.read_csv(train_path, nrows=8921483, dtype=dtypes, verbose=True)
    train_data = pd.read_csv(train_path, nrows=10000, dtype=dtypes, verbose=True)
    #train_data = train_data.append(unique_train)
    #train_data = train_data.drop_duplicates()

    test_data = pd.read_csv(test_path, nrows=10000, dtype=dtypes, verbose=True)
    #test_data = test_data.drop(test_data.columns[0], axis=1)
    #test_data.to_csv("test_subset.csv", index=False)
    ytr = train_data["HasDetections"].to_numpy()
    
    train_data = Transform.transform_dataframe(train_data)
    test_data = Transform.transform_dataframe(test_data)
    print('HERE')
    
    train_data = Transform.transform_categorical(train_data)
    
    # test_chunks = Transform.split_dataframe(test_data, chunk_size=100000)
    #
    # train_cols = list(train_data.columns)
    #
    # fake_train = train_data.head(3)
    #
    # del test_data, train_data
    # gc.collect()
    #
    # print(len(test_chunks))
    #
    # #test_data = pd.DataFrame(columns=train_cols)
    # list_of_chunks = []
    # print('Starting...\n')
    # for chunk in test_chunks:
    #     chunk = Transform.transform_categorical(chunk)
    #
    #     chunk = Transform.make_matching(fake_train, chunk)
    #     chunk = Transform.add_missing_columns(fake_train, chunk)
    #
    #     print(fake_train.shape)
    #     print(chunk.shape)
    #
    #     l1 = list(fake_train.columns)
    #     l2 = list(chunk.columns)
    #
    #     diff = list(set(l1) - set(l2))
    #
    #     print(diff)
    #     print(len(diff))
    #
    #     print('Appending...')
    #     list_of_chunks.append(chunk)
    #
    #     print('\nNext\n')
    #
    # del test_chunks
    # gc.collect()
    
    #print('Starting first concat....')
    # with open("test_submission.csv", 'a') as f:
    #
    #     for i,chunk in enumerate(list_of_chunks):
    #         if i == 0:
    #             chunk.to_csv(f, header=True)
    #         else:
    #             chunk.to_csv(f, header=False)
    #         print(i)

    # df = pd.concat(list_of_chunks, axis=1)
    #
    # print('Finished concat')
    #
    # df.to_csv("test_submission.csv", index=False)
    
    # sys.exit(1)
        
        
        

#    train_data = Transform.transform_categorical(train_data)
    #test_data = Transform.transform_categorical(test_data)
    
    test_data = Transform.make_matching(train_data, test_data)

    
    #test_data.fillna(0)
    train_data = train_data.drop(['HasDetections'], axis=1)
    train_data = train_data.drop(['MachineIdentifier'], axis=1)
    #test_data = Transform.remove_cols(test_data)
    test_data.replace(np.nan, 0)
    print(train_data.shape)
    print(test_data.shape)

    l1 = list(train_data.columns)
    l2 = list(test_data.columns)

    diff = list(set(l2) - set(l1))
    diff2 = list(set(l1) - set(l2))
    print(diff)
    print(len(diff))
    print(diff2)
    print(len(diff2))
    ident = test_data["MachineIdentifier"].to_numpy()
    
    df = test_data[['MachineIdentifier']].copy()
    
    test_data = test_data.drop(["MachineIdentifier"], axis=1)
    
    Xtr = train_data.to_numpy()
    Xte = test_data.to_numpy(dtype='float64')
    
    print(np.isnan(Xte))
    
    Xtr = np.nan_to_num(Xtr)
    Xte = np.nan_to_num(Xte)
    
    
    
    labels = list(train_data.columns)
    

    selection, model = perform_softmax(Xtr, ytr, labels, False)
    
    #feature_importance(X_train, y_train, X_labels, selection)
    print(test_data.head())

#    Xte = selection.transform(Xte)
    #(np.all(np.isfinite(Xte)))
    #print(Xte)
    #print(Xte.dtype)
    #print(np.isnan(Xte).any())
    Xte = preprocessing.StandardScaler().fit_transform(Xte)

    yte = model.predict_proba(Xte)


    results = yte[:,1]

    # Creating the template for submission to Kaggle
    #columns_to_drop = [i for i in range(1, len(test_data.columns), 1)]
    #df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
    
    df.insert(1, "HasDetections", results)
    df.to_csv("submission.csv", index=False)
