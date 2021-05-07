import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam

import transform_helper as Transform 

def neural_network(X, y):
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
    
    cols = X_train.shape[1]
    model = Sequential()
    model.add(Dense(100,input_dim=cols))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, epochs=150, batch_size=10)

if __name__ == "__main__":
    data_path = f'data{os.sep}'
    useSubset = False
    train_path = ""
    if not useSubset:
        train_path = data_path + 'train.csv'
    else:
        train_path = "train_subset.csv"
    test_path = data_path + 'test.csv'
    unique_train_path = 'all_values_training.csv'
    
    dtypes = Transform.get_dtypes()
    
    train_data = pd.read_csv(train_path, nrows=100000, dtype=dtypes)
    
    
    ytr = train_data["HasDetections"].to_numpy()
    
    train_data = Transform.transform_dataframe(train_data)
    #test_data = Transform.transform_dataframe(test_data)
    print('HERE')
    
    train_data = Transform.transform_categorical(train_data)
    
    train_data = train_data.drop(['MachineIdentifier'], axis=1)
    
    Xtr = train_data.to_numpy()
    
    Xtr = np.nan_to_num(Xtr)
    
    neural_network(Xtr, ytr)