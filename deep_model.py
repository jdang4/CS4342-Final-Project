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
import tensorflow as tf

import transform_helper as Transform 
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

from sklearn.metrics import roc_auc_score

space = [Real(1**-5, 3, "log-uniform", name="learning_rate"),
         Integer(1, 150, name="num_neurons"),
         Integer(1, 5, name="num_hidden"),
         Categorical(['relu', 'sigmoid'], name="activation")
        ]

#Creates a keras model.
"""
learning_rate: the learning rate of the model.
num_neurons: the number of neurons in each hidden layer.
num_hidden: the number of hidden layers.
activation: the activation function used by this model
"""
def build_model(input_dim, learning_rate, num_neurons, num_hidden, activation):

    model = Sequential()
    model.add(Dense(num_neurons, input_dim=input_dim, activation=activation))
    model.add(BatchNormalization())

    for x in range(0, num_hidden-1):
        model.add(Dense(num_neurons, activation=activation))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
    return model


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

    @use_named_args(space)
    def objective(**params):
        nn = build_model(cols, **params)
        nn.fit(X_train, y_train, epochs=25, batch_size=10, validation_data = (X_test, y_test))

        yhat = nn.predict_proba(X_test)
        yhat = yhat[:, 1]

        score = metrics.roc_auc_score(y_test, yhat, average=None)

        #skopt is minimizing this.
        #so we return -score so that higher scores have lower return values
        return -score

    best_hyperparameters = gp_minimize(objective, space, n_calls=50, random_state=1)
    
    print("Best score: " + str(-best_hyperparameters.fun))
    print("learning_rate: " + str(best_hyperparameters.x[0]))
    print("num_neurons: " + str(best_hyperparameters.x[1]))
    print("num_hidden: " + str(best_hyperparameters.x[2]))
    print("activation: " + str(best_hyperparameters.x[3]))

    '''
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
    model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
    
    model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data = (X_test, y_test))
    '''


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
    train_data = train_data.drop(['HasDetections'], axis=1)
    
    Xtr = train_data.to_numpy()
    
    Xtr = np.nan_to_num(Xtr)
    
    neural_network(Xtr, ytr)