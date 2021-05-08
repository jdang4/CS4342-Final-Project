import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
import sys, math, joblib, gc
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
import tensorflow as tf

from sklearn import metrics
import transform_helper as Transform 

class Model:
    def __init__(self, X, y, labels, model_num=1):
        self.X = X
        self.y = y
        self.labels = labels
        self.model_num = model_num
        
    
    def train_model(self):
        if self.model_num == 2:
            return self.perform_neural_network()
        
        else:
            return self.perform_softmax()
            
    
    def preprocess_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=1)
        
        trees = ExtraTreesClassifier(random_state=1)
        trees.fit(X_train, y_train)
        
        selector = SelectFromModel(trees, prefit=True, threshold=-np.inf)
        
        #NEW X_TRAIN FROM SELECTED FEATURES:
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        #standardize data
        X_train = preprocessing.StandardScaler().fit_transform(X_train)
        X_test = preprocessing.StandardScaler().fit_transform(X_test)
        
        return X_train, X_test, y_train, y_test, selector
    
    
    def feature_importance(self, X, y, selection):
        # Finding the important features
        model = ExtraTreesClassifier(random_state=1) 
        model.fit(X, y)
    
        mask = selection.get_support()

        importances = model.feature_importances_

        labels = np.array(labels)[mask]
    
        feature_list = []
    
        # map the scores to corresponding column name
        for i in range(len(importances)):
            feature_list.append( (self.labels[i], importances[i]) )
        
        feature_list = Transform.sort_tuple(feature_list)

        print("\n################# FEATURE IMPORTANCES: #################\n")
    
        for i,v in feature_list:
            print(f"Feature {i}: {v}")
        
        print("\n########################################################\n")
    
    
    def perform_softmax(self, plot=False, display_feature=False):
        
        X_train, X_test, y_train, y_test, selector = self.preprocess_data()
        
        #{'C': 0.040980805223454236, 'tol': 0.0037189066625450827, 'max_iter': 61, 'solver': 'newton-cg', 'penalty': 'l2'}
        model = LogisticRegression(C=0.040980805223454236, tol=0.0037189066625450827, penalty='l2', max_iter=1000, solver='newton-cg')
        model.fit(X_train, y_train)
        yhat = model.predict_proba(X_test)

        yhat = yhat[:, 1]
    
        score = metrics.roc_auc_score(y_test, yhat, average=None)
    
        print(f"\nAUC Score: {score}\n")
    
        if display_feature:
            feature_importance(X_train, y_train, selector)
    
        if plot:
            metrics.plot_roc_curve(model, X_test, y_test)
            plt.show()
    
    
        return selector, model
    
    
    def perform_neural_network(self):
        X_train, X_test, y_train, y_test, selector = self.preprocess_data()
        
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
        model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
    
        model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data = (X_test, y_test))
        
        return selector, model
        
        