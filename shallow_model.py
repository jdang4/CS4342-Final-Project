import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import transform_helper as Transform
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import metrics

def perform_softmax(X, y, labels, plot=False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    trees = ExtraTreesClassifier(random_state=1)
    trees.fit(X_train, y_train)

    selector = SelectFromModel(trees, prefit=True, threshold=-np.inf)

    print(X_train.shape)

    #NEW X_TRAIN FROM SELECTED FEATURES:
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    print(X_train.shape)

    model = LogisticRegression(C=0.05, tol=0.0001, penalty='l2', max_iter=1000)
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_test)
    
    yhat = yhat[:, 1]
    
    score = metrics.roc_auc_score(y_test, yhat, average=None)
    
    print(f"\nAUC Score: {score}\n")
    
    if plot:
        metrics.plot_roc_curve(model, X_test, y_test)
        plt.show()
        
    return X_train, y_train, selector


if __name__ == "__main__":
    
    data_path = f'data{os.sep}'
    useSubset = False
    train_path = ""
    if not useSubset:
        train_path = data_path + 'train.csv'
    else:
        train_path = "train_subset.csv"
    test_path = data_path + 'test.csv'
    
    # LOAD AND FREQUENCY-ENCODE
    frequency_encode = ['EngineVersion','AppVersion','AvSigVersion','Census_OSVersion']

    # LOAD AND ONE-HOT-ENCODE
    one_hot_encode = ['RtpStateBitfield',
                      'IsSxsPassiveMode',
                      'DefaultBrowsersIdentifier',
                      'AVProductStatesIdentifier',
                      'AVProductsInstalled', 
                      'AVProductsEnabled',
                      'CountryIdentifier', 
                      'CityIdentifier', 
                      'GeoNameIdentifier', 
                      'LocaleEnglishNameIdentifier',
                      'Processor', 
                      'OsBuild', 
                      'OsSuite',
                      'SmartScreen','Census_MDC2FormFactor',
                      'Census_OEMNameIdentifier', 
                      'Census_ProcessorCoreCount',
                      'Census_ProcessorModelIdentifier', 
                      'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
                      'Census_HasOpticalDiskDrive',
                      'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',
                      'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                      'Census_InternalPrimaryDisplayResolutionHorizontal',
                      'Census_InternalPrimaryDisplayResolutionVertical',
                      'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
                      'Census_InternalBatteryNumberOfCharges',
                      'Census_OSEdition', 'Census_OSInstallLanguageIdentifier',
                      'Census_GenuineStateName','Census_ActivationChannel',
                      'Census_FirmwareManufacturerIdentifier',
                      'Census_IsTouchEnabled', 'Census_IsPenCapable',
                      'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
                      'Wdft_RegionIdentifier']
    
    # LOAD ALL AS CATEGORIES
    dtypes = {}
    
    all_encodes = frequency_encode + one_hot_encode 
    
    for x in all_encodes:
        dtypes[x] = 'category'
        
    dtypes['MachineIdentifier'] = 'str'
    dtypes['HasDetections'] = 'int8'
    
    train_data = pd.read_csv(train_path, nrows=2000, usecols=dtypes.keys(), dtype=dtypes)
    
    cols = []
    cols2 = []
    dd = []
    
    for feature in frequency_encode:
        cols += Transform.encode_frequency(train_data, feature)
    
    train_data = train_data.drop(frequency_encode, axis=1)
    
    train_data = Transform.transform_categorical(train_data, one_hot_encode)
    
    X_labels = list(train_data.columns[1:])
    Xtr = train_data[X_labels].to_numpy()
    ytr = train_data["HasDetections"].to_numpy()

    Xtr = np.nan_to_num(Xtr)
    
    X_train, y_train, selection = perform_softmax(Xtr, ytr, X_labels, True)
    
    
    
    