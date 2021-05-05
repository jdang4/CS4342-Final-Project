import os
import numpy as np
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

from sklearn import metrics

#Number of features as a command line argument
NUM_FEATURES = 5

if len(sys.argv) > 1:
    NUM_FEATURES = int(sys.argv[1])

def perform_softmax(X, y, labels, plot=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    trees = ExtraTreesClassifier(random_state=1)
    trees.fit(X_train, y_train)

    selector = SelectFromModel(trees, prefit=True, threshold=-np.inf, max_features=NUM_FEATURES)

    print(X_train.shape)

    #NEW X_TRAIN FROM SELECTED FEATURES:
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    #standardize data
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)

    print(X_train.shape)

    model = LogisticRegression(C=5, tol=0.0001, penalty='l2', max_iter=1000, solver='sag')
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_test)
    
    print(yhat)
    yhat = yhat[:, 1]
    
    score = metrics.roc_auc_score(y_test, yhat, average=None)
    
    print(f"\nAUC Score: {score}\n")
    
    if plot:
        metrics.plot_roc_curve(model, X_test, y_test)
        plt.show()
        
    return X_train, y_train, selector
    
def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

def sort_tuple(list):
    list.sort(key= lambda x: x[1], reverse=True)
    
    return list

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
        
    feature_list = sort_tuple(feature_list)

    print("\n################# FEATURE IMPORTANCES: #################\n")
    
    for i,v in feature_list:
        print(f"Feature {i}: {v}")
        
        
    print("\n########################################################\n")
    
    
def convert(x):
    try:
        d = datetime.strptime(x.split('.')[4], '%y%m%d-%H%M').timestamp()
    
    except:
        d = np.nan
        
    return d

  
def add_timestamp(train_data):
    os_times = np.load("OSVersionTimestamps.npy", allow_pickle=True).item()
    
    datedictAS = np.load('AvSigVersionTimestamps.npy', allow_pickle=True)[()]

    for k,v in os_times.items():
        os_times[k] = v.timestamp()

    #map each version to a timestamp
    os_timestamps = train_data["Census_OSVersion"].map(os_times)

    train_data["Census_OSVersion"] = os_timestamps
    #print(type(datedictAS))
    train_data['AvSigVersion'] = train_data['AvSigVersion'].map(datedictAS)
    train_data['AvSigVersion'] = pd.to_numeric(train_data['AvSigVersion'], errors='coerce').astype(np.float64)
    
    train_data['OsBuildLab'] = train_data['OsBuildLab'].map(convert)
    
    return train_data

def transform_categorical(df):
    cols = df.select_dtypes(include=['category']).columns[1:]
    categorical_cols = list(cols)
    categorical_cols.remove('AppVersion')
    list_of_one_hots = [df]
    
    for category in categorical_cols:
        tmp_df = pd.get_dummies(df[category], prefix=category)
        list_of_one_hots.append(tmp_df)
        
    new_df = pd.concat(list_of_one_hots, axis=1)
    new_df = new_df.drop(categorical_cols, axis=1)
    
    return new_df

#orders and labels appVersions by time.
def label_appVersion_time(train_data):

    def sortAppVersion_time(a, b):
        aArr = a.split('.')
        bArr = b.split('.')

        for i in range(0, 4):

            if int(aArr[i]) != int(bArr[i]):
                return int(aArr[i]) - int(bArr[i])

        #if we reach here, they were equal
        return 0



    appVersions = train_data["AppVersion"].unique().tolist()

    appVersions.sort(key=cmp_to_key(sortAppVersion_time))

    argsort_dict = {version:idx for idx, version in enumerate(appVersions, start=1)}

    train_data["AppVersionTimeOrder"] = train_data["AppVersion"].map(argsort_dict) 
    train_data[["AppVersionTimeOrder"]] = train_data[["AppVersionTimeOrder"]].apply(pd.to_numeric)

    '''
    print("TESTING APP VERSION TIME ORDER!!!")
    print(train_data[['AppVersion', 'AppVersionTimeOrder']].head())

    print("Order was: ")
    print(appVersions)
    input()
    '''

    return train_data

def label_appVersion_count(trainData):
    vc_item1 = train_data['AppVersion'].value_counts()

    train_data['AppVersionCounts'] = train_data['AppVersion'].apply(lambda x: vc_item1[x])

    return train_data;
    
if __name__ == "__main__":
    data_path = f'data{os.sep}'
    useSubset = False
    train_path = ""
    if not useSubset:
        train_path = data_path + 'train.csv'
    else:
        train_path = "train_subset.csv"
    test_path = data_path + 'test.csv'
    
    dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int32',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float32',
        'LocaleEnglishNameIdentifier':                          'int32',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float32',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float64',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float32',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float32',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float32',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int32',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int32',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float32',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float32',
        'HasDetections':                                        'int8'
    }
    
    pd.set_option('display.max_rows', None)
    
    train_data = pd.read_csv(train_path, nrows=100000, dtype=dtypes)
     
    missing_columns = [
        'DefaultBrowsersIdentifier',
        'PuaMode',
        'Census_ProcessorClass',
        'Census_InternalBatteryType',
        'Census_IsFlightingInternal',
        'Census_ThresholdOptIn',
        'Census_IsWIMBootEnabled',
        'OsVer',
        'EngineVersion'
    ]
    
    train_data = train_data.drop(missing_columns, axis=1)
    
    train_data = add_timestamp(train_data)

    train_data = label_appVersion_time(train_data)
    train_data = label_appVersion_count(train_data)
    
    #print(train_data['AppVersionTimeOrder'].dtype)
    
    train_data = transform_categorical(train_data)

    #train_data = train_data.drop(columns="AvSigVersion")

    X_labels = list(train_data.columns[1:])
    X_labels.remove("AppVersion") 
    X_labels.remove("HasDetections")
    Xtr = train_data[X_labels].to_numpy()
    ytr = train_data["HasDetections"].to_numpy()

    Xtr = np.nan_to_num(Xtr)
    
    X_train, y_train, selection = perform_softmax(Xtr, ytr, X_labels, False)
    
    feature_importance(X_train, y_train, X_labels, selection)
    
    
    
    
