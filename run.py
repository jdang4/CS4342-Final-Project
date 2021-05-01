import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot

from sklearn import metrics

def perform_softmax(X, y, labels, plot=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    model = LogisticRegression(C=0.05, tol=0.001, penalty='l2')
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_test)
    
    yhat = yhat[:, 1]
    
    score = metrics.roc_auc_score(y_test, yhat, average=None)
    
    print(f"\nAUC Score: {score}\n")
    
    if plot:
        metrics.plot_roc_curve(model, X_test, y_test)
        plt.show()
        
    return X_train, y_train
    

def feature_importance(X, y, labels):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    
    # learn relationship from training data
    fs.fit(X, y)
    
    X_train_fs = fs.transform(X)
    
    print("\n################# FEATURE IMPORTANCES: #################\n")
    # scores for the features
    for i in range(len(fs.scores_)):
        print(f'Feature {labels[i]}: {fs.scores_[i]}')
        
    print("\n########################################################\n")
    

    
def transform_categorical(array):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(array)
    
    return enc.transform(array).toarray()
    
if __name__ == "__main__":
    data_path = f'data{os.sep}'
    train_path = data_path + 'train.csv'
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
    
    train_data = pd.read_csv(train_path, nrows=1000)

    os_times = np.load("OSVersionTimestamps.npy", allow_pickle=True).item()

    #turn Census_OSVersion into a unix timestamp.
    for k,v in os_times.items():
        os_times[k] = v.timestamp()

    #map each version to a timestamp
    os_timestamps = train_data["Census_OSVersion"].map(os_times)

    train_data["OSVersionTimestamps"] = os_timestamps

    print(train_data[["Census_OSVersion", "OSVersionTimestamps"]])
    
    X_labels = ['Firewall', 'HasTpm', 'OSVersionTimestamps']
    Xtr = train_data[X_labels].to_numpy()
    ytr = train_data["HasDetections"].to_numpy()
    
    Xtr = np.nan_to_num(Xtr)
    ytr = np.nan_to_num(ytr)
    
    X_train, y_train = perform_softmax(Xtr, ytr, X_labels)
    
    feature_importance(X_train, y_train, X_labels)
    
    
    
    
