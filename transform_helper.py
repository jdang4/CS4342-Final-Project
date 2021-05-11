import numpy as np
import pandas as pd
from datetime import datetime
from functools import cmp_to_key
from sklearn.preprocessing import OneHotEncoder
import os
import math
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

def get_dtypes():
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
    
    return dtypes


def get_categorical_cols():
    categories = [
        'ProductName', 'Platform', 
        'Processor', 'SmartScreen', 'Census_MDC2FormFactor', 'Census_DeviceFamily', 'Census_ChassisTypeName',
        'Census_PowerPlatformRoleName', 'Census_OSBranch', 'Census_OSEdition', 'Census_OSSkuName',
        'Census_GenuineStateName', 'Census_FlightRing', 'OsPlatformSubRelease', 'SkuEdition', 
        'Census_PrimaryDiskTypeName', 'Census_OSArchitecture', 'Census_OSInstallTypeName', 
        'Census_OSWUAutoUpdateOptionsName', 'Census_ActivationChannel', 
    ]
    
    return categories

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day


def sort_tuple(list):
    list.sort(key= lambda x: x[1], reverse=True)
    
    return list


def convert(x):
    try:
        d = datetime.strptime(x.split('.')[4], '%y%m%d-%H%M').timestamp()
    
    except:
        d = np.nan
        
    return d

  
def add_timestamp(df):
    os_times = np.load(f'mapped_data{os.sep}OSVersionTimestamps.npy', allow_pickle=True).item()
    
    datedictAS = np.load(f'mapped_data{os.sep}AvSigVersionTimestamps.npy', allow_pickle=True)[()]

    for k,v in os_times.items():
        os_times[k] = v.timestamp()

    #map each version to a timestamp
    os_timestamps = df["Census_OSVersion"].map(os_times)

    df["Census_OSVersion"] = os_timestamps
    #print(type(datedictAS))
    df['AvSigVersion'] = df['AvSigVersion'].map(datedictAS)
    df['AvSigVersion'] = pd.to_numeric(df['AvSigVersion'], errors='coerce').astype(np.float64)
    
    df['OsBuildLab'] = df['OsBuildLab'].map(convert)
    
    return df

def transform_categorical2(df):
    categorical_cols = get_categorical_cols()
    #numeric = df.select_dtypes(include=np.number).columns.tolist()
    #numeric_df = df[[numeric]]
    category_df = df[categorical_cols].copy()
    category_df = category_df.astype('string')
    category_df = category_df.fillna('None')
    category_df = category_df.astype('category')
    
    myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    myEncoder.fit(category_df[categorical_cols])
    
    new_df = pd.concat([df.drop(categorical_cols, 1), pd.DataFrame(myEncoder.transform(category_df[categorical_cols]))], axis=1).reindex()
    
    # tmp_df = pd.get_dummies(df[categorical_cols])
    # list_of_one_hots = [df, tmp_df]
    # new_df = pd.concat(list_of_one_hots, axis=1)
    # new_df = new_df.drop(categorical_cols, axis=1)
    
    return new_df


def transform_categorical(df):
    list_of_one_hots = [df]
    cols = get_categorical_cols()
    for category in cols:
        tmp_df = pd.get_dummies(df[category], prefix=category)
        list_of_one_hots.append(tmp_df)
        #df.drop(category, axis=1)
        
    new_df = pd.concat(list_of_one_hots, axis=1)
    new_df = new_df.drop(cols, axis=1)
    
    return new_df

#orders and labels appVersions by time.
def label_appVersion_time(df):

    def sortAppVersion_time(a, b):
        aArr = a.split('.')
        bArr = b.split('.')

        for i in range(0, 4):

            if int(aArr[i]) != int(bArr[i]):
                return int(aArr[i]) - int(bArr[i])

        #if we reach here, they were equal
        return 0



    appVersions = df["AppVersion"].unique().tolist()

    appVersions.sort(key=cmp_to_key(sortAppVersion_time))

    argsort_dict = {version:idx for idx, version in enumerate(appVersions, start=1)}

    df["AppVersionTimeOrder"] = df["AppVersion"].map(argsort_dict) 
    df[["AppVersionTimeOrder"]] = df[["AppVersionTimeOrder"]].apply(pd.to_numeric)


    return df

def label_appVersion_count(df):
    vc_item1 = df['AppVersion'].value_counts()

    df['AppVersionCounts'] = df['AppVersion'].apply(lambda x: vc_item1[x])
    
    df = df.drop(["AppVersion"], axis=1)
    
    return df

def make_matching(train, test):
    l1 = list(train.columns)
    l2 = list(test.columns)
    diff = list(set(l2) - set(l1))
    
    test = test.drop(diff, axis=1)
    
    
    return test

def get_missing_columns(train, test):
    l1 = list(train.columns)
    l2 = list(test.columns)
    
    diff = list(set(l1) - set(l2))
    
    return diff


def add_missing_columns(train, test):
    missing_cols = get_missing_columns(train, test)
    
    for col in missing_cols:
        if col != 'HasDetections':
            mean_val = train[[col]].mean() 
            test[col] = mean_val
            
    return test


def remove_cols(df):
    missing_columns = [
        'Census_ChassisTypeName_35',
        'Census_ChassisTypeName_IoTGateway',
        'Census_OSBranch_rs5_release_edge',
        'Census_PowerPlatformRoleName_PerformanceServer',
        'Census_OSSkuName_SB_SOLUTION_SERVER',
        'Census_OSBranch_rs5_release_sigma',
        'Census_OSEdition_ProfessionalEducationN',
        'Unnamed: 0',
        'Census_OSEdition_ServerDatacenterEval',
        'SmartScreen_&#x01;',
        'SmartScreen_on',
        'Census_MDC2FormFactor_ServerOther',
        'Census_OSSkuName_ENTERPRISE_N',
        'Census_ChassisTypeName_82',
        'Census_ChassisTypeName_0',
        'Census_ChassisTypeName_Blade',
        'Census_OSEdition_ServerSolution',
        'SmartScreen_&#x02;',
        'Census_ChassisTypeName_36',
        'Census_ChassisTypeName_BusExpansionChassis',
        'Census_OSEdition_EnterpriseN',
        'Census_OSSkuName_DATACENTER_EVALUATION_SERVER'
    ]
    df = df.drop(missing_columns, axis=1)

    # df = df.drop(['MachineIdentifier'], axis=1)

    return df

def transform_dataframe(df):
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
    try:
        df = df.drop(missing_columns, axis=1)
    except:
        return df
    df = add_timestamp(df)
    df = label_appVersion_time(df) 
    df = label_appVersion_count(df)
    
    #df = df.drop(['MachineIdentifier'], axis=1)
    
    return df

def split_dataframe(df, chunk_size = 500000):
    chunks = list() 
    num_chunks = math.ceil(len(df) / chunk_size) 
    
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size: (i+1) * chunk_size])
        
    return chunks