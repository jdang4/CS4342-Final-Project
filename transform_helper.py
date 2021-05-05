import pandas as pd
import math 

def encode_frequency(df, col):
    count = df[col].value_counts(dropna=False)
    col_name = f"{col}_FE"
    df[col_name] = df[col].map(count) / count.max()
    
    return [col_name]


def transform_categorical(df, cols):
    list_of_one_hots = [df]
    
    for category in cols:
        tmp_df = pd.get_dummies(df[category], prefix=category)
        list_of_one_hots.append(tmp_df)
        df.drop(category, axis=1)
        
    new_df = pd.concat(list_of_one_hots, axis=1)
    #new_df = new_df.drop(cols, axis=1)
    
    return new_df
    
    

