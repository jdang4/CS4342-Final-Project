import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    data_path = f'data{os.sep}'
    train_path = data_path + 'train.csv'
    test_path = data_path + 'test.csv'
    
    train_data = pd.read_csv(train_path, nrows=1000)