import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
import sys, math, joblib, gc
from sklearn import preprocessing
import pickle
from sklearn import metrics
import transform_helper as Transform 
from Model import Model


def repeat_softmax(X, y, preBuilt=False, model=None):
	X_train = X 
	y_train = y
	
	if not preBuilt:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
		
		model = LogisticRegression(C=0.040980805223454236, tol=0.0037189066625450827, penalty='l2', max_iter=100,
							   solver='newton-cg', warm_start=True)
	
	
	trees = ExtraTreesClassifier(random_state=1)
	trees.fit(X_train, y_train)
	selector = SelectFromModel(trees, prefit=True, threshold=-np.inf)
		
	#NEW X_TRAIN FROM SELECTED FEATURES:
	X_train = selector.transform(X_train)

	#standardize data
	X_train = preprocessing.StandardScaler().fit_transform(X_train)
	
	model.fit(X_train, y_train)

	return model


def evaluate_model(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    
    trees = ExtraTreesClassifier(random_state=1)
    trees.fit(X_train, y_train)
    
    selector = SelectFromModel(trees, prefit=True, threshold=-np.inf)
    
    X_test = selector.transform(X_test)
    X_test = preprocessing.StandardScaler().fit_transform(X_test) 
    
    yhat = model.predict_proba(X_test)
    yhat = yhat[:, 1]
    
    score = metrics.roc_auc_score(y_test, yhat, average=None)
    
    print(f"\nAUC Score: {score}\n")
    

if __name__ == "__main__":
	
	MODEL_NUM = 1       # 1 - softmax, 2 - neural network
	TRAIN_CHUNKS = 1    # 0 - False, 1 - True
	
	model_dict = {
		1: 'softmax',
		2: 'neural_network'
	}
	
	if len(sys.argv) > 2:
		MODEL_NUM = int(sys.argv[1])
		TRAIN_CHUNKS = int(sys.argv[2])
		
	elif len(sys.argv) > 1:
		MODEL_NUM = int(sys.argv[1])
		
	data_path = f'data{os.sep}'
	useSubset = False
	train_path = ""
	
	if not useSubset:
		train_path = data_path + 'train.csv'
	else:
		train_path = "train_subset.csv"
	
	dtypes = Transform.get_dtypes()
	
	print('Reading from csv...')
	
	train_data = pd.read_csv(train_path, nrows=100000, dtype=dtypes)
	
	print('Done\n')

	ytr = train_data["HasDetections"].to_numpy()
	
	print('Transforming Dataframe...')
	
	train_data = Transform.transform_dataframe(train_data)
	
	train_data = Transform.transform_categorical(train_data)    # perform one-hot encoding on categorical columns
	
	labels = list(train_data.columns)
	
	tmp_df = pd.DataFrame(columns=labels)
	tmp_df.to_csv('final_train.csv', index=False)
	
	train_data = train_data.drop(['MachineIdentifier', 'HasDetections'], axis=1)  # drop unnecessary columns
	
	print('Done\n')
	
	print('Training model...')
	
	selection = None
	model = None
	
	if TRAIN_CHUNKS == 1 and MODEL_NUM == 1:
		Xtr = train_data.to_numpy(dtype='float64')
		Xtr = np.nan_to_num(Xtr)
		
		Xtr_evaluator = np.copy(Xtr)
		
		train_chunks = Transform.split_dataframe(train_data, chunk_size=100000)  # 100000
		ytr_chunks = Transform.split_dataframe(ytr, chunk_size=100000)
		
		del train_data 
		gc.collect()
		
		list_of_chunks = []
		
		for i,chunk in enumerate(train_chunks):
			print(f'Chunk #{i}')
			Xtr = chunk.to_numpy(dtype='float64')
			Xtr = np.nan_to_num(Xtr)
			
			model = repeat_softmax(Xtr, ytr_chunks[i], i > 0, model)
   
		del Xtr
		gc.collect()
  
		print('\nEvaluating Model...')
		
		evaluate_model(Xtr_evaluator, ytr, model)
		  
			
	else:
		Xtr = train_data.to_numpy(dtype='float64')
		Xtr = np.nan_to_num(Xtr)
	
		model = Model(Xtr, ytr, labels, MODEL_NUM)
		
		selection, model = model.train_model()
	
	
	print('Done\n')
	
	# save the model to disk
	model_name = model_dict.get(MODEL_NUM)
	
	print('Saving....\n')
	
	if MODEL_NUM == 2:
		model_json = model.to_json()
	
		with open('model.json', 'w') as json_file:
			json_file.write(model_json)

		# serialize the weights
		model.save_weights('model.h5')
		
	else:
		filename = 'model.sav'
		joblib.dump(model, filename)
	
	f = open('model_num.pckl', 'wb')
	pickle.dump(MODEL_NUM, f)
	f.close()
	
	print(f'{model_name} model saved')
	
	
	
	
	
	
	