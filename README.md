# CS4342-Final-Project

## Set Up
Add the Kaggle data to the ```data/``` directory in the project's directory. Currently, the zip files are stored in the ```data/``` directory, so just need to unzip them.

## How To Run
Run the code in this order:
1. train_model.py MODEL_NUM CHUNKS_FLAG
    * MODEL_NUM: allows the user to define which model they want to train. 1 - Softmax, 2 - Neural Network, 3 - Random Forest
    * CHUNKS_FLAG: allows the user to decide if they want to train the model in chunks. 0 - False (Not in Chunks), 1 - True (Yes, to Chunks)
2. generate_test_submission.py 
3. test_model.py
    - also, you can run test_model_chunks.py if you want to test the model in chunks 

## Code Base Description
In an effort to conserve resources on the machine, we split our code into seperate scripts. There is a script to train the model (```train_model.py```), and once it trains the model, it will then save the model for it to be used in other scripts. There is a script ```generate_test_submission.py``` to handle the transformation of the testing set and then writing the transformation to a csv file. Then once the csv file has been generated, the user can then run ```test_model.py``` to generate the Kaggle test submission csv file. The user can also run ```test_model_chunks.py``` to do the same functionality as the ```test_model.py``` file, but in chunks. 

## Code Base Structure
* chunk_model_tf/
    * location where the keras model is saved when training it in chunks
* data/
    * where the code reads the Kaggle dataset
    * should work on all machines with the use of ```os.sep```
* mapped_data/
    * location that stores the npy files that are used to help map the version numbers to their time of release
* saved_models/
    * location that stores the trained models
    * there also consists of other saved models that are in folders. These models are the old models that we saved that are not optimized but used to compare final models to these older unoptimized models
* find_optimize_hyperparameters.py
    * script to retrieve the optimized hyperparameters for the Neural Network model