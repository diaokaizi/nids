import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import tensorflow as tf
from keras.callbacks import Callback
import warnings
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
'''
Custom early stopping callback. 

The callback monitors the AUROC, calculated on a provided validation set, during the training of the autoencoder.

'''

'''
Load and preprocess the data sources.

'''

datasets = {
     'NF-UNSW-NB15' : '/root/work/NIDS/data/NF-UNSW-NB15.parquet',
}

features_to_remove = ['Attack', 'Label']

scaler = QuantileTransformer(output_distribution='normal')

x_train = {}
x_val = {}
x_test = {}


for key, value in datasets.items():
    print(f'Processing {key}')
    print('='*20 + '\n')
    df = pd.read_parquet(value)
    Y = df.Label
    X_train, X_test, y_train, y_val = train_test_split(df, Y, test_size=0.05,stratify=df.Attack, random_state=42)
    del df
    del Y
    gc.collect()
    X_val, X_test, y_val, y_test = train_test_split(X_test, X_test.Label, test_size=0.15, stratify=X_test.Attack, random_state=42)
    X_train = X_train[X_train.Label==0].drop(columns=features_to_remove, axis=1)
    X_val.drop(columns=features_to_remove, axis=1, inplace=True)
    X_test.drop(columns=features_to_remove, axis=1, inplace=True)
    x_train[key] = X_train
    x_val[key] = (X_val, y_val)
    x_test[key] = (X_test, y_test)
    del X_train
    del X_val
    del X_test
    gc.collect()

print('\nFinished processing data sources.\n')


'''
Function for creating the autoencoder.

'''

def create_generic_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(10,)))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


'''
Function that returns thet MAE.

'''

def calculate_loss(X, Y):
    return np.mean(abs(X - Y), axis=1)

'''
Function that calculates the threshold for the given validation data.

Two options:
- using ROC-curve => method = 'ROC'
- using PR-curve  => method = 'PR'

'''

def give_threshold(model, method, val_data, y_val):
        
        predictions = model.predict(val_data, verbose=0)
        loss = calculate_loss(val_data, predictions)
    
    
        if method == 'ROC':
            #Evaluate model using roc-curve and the Youden J statistic
            fpr, tpr, thresholds = roc_curve(y_val, loss)
            J = tpr - fpr
            return thresholds[J.argmax()]
        elif method == 'PR':
            #Evaluate model using precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_val, loss)
            #The Euclidean distance between each point on the curve and the upper right point (1,1) (=the ideal point)
            distance = np.sqrt((1-precision)**2 + (1-recall)**2)
            return thresholds[distance.argmin()]
        
        return
    

'''
Function to evaluate a model base the provided test set and threshold.

'''
def evaluate(model, test_data, y_test, roc_threshold, pr_threshold):
    predictions =  model.predict(test_data, verbose=0)   
    loss = calculate_loss(test_data, predictions)
    
    roc_result = loss > roc_threshold
    pr_result = loss > pr_threshold

    print('\tUSING ROC-CURVE & Youden:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, roc_result)}\n\t\tPrecision={precision_score(y_test, roc_result)}\n\t\tRecall={recall_score(y_test, roc_result)}\n\t\tF1={f1_score(y_test, roc_result)}\n')
    print('\tUSING PR-CURVE & Distance:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, pr_result)}\n\t\tPrecision={precision_score(y_test, pr_result)}\n\t\tRecall={recall_score(y_test, pr_result)}\n\t\tF1={f1_score(y_test, roc_result)}\n')


'''
Function for fitting the model on the training data, and perform intra- and inter-dataset evaluation.

'''
def fit_and_test_model(model, name, min_delta):

    print(f'Fitting and testing for {name}:')
    print('='*60 + '\n')
    
    X_train = scaler.fit_transform(x_train[name])
    X_val = scaler.transform(x_val[name][0])
    X_test = scaler.transform(x_test[name][0])
    #Train the model, using only the training data originating from the specific dataset.
    epochs = 200
    batch_size = 128
    early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)

    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.05, callbacks=[early_stopping])
    
    # Determine the thresholds
    roc_threshold = give_threshold(model, 'ROC', X_val, x_val[name][1])
    pr_threshold = give_threshold(model, 'PR', X_val, x_val[name][1])
    # Intra-dataset evaluation
    print('INTRA-DATASET EVALUATION:\n')

    evaluate(model, X_test, x_test[name][1], roc_threshold, pr_threshold)
        

        
print('TESTING WITH MIN-DELTA 0.001:\n\n')
for dataKey in datasets.keys():
    fit_and_test_model(create_generic_model(), dataKey, 0.001)