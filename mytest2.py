import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow.python.keras.callbacks import EarlyStopping
import warnings
import models
import models.autoencoder
from NIDS.resultObj import Draw


datasets = {
    'NF-UNSW-NB15' : '/root/work/NIDS/data/NF-BoT-IoT.parquet',
    'NF-BoT-IoT' : '/root/work/NIDS/data/NF-BoT-IoT.parquet',
    'NF-CSE-CIC-IDS2018' : '/root/work/NIDS/data/NF-CSE-CIC-IDS2018.parquet',
    'NF-ToN-IoT' : '/root/work/NIDS/data/NF-ToN-IoT.parquet',
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
    print(df.to_csv(f'/root/work/NIDS/data/{key}.csv', index=False))
    # Y = df.Label
    # X_train, X_test, y_train, y_val = train_test_split(df, Y, test_size=0.3,stratify=df.Attack, random_state=42)
    # del df
    # del Y
    # gc.collect()
    # X_val, X_test, y_val, y_test = train_test_split(X_test, X_test.Label, test_size=0.15, stratify=X_test.Attack, random_state=42)
    # X_train = X_train[X_train.Label==0].drop(columns=features_to_remove, axis=1)
    # X_val.drop(columns=features_to_remove, axis=1, inplace=True)
    # X_test.drop(columns=features_to_remove, axis=1, inplace=True)
    # x_train[key] = X_train
    # x_val[key] = (X_val, y_val)
    # x_test[key] = (X_test, y_test)
    # del X_train
    # del X_val
    # del X_test
    # gc.collect()




def calculate_loss(X, Y):
    return np.mean(abs(X - Y), axis=1)


def give_threshold(model, method, val_data, y_val):
        
        predictions = model.predict(val_data, verbose=0)
        loss = calculate_loss(val_data, predictions)
    
    
        if method == 'ROC':
            fpr, tpr, thresholds = roc_curve(y_val, loss)
            J = tpr - fpr
            return thresholds[J.argmax()]
        elif method == 'PR':
            precision, recall, thresholds = precision_recall_curve(y_val, loss)
            distance = np.sqrt((1-precision)**2 + (1-recall)**2)
            return thresholds[distance.argmin()]
        
        return
    

def evaluate(model, test_data, y_test, roc_threshold, pr_threshold):
    predictions =  model.predict(test_data, verbose=0)   
    loss = calculate_loss(test_data, predictions)
    
    roc_result = loss > roc_threshold
    pr_result = loss > pr_threshold

    print('\tUSING ROC-CURVE & Youden:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, roc_result)}\n\t\tPrecision={precision_score(y_test, roc_result)}\n\t\tRecall={recall_score(y_test, roc_result)}\n\t\tF1={f1_score(y_test, roc_result)}\n')
    print('\tUSING PR-CURVE & Distance:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, pr_result)}\n\t\tPrecision={precision_score(y_test, pr_result)}\n\t\tRecall={recall_score(y_test, pr_result)}\n\t\tF1={f1_score(y_test, pr_result)}\n')

draw = Draw()
def fit_and_test_model(model, name, min_delta):

    print(f'Fitting and testing for {name}:')
    print('='*60 + '\n')
    
    X_train = scaler.fit_transform(x_train[name])
    X_val = scaler.transform(x_val[name][0])
    X_test = scaler.transform(x_test[name][0])
    print(name, X_train.shape, X_val.shape, X_test.shape)

    # early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)

    # #Train the model, using only the training data originating from the specific dataset.
    # history = model.fit(
    #         X_train,
    #         X_train,
    #         epochs=30,
    #         batch_size=128,
    #         shuffle=True,
    #         verbose=0,
    #         callbacks=[early_stopping]
    # )
    # draw.draw_loss(history, name)
    
    # # Determine the thresholds
    # roc_threshold = give_threshold(model, 'ROC', X_val, x_val[name][1])
    # pr_threshold = give_threshold(model, 'PR', X_val, x_val[name][1])
    # # Intra-dataset evaluation
    # print('INTRA-DATASET EVALUATION:\n')

    # evaluate(model, X_test, x_test[name][1], roc_threshold, pr_threshold)
        
    # 下面是在该数据集下训练的模型在其他数据集上的验证
#     print('INTER-DATASET EVALUATION:\n')
#     list_of_names = list(datasets.keys())
#     list_of_names.remove(name)
#     for test_name in list_of_names:
#         print(f'Evaluation on {test_name}:\n')
        
#         X_test = scaler.transform(x_test[test_name][0])
#         evaluate(model, X_test, x_test[test_name][1], roc_threshold, pr_threshold)


        
# print('TESTING WITH MIN-DELTA 0.001:\n\n')
# for dataKey in datasets.keys():
#     fit_and_test_model(models.autoencoder.create_generic_model(), dataKey, 0.001)