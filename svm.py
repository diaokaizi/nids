import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    X_train, X_test, y_train, y_val = train_test_split(df, Y, test_size=0.3,stratify=df.Attack, random_state=42)
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

print('='*60 + '\n')
print('TESTING WITH MIN-DELTA 0.001:\n\n')
# 定义评估模型函数
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred = np.where(y_pred == 1, 0, 1)  # 1代表正常，-1代表异常; 将-1转换为1, 将1转换为0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm

for dataKey in datasets.keys():
    print(f'Fitting and testing for {dataKey}:')

    # 数据标准化
    X_train = scaler.fit_transform(x_train[dataKey])
    X_val = scaler.transform(x_val[dataKey][0])
    X_test = scaler.transform(x_test[dataKey][0])

    # 训练One-Class SVM
    clf = OneClassSVM(kernel='rbf', gamma='auto').fit(X_train)
    
    # 验证集评估
    y_val = x_val[dataKey][1]
    val_accuracy, val_precision, val_recall, val_f1, val_cm = evaluate_model(clf, X_val, y_val)
    print("Validation Set Evaluation")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print("Confusion Matrix:")
    print(val_cm)

    # 测试集评估
    y_test = x_test[dataKey][1]
    test_accuracy, test_precision, test_recall, test_f1, test_cm = evaluate_model(clf, X_test, y_test)
    print("Test Set Evaluation")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    print('='*60 + '\n')