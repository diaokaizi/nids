import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
import models.ave
import models.cnn_autoencoder
from resultObj import Result

'''
Custom early stopping callback. 

The callback monitors the AUROC, calculated on a provided validation set, during the training of the autoencoder.

'''


class AUROCEarlyStoppingPruneCallback(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        x_val:
            Input vector of validation data.
        y_val:
            Labels for input vector of validation data.
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 x_val, 
                 y_val, 
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(AUROCEarlyStoppingPruneCallback, self).__init__()

        self.x_val = x_val
        self.y_val = y_val
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1


    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_AUROC()
        if current is None:
            return
        
        if self.verbose > 0:
            print(f'Epoch #{epoch}\tValidation AUROC: {current}\tBest AUROC: {self.best}')
        

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
    
    # Evaluation on custom metric
    def get_AUROC(self):
        x_pred = self.model.predict(self.x_val, verbose=0)
        sse = np.mean(abs(self.x_val - x_pred), axis=1)
        fpr, tpr, thresholds = roc_curve(self.y_val, sse)
        return auc(fpr, tpr)



'''
Load and preprocess the data sources.

'''

datasets = {
    'NF-UNSW-NB15' : '/root/work/NIDS/data/NF-UNSW-NB15.parquet',
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

resultObj = Result()





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

    roc_metrics = {
        'Accuracy': accuracy_score(y_test, roc_result),
        'Precision': precision_score(y_test, roc_result),
        'Recall': recall_score(y_test, roc_result),
        'F1': f1_score(y_test, roc_result)
    }

    pr_metrics = {
        'Accuracy': accuracy_score(y_test, pr_result),
        'Precision': precision_score(y_test, pr_result),
        'Recall': recall_score(y_test, pr_result),
        'F1': f1_score(y_test, pr_result)
    }

    print('\tUSING ROC-CURVE & Youden:\n')
    print(roc_metrics)
    print('\tUSING PR-CURVE & Distance:\n')
    print(pr_metrics)
    return roc_metrics, pr_metrics

'''
Function for fitting the model on the training data, and perform intra- and inter-dataset evaluation.

'''
def fit_and_test_model(model, name, min_delta):

    print(f'Fitting and testing for {name}:')
    print('='*60 + '\n')
    
    X_train = scaler.fit_transform(x_train[name])
    X_val = scaler.transform(x_val[name][0])
    X_test = scaler.transform(x_test[name][0])
    print(name, X_train.shape, X_val.shape, X_test.shape)

    early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)

    history = model.fit(X_train, X_train, epochs=100, verbose=0, batch_size=128,
                        validation_split=0.1, callbacks=[early_stopping])
    resultObj.draw_loss(history, name)
    
    # Determine the thresholds
    roc_threshold = give_threshold(model, 'ROC', X_val, x_val[name][1])
    pr_threshold = give_threshold(model, 'PR', X_val, x_val[name][1])
    # Intra-dataset evaluation
    print('INTRA-DATASET EVALUATION:\n')
    roc_metrics, pr_metrics = evaluate(model, X_test, x_test[name][1], roc_threshold, pr_threshold)
    resultObj.save_result(roc_metrics, pr_metrics, name)
        
print('TESTING WITH MIN-DELTA 0.001:\n\n')
for dataKey in datasets.keys():
    fit_and_test_model(models.autoencoder.create_generic_model(), dataKey, 0.001)