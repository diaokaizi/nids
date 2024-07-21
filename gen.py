import numpy as np
import pandas as pd
import gc
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

datasets = {
    'NF-UNSW-NB15' : '/root/work/NIDS/data/NF-BoT-IoT.parquet',
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

def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.g_optimizer = Adam(learning_rate=0.0001)
        self.d_optimizer = Adam(learning_rate=0.0004)
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def compile(self):
        super(GAN, self).compile()
        self.generator.compile(optimizer=self.g_optimizer)
        self.discriminator.compile(optimizer=self.d_optimizer)

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_data = self.generator(random_latent_vectors)

        combined_data = tf.concat([generated_data, real_data], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.bce(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.bce(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}
    
    def call(self, inputs):
        return self.generator(inputs)
    
def calculate_loss(test_data, predictions):
    return np.mean(np.square(test_data - predictions), axis=1)

def give_threshold(model, method, val_data, y_val):
    predictions = model.generator.predict(tf.random.normal(shape=(len(val_data), model.latent_dim)), verbose=0)
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
    predictions = model.generator.predict(tf.random.normal(shape=(len(test_data), model.latent_dim)), verbose=0)
    loss = calculate_loss(test_data, predictions)
    
    roc_result = loss > roc_threshold
    pr_result = loss > pr_threshold

    print('\tUSING ROC-CURVE & Youden:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, roc_result)}\n\t\tPrecision={precision_score(y_test, roc_result)}\n\t\tRecall={recall_score(y_test, roc_result)}\n\t\tF1={f1_score(y_test, roc_result)}\n')
    print('\tUSING PR-CURVE & Distance:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, pr_result)}\n\t\tPrecision={precision_score(y_test, pr_result)}\n\t\tRecall={recall_score(y_test, pr_result)}\n\t\tF1={f1_score(y_test, roc_result)}\n')

def fit_and_test_gan(name):
    print(f'Fitting and testing for {name}:')
    print('='*60 + '\n')
    
    X_train = scaler.fit_transform(x_train[name])
    X_val = scaler.transform(x_val[name][0])
    X_test = scaler.transform(x_test[name][0])
    
    latent_dim = 10
    generator = build_generator(latent_dim, X_train.shape[1])
    discriminator = build_discriminator(X_train.shape[1])
    gan = GAN(generator, discriminator, latent_dim)
    gan.compile()

    epochs = 10000
    batch_size = 128
    for epoch in range(epochs):
        history = gan.fit(X_train, epochs=1, batch_size=batch_size, verbose = False)
        
        if epoch % 20 == 0:
            roc_threshold = give_threshold(gan, 'ROC', X_val, x_val[name][1])
            pr_threshold = give_threshold(gan, 'PR', X_val, x_val[name][1])

            print(epoch, 'INTRA-DATASET EVALUATION:\n')
            evaluate(gan, X_test, x_test[name][1], roc_threshold, pr_threshold)

print('TESTING WITH GAN:\n\n')
for dataKey in datasets.keys():
    fit_and_test_gan(dataKey)