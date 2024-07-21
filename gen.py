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

def build_generator(input_dim):
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(input_dim, activation='linear'))
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 自定义GAN类
class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, input_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.input_dim = input_dim
        self.g_optimizer = Adam(learning_rate=0.0001)
        self.d_optimizer = Adam(learning_rate=0.0004)
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def compile(self):
        super(GAN, self).compile()
        self.generator.compile(optimizer=self.g_optimizer)
        self.discriminator.compile(optimizer=self.d_optimizer)

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        # 训练判别器
        generated_data = self.generator(real_data)

        with tf.GradientTape() as tape:
            predictions_on_real = self.discriminator(real_data)
            predictions_on_fake = self.discriminator(generated_data)
            d_loss_real = self.bce(tf.ones_like(predictions_on_real), predictions_on_real)
            d_loss_fake = self.bce(tf.zeros_like(predictions_on_fake), predictions_on_fake)
            d_loss = (d_loss_real + d_loss_fake) / 2

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # 训练生成器
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions_on_fake = self.discriminator(self.generator(real_data))
            g_loss = self.bce(misleading_labels, predictions_on_fake)

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def call(self, inputs):
        return self.generator(inputs)

# 计算损失
def calculate_loss(test_data, predictions):
    return np.mean(np.square(test_data - predictions), axis=1)

# 给定阈值
def give_threshold(model, method, val_data, y_val):
    predictions = model.generator.predict(val_data, verbose=0)
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

# 评估模型
def evaluate(model, test_data, y_test, roc_threshold, pr_threshold):
    predictions = model.generator.predict(test_data, verbose=0)
    loss = calculate_loss(test_data, predictions)

    roc_result = loss > roc_threshold
    pr_result = loss > pr_threshold

    print('\tUSING ROC-CURVE & Youden:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, roc_result)}\n\t\tPrecision={precision_score(y_test, roc_result)}\n\t\tRecall={recall_score(y_test, roc_result)}\n\t\tF1={f1_score(y_test, roc_result)}\n')
    print('\tUSING PR-CURVE & Distance:\n')
    print(f'\t\tAccuracy={accuracy_score(y_test, pr_result)}\n\t\tPrecision={precision_score(y_test, pr_result)}\n\t\tRecall={recall_score(y_test, pr_result)}\n\t\tF1={f1_score(y_test, pr_result)}\n')

# 训练和测试GAN
def fit_and_test_gan(name):
    print(f'Fitting and testing for {name}:')
    print('='*60 + '\n')

    X_train = scaler.fit_transform(x_train[name])
    X_val = scaler.transform(x_val[name][0])
    X_test = scaler.transform(x_test[name][0])

    input_dim = X_train.shape[1]
    generator = build_generator(input_dim)
    discriminator = build_discriminator(input_dim)
    gan = GAN(generator, discriminator, input_dim)
    gan.compile()

    epochs = 5000
    batch_size = 128
    for epoch in range(epochs):
        gan.fit(X_train, epochs=1, batch_size=batch_size)

        if epoch % 20 == 0:
            roc_threshold = give_threshold(gan, 'ROC', X_val, x_val[name][1])
            pr_threshold = give_threshold(gan, 'PR', X_val, x_val[name][1])

            print('INTRA-DATASET EVALUATION:\n')
            evaluate(gan, X_test, x_test[name][1], roc_threshold, pr_threshold)

print('TESTING WITH GAN:\n\n')
for dataKey in datasets.keys():
    fit_and_test_gan(dataKey)