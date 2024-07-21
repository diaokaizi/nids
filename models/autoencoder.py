import tensorflow as tf

def create_generic_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(10,)))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model