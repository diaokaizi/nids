import tensorflow as tf
from keras.models import Model
class AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Reshape((input_dim, 1)),  # Reshape to 3D for Conv1D
            tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2, padding="same"),
            tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2, padding="same"),
            tf.keras.layers.Conv1D(latent_dim, 3, strides=1, activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2, padding="same"),
        ])
        # Previously, I was using UpSampling. I am trying Transposed Convolution this time around.
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(latent_dim, 3, strides=1, activation='relu', padding="same"),
#             tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
#             tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
#             tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(input_dim)
        ])

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded


def create_generic_model_cnn():
    # Example usage:
    input_dim = 10
    latent_dim = 4  # Assuming a 2-dimensional latent space for visualization

    model = AutoEncoder(input_dim, latent_dim)
    model.build((None, input_dim))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")
    model.summary()
    return model
