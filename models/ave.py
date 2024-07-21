import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

# Define VAE model architecture based on your provided structure
class VAE(Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = tf.keras.Sequential([
            Dense(16, activation='relu'),
            Dense(8, activation='relu')
        ])
        
        # Latent space layers
        self.z_mean = Dense(latent_dim, name='z_mean')
        self.z_log_var = Dense(latent_dim, name='z_log_var')
        
        # Reparameterization trick
        self.sampling = Sampling()
        
        # Decoder layers
        self.decoder = tf.keras.Sequential([
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(input_dim, activation='linear')
        ])
    
    def call(self, inputs):
        # Encoder
        x = self.encoder(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        # Reparameterization trick
        z = self.sampling([z_mean, z_log_var])
        
        # Decoder
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

# Custom layer for reparameterization
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Loss function for VAE
def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    total_loss = reconstruction_loss + kl_loss
    return total_loss


def create_generic_model_ave():
    # Example usage:
    input_dim = 10
    latent_dim = 4  # Assuming a 2-dimensional latent space for visualization

    # Instantiate the VAE model
    vae_model = VAE(input_dim, latent_dim)

    # Define input and compile the model with custom loss function
    inputs = Input(shape=(input_dim,))
    outputs, z_mean, z_log_var = vae_model(inputs)
    vae_model = Model(inputs, outputs)
    vae_model.add_loss(vae_loss(inputs, outputs, z_mean, z_log_var))
    vae_model.compile(optimizer='adam')
    return vae_model