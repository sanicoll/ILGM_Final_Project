import numpy as np
import scipy.spatial as sp
import tensorflow as tf
from tensorflow import keras
from keras import Model
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from keras.layers import Layer, Reshape, Conv2D, Dropout, BatchNormalization, Flatten, Cropping1D, Dense, Input, LSTM, RepeatVector, TimeDistributed
from keras import optimizers
from keras import regularizers
import os, psutil
from keras.utils import np_utils
from keras.backend import int_shape, var

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Conv_Encoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Conv_Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mean = None
        self.var = None
        self.encoder = self.build_encoder()

    def build_encoder(self):
        x = Input(shape = self.input_dim)
        hidden_1 = Conv2D(64, (1,1), activation='relu')(x)
        hidden_1 = BatchNormalization()(hidden_1)
        hidden_1 = Dropout(0.25)(hidden_1)
        print(hidden_1.shape)
        hidden_2 = Conv2D(64, (1,1), activation='relu')(hidden_1)
        hidden_2 = BatchNormalization()(hidden_2)
        hidden_2 = Dropout(0.25)(hidden_2)
        print(hidden_2.shape)
        hidden_3 = Conv2D(64, (1,1), activation = 'relu')(hidden_2)
        hidden_3 = BatchNormalization()(hidden_3)
        hidden_3 = Dropout(0.25)(hidden_3)
        print(hidden_3.shape)
        flat = Flatten()(hidden_3)
        z_mu = Dense(self.latent_dim)(flat)
        z_logvar = Dense(self.latent_dim)(flat)
        z = Sampling()([z_mu, z_logvar])
        encoder = Model(x, [z_mu, z_logvar, z], name = 'Conv_Encoder')
        return encoder

    def call(self, x):
        outputs = self.encoder(x)
        return outputs

class Conv_Decoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Conv_Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.decoder = self.build_decoder()

    def build_decoder(self):
        x = Input(shape = (self.latent_dim,))
        x_1 = RepeatVector(80)(x)
        print(x_1.shape)
        lstm_out = LSTM(80)(x_1)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.25)(lstm_out)
        print(lstm_out.shape)
        output = Dense(self.input_dim[0]*self.input_dim[1], activation = 'sigmoid')(lstm_out)
        print(output.shape)
        output = Reshape(self.input_dim)(output)
        print(output.shape)
        decoder = Model(x, output, name = 'Conv_Decoder')
        return decoder

    def call(self, x):
        outputs = self.decoder(x)
        return outputs
        
class FC_Encoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(FC_Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_dim, latent_dim)
        #print(self.encoder.summary())

    def build_encoder(self, input_dim, latent_dim):
        #print(input_dim)
        x = Input(shape = input_dim)
        h = Dense((latent_dim + input_dim[0])/2, activation = 'relu')(x)
        z_mu = Dense(latent_dim)(h)
        z_logvar = Dense(latent_dim)(h)
        z = Sampling()([z_mu, z_logvar])
        encoder = Model(x, [z_mu, z_logvar, z], name="FC_Encoder")
        return encoder

    def call(self, x):
        outputs = self.encoder(x)
        return outputs

class FC_Decoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(FC_Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.decoder = self.build_decoder(input_dim, latent_dim)

    def build_decoder(self, input_dim, latent_dim):
        decoder = Sequential()
        decoder.add(Input(shape = (latent_dim,)))
        decoder.add(Dense((latent_dim + input_dim[0])/2, activation = 'relu'))
        decoder.add(Dense(input_dim[0], activation='relu', name='bottleneck_reverse'))
        return decoder

    def call(self, x):
        outputs = self.decoder(x)
        return outputs

class VAE(Model):
    def __init__(self, name, loss_type, input_dim, latent_dim):
        super(VAE, self).__init__()
        if (name == 'conv'):
            print("Building Encoder")
            self.encoder = Conv_Encoder(input_dim, latent_dim)
            print("Building Decoder")
            self.decoder = Conv_Decoder(input_dim, latent_dim)
        elif (name == 'fc'):
            self.encoder = FC_Encoder(input_dim, latent_dim)
            self.decoder = FC_Decoder(input_dim, latent_dim)
        else:
            raise Exception("Encoder type {} not recognized. Allowed keywords: fc, conv".format(name))
        print("Finished Building")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.loss_type = loss_type

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #print(reconstruction.shape)
            #print(reconstruction[0][0][0])
            #print(data[0][0][0])
            print(self.loss_type)
            if (self.loss_type == 'cce'):
                data = tf.squeeze(data, axis = [-1])
                reconstruction = tf.squeeze(reconstruction, axis = [-1])
                print("Training with categorical cross-entropy loss")
                reconstruction_loss = tf.reduce_sum(keras.losses.categorical_crossentropy(data, reconstruction))
            else:
                print("Training with mean squared error loss")
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded[2])
        return decoded    
    
class LSTM_Head(Model):
    def __init__(self, name, timesteps, n_features, latent_dim):
        super(Data_head, self).__init__()
        self.encoder = None
        self.decoder = None
        self.build_head(timesteps, n_features, latent_dim)

    def build_head(self, timesteps, n_features, latent_dim):
        encoder_inputs = Input(shape = (timesteps, n_features))
        encoder_1 = LSTM(64, return_state=True)
        encoder_outputs, state_h, state_c = encoder_1(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape = (latent_dim,))
        repeat = RepeatVector(timesteps)
        decoder_inputs_int = repeat(decoder_inputs)
        decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
        decoder_int, _, _ = decoder_lstm(decoder_inputs_int)#, initial_state=encoder_states)
        decoder_dense = Dense(n_features, activation='sigmoid')
        decoder_outputs = decoder_dense(decoder_int)

        decoder_model = Model(decoder_inputs, decoder_outputs)
        self.decoder = decoder_model

        encoder_model = Model(encoder_inputs, encoder_outputs)
        self.encoder = encoder_model

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Contrastive(Model):
    def __init__(self, num_data_types, input_dim):
        super(Contrastive, self).__init__()
        self.encoder = None
        self.classifier = self.build_classifier(num_data_types, input_dim)

    def build_classifier(self, num_data_types, input_dim):
        inputs = Input(shape=input_dim)
        dense_1 = Dense(32, activation="relu")
        features = dense_1(inputs)
        dropout_1 = Dropout(0.3)
        features = dropout_1(features)
        dense_2 = Dense(num_data_types, activation="softmax")
        outputs = dense_2(features)
        class_model = Model(inputs, outputs)
        return class_model

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        classified = self.classifier(z)
        return classified
