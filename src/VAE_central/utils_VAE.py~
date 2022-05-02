from classifier import *
import numpy as np
import scipy.spatial as sp
import tensorflow as tf
from tensorflow import keras
from keras import Model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Layer, Dropout, Cropping1D, Dense, Input, LSTM, RepeatVector, TimeDistributed
from keras import optimizers
from keras import regularizers
import os, psutil
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.backend import int_shape, var
from datetime import datetime

class User(object):
    def __init__(self, name, data_series, max_seq_len):
        #name of student associated with the node
        self.name = str(name)
        self.max_seq_lens = max_seq_len
        self.autoencoder = VAE((64,), 8)
        self.autoencoder.compile(optimizer='adam')
        self.KL = []
        self.recon = []

        #note that the contrastive output has higher dimension than the input from the encoder
        self.contrast = Contrastive(1, (8,))
        self.contrast.encoder = self.autoencoder.encoder
        self.contrast.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.data_heads = dict()
        self.losses = {'contrast': [], 'autoencoder': []}
        self.weights = {'contrast': [], 'autoencoder': []}
        self.acc_list = {'contrast': [], 'autoencoder': []}
        self.data = data_series
        self.data_types = []
        self.times = []
        self.Server = None

    def set_Server(self, server):
        self.Server = server
        
    def local(self, time_a, time_b):
        #print(self.name + " currently training")
        current_data = []
        formatted_data_by_type = dict()
        transition_by_type = dict()
        for item in self.data_types:
            formatted_data_by_type[item] = []
            transition_by_type[item] = []
            
        for data_point in self.data:
            stu_time = datetime.fromisoformat(data_point['time'])
            if(stu_time <= time_b):
                current_data.append(data_point)

        for i in range(len(current_data)): #data_point in current_data:
            data_point = current_data[i]
            data_formatted = np.array(data_point['data'])
            data_shape = data_formatted.shape

            if (len(data_shape) == 1):
                data_formatted = np.expand_dims(data_formatted, 0) #add channel dimension

            if (data_point['name'] not in self.data_types):
                t = data_point['name']
                self.data_types.append(t)
                #print("New Data Type(s)")
                self.contrast = Contrastive(len(self.data_types)**2, (8,))
                self.contrast.encoder = self.autoencoder.encoder
                self.contrast.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

                if (len(self.data_types)**2 in self.Server.contrast.keys()):
                    final_weights = self.Server.contrast[len(self.data_types)**2]
                    self.contrast.classifier.set_weights(final_weights)
        
                transition_by_type[t] = []
                formatted_data_by_type[t] = []
                self.losses[t] = []
                self.weights[t] = []
                self.acc_list[t] = []

                self.data_heads[t] = Data_head(t + '_' + self.name, self.max_seq_lens[t], data_formatted.shape[1], 64)
                #self.data_heads[t].autoencoder = self.autoencoder
                if (t in self.Server.encoders.keys()):
                    self.data_heads[t].encoder.set_weights(self.Server.encoders[t])
                    self.data_heads[t].decoder.set_weights(self.Server.decoders[t])
                self.data_heads[t].build((None, self.max_seq_lens[t], data_formatted.shape[1]))
                self.data_heads[t].compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
                    
        for i in range(len(current_data)-1):
            data_point = current_data[i]
            next_data_point = current_data[i+1]
            data_formatted = np.array(data_point['data'])
            data_shape = data_formatted.shape
            if (len(data_shape) == 1):
                data_formatted = np.expand_dims(data_formatted, 0) #add channel dimension
            
            curr_ind = self.data_types.index(data_point['name'])
            next_ind = self.data_types.index(next_data_point['name'])
            transition = np.zeros((len(self.data_types)**2,))
            ind = curr_ind * len(self.data_types) + next_ind
            transition[ind] = 1
            transition_by_type[data_point['name']].append(transition)
            formatted_data_by_type[data_point['name']].append(data_formatted)
        
        concatenated_outputs = []
        concatenated_labels = []
        for data_type in formatted_data_by_type:
            max_len = self.max_seq_lens[data_type]
            for m in range(len(formatted_data_by_type[data_type])):
                temp_data = formatted_data_by_type[data_type][m]
                formatted_data_by_type[data_type][m] = np.concatenate((temp_data, np.zeros((max_len - temp_data.shape[0], temp_data.shape[1]))), axis = 0)
            if (len(formatted_data_by_type[data_type]) > 0):
                data = np.stack(formatted_data_by_type[data_type], axis = 0)
                history = self.data_heads[data_type].fit(data, data, batch_size=1, epochs=10, verbose=0)
                self.losses[data_type].extend(history.history['loss'])
                self.acc_list[data_type].extend(history.history['accuracy'])
                concatenated_outputs.extend(self.data_heads[data_type].encoder.predict(data))
                concatenated_labels.extend(transition_by_type[data_type])
            
        if (len(concatenated_outputs) > 0):
            concatenated_outputs = np.stack(concatenated_outputs, axis = 0)
            data_labels = np.stack(concatenated_labels, axis = 0)
            history = self.autoencoder.fit(concatenated_outputs, batch_size = len(data_labels), epochs = 10, shuffle=True, verbose = 0)
            print(history)
            self.recon.extend(history.history['reconstruction_loss'])
            self.KL.extend(history.history['kl_loss'])
            history = self.contrast.fit(concatenated_outputs, data_labels, batch_size=len(data_labels), epochs=10, shuffle=True, verbose=0)
            self.losses['contrast'].extend(history.history['loss'])
            self.acc_list['contrast'].extend(history.history['accuracy'])
        pass
        
class Server(object):
    def __init__(self, num_aggregations, clients):
        self.num_aggregations = num_aggregations
        self.data_types = []
        self.encoders = dict()
        self.decoders = dict()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.contrast = dict()

    def aggregate_contrast(self):
        for client in self.clients:
            if (len(client.data_types)**2 not in self.contrast.keys() and len(client.data_types) > 1):
                self.contrast[len(client.data_types)**2] = []

        for length in self.contrast.keys():
            weights = []
            for client in self.clients:
                if (len(client.data_types)**2 == length):
                    weights.append(client.contrast.classifier.get_weights())

            self.contrast[length] = self.FedAvg(weights)

        enc_weights = []
        dec_weights = []
        for client in self.clients:
            enc_weights.append(client.contrast.encoder.get_weights())
            dec_weights.append(client.autoencoder.decoder.get_weights())
        enc_weights = self.FedAvg(enc_weights)
        dec_weights = self.FedAvg(dec_weights)
        for client in self.clients:
            client.contrast.encoder.set_weights(enc_weights)
            client.autoencoder.encoder.set_weights(enc_weights)
            client.autoencoder.decoder.set_weights(dec_weights)
                        
    def aggregate_data_heads(self):                
        for client in self.clients:
            for data_type in client.data_types:
                if (data_type not in self.data_types):
                    self.data_types.append(data_type)

        for data_type in self.data_types:
            enc_weights = []
            dec_weights = []
            for client in self.clients:
                if data_type in client.data_heads.keys():
                    enc_weights.append(client.data_heads[data_type].encoder.get_weights())
                    dec_weights.append(client.data_heads[data_type].decoder.get_weights())
            self.encoders[data_type] = self.FedAvg(enc_weights)
            self.decoders[data_type] = self.FedAvg(dec_weights)
            
            for client in self.clients:
                if (data_type in client.data_types):
                    client.data_heads[data_type].encoder.set_weights(self.encoders[data_type])
                    client.data_heads[data_type].decoder.set_weights(self.decoders[data_type])

    def run_sim(self, start_time, end_time):
        print("Number of Aggregations: {}".format(self.num_aggregations))
        period = (end_time - start_time) / self.num_aggregations
        for n in range(self.num_aggregations):
            print("Round: {}".format(n+1))
            print("Time: {}".format(str(start_time + (n+1)*period)))
            for client in self.clients:
                client.local(start_time + n*period, start_time + (n+1)*period)
            print("Aggregation {}".format(n+1))
            self.aggregate_data_heads()
            self.aggregate_contrast()
            
        #X_train, X_test, y_train, y_test = lstm_pipeline(self.clients)
        #lstm_predict(X_train, X_test, y_train, y_test)
        return self.clients
    
    def FedAvg(self, w):
        avg_weights = []
        #print(w)
        for i in range(len(w[0])):
            avg_weights.append(np.zeros(w[0][i].shape))
        for i in range(len(w)):
            for j in range(len(w[0])):
                avg_weights[j] = np.add(avg_weights[j],w[i][j])

        for j in range(len(w[0])):
            avg_weights[j] = avg_weights[j]/float(len(w))

        return avg_weights

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mean = None
        self.var = None
        self.encoder = self.build_encoder(input_dim, latent_dim)
        #print(self.encoder.summary())

    def build_encoder(self, input_dim, latent_dim):
        #print(input_dim)
        x = Input(shape = input_dim)
        h = Dense((latent_dim + input_dim[0])/2, activation = 'relu')(x)
        z_mu = Dense(latent_dim)(h)
        z_logvar = Dense(latent_dim)(h)
        z = Sampling()([z_mu, z_logvar])
        encoder = Model(x, [z_mu, z_logvar, z], name="encoder")
        #encoder.summary()
        return encoder

    def call(self, x):
        outputs = self.encoder(x)
        self.mean = outputs[0]
        self.var = tf.exp(outputs[1])
        #print(self.mean)
        #print(self.var)
        return outputs

class Decoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()
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
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        #print(input_dim)
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print(keras.losses.mean_squared_error(data, reconstruction))
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        #self.mean = z_mean
        #self.var = tf.exp(z_log_var)
        return {"loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}
        
    #def call(self, x):
    #    #encoded = self.encoder(x)
    #    #decoded = self.decoder(encoded)
    #    return self.autoencoder(x)

class Data_head(Model):
    def __init__(self, name, timesteps, n_features, latent_dim):
        super(Data_head, self).__init__()
        self.encoder = None
        self.decoder = None
        self.build_head(timesteps, n_features, latent_dim)

    def build_head(self, timesteps, n_features, latent_dim):
        #print("{}, {}".format(timesteps, n_features))
        #print(latent_dim)
        encoder_inputs = Input(shape = (timesteps, n_features))
        #print(encoder_inputs.shape)
        encoder_1 = LSTM(64, return_state=True)
        encoder_outputs, state_h, state_c = encoder_1(encoder_inputs)
        #print(encoder_outputs.shape)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape = (latent_dim,))
        #print(decoder_inputs.shape)
        repeat = RepeatVector(timesteps)
        decoder_inputs_int = repeat(decoder_inputs)
        #print(decoder_inputs_int.shape)
        decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
        decoder_int, _, _ = decoder_lstm(decoder_inputs_int)#, initial_state=encoder_states)
        #print(decoder_int.shape)
        decoder_dense = Dense(n_features, activation='sigmoid')
        decoder_outputs = decoder_dense(decoder_int)
        #print(decoder_outputs.shape)

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


