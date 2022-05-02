import numpy as np
import pandas as pd
import keras
from keras.preprocessing import sequence
from keras import Model
from keras.models import Sequential, Model
from keras.layers import Dropout, Cropping1D, Dense, Input, LSTM, RepeatVector, TimeDistributed
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def lstm_pipeline(data, max_seq_lens, model, data_heads = None):
    data_types = ['assess', 'click', 'post', 'event']
    data_by_student = dict()
    y_by_student = dict()
    students = data.keys()
    for student in students:
        data_by_student[student] = []
    for student in data:
        for data_point in data[student]:
            data_type = data_point['name']
            data_formatted = data_point['data']
            if (data_heads == None):
                max_len = max(max_seq_lens.values())
                max_features = 28
                data_formatted = np.vstack((data_formatted, np.zeros((max_len - data_formatted.shape[0], data_formatted.shape[1]))))
                data_formatted = np.hstack((data_formatted, np.zeros((max_len, max_features - data_formatted.shape[1]))))
                data_formatted = np.expand_dims(data_formatted, axis = -1)
                data_formatted = np.expand_dims(data_formatted, axis = 0)
                history = model.encoder.predict(data_formatted)
            else:
                max_len = max_seq_lens[data_type]
                data_formatted = np.vstack((data_formatted, np.zeros((max_len - data_formatted.shape[0], data_formatted.shape[1]))))
                data_formatted = np.expand_dims(data_formatted, axis = -1)
                data_formatted = np.expand_dims(data_formatted, axis = 0)
                _, _, intermediate_data = data_heads[data_type].encoder.predict(data_formatted)
                history = model.encoder.predict(intermediate_data)
            data_by_student[student].extend(history[2])

    window = 19
    X,y = [],[]
    for student in data_by_student.keys():
        for i in range(len(data_by_student[student]) - window -1):
            X.append(data_by_student[student][i:i+window])
            y.append(data_by_student[student][i+window+1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 43)
    return X_train, X_test, y_train, y_test

def lstm_predict(X_train, X_test, y_train, y_test):
    #num_timesteps = len(X_train[0])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    num_timesteps = X_train.shape[1]
    num_features = X_train.shape[2]
    model = Sequential()
    model.add(LSTM(4, input_shape=(num_timesteps, num_features)))
    model.add(Dense(num_features, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=1)
    model.evaluate(X_test, y_test)
    
def type_student_pipeline(data, models):
    data_types = ['post', 'click', 'event']
    students = list(set([data_point['student'] for data_point in data]))
    data_by_student = dict()
    y_by_student = dict()
    max_seq_len = 0
    for i in range(len(data)):
        if (data[i]['name'] == 'click' and len(data[i]['data']) > max_seq_len):
            max_seq_len = len(data[i]['data'])
    for student in students:
        data_by_student[student] = []
        y_by_student[student] = []
    
    for data_point in data:
        data_type = data_point['name']
        ind = data_types.index(data_type)
        y1 = [0,0,0]
        y1[ind] = 1
        student = data_point['student']
        y2 = [0] * len(students)
        ind = students.index(student)
        y2[ind] = 1

        data_formatted = np.array(data_point['data'])
        data_shape = data_formatted.shape
        if (not(data_point['name'] == 'click') or len(data_shape) == 1):
            data_formatted = np.expand_dims(data_formatted, 0) #add channel dimension
        else:
            data_formatted = np.vstack((data_formatted, np.zeros((max_seq_len - data_formatted.shape[0], 48))))
        data_formatted = np.expand_dims(data_formatted, 0)
        #data_encoded = model.data_heads[data_type].encoder.predict(data_formatted)
        #history = model.autoencoder.encoder.predict(data_encoded)
        history = models[data_type].encoder.predict(data_formatted)
        data_by_student[student].append(history[0])
        y_by_student[student].append(y1+y2)

    X_train, X_test, y_train, y_test = [], [], [], []
    for student in students:
        if (len(data_by_student[student]) > 1):
            X_student_train, X_student_test, y_student_train, y_student_test = train_test_split(data_by_student[student], y_by_student[student], test_size=0.5, random_state = 42)
            X_train.extend(X_student_train)
            X_test.extend(X_student_test)
            y_train.extend(y_student_train)
            y_test.extend(y_student_test)
    return X_train, X_test, y_train, y_test
            
def type_pipeline(models):
    students = [model.name for model in models]
    X = []
    y = []
    data_types = ['post', 'click', 'event']
    bins = [0,0,0]
    X_data_type = dict()
    for data_type in data_types:
        X_data_type[data_type] = []
    for model in models:
        if (model.name in students):
            for data_point in model.data:
                data_type = data_point['name']
                #if (data_type in model.data_heads.keys()):
                if (data_type in model.encoders.keys()):
                    ind = data_types.index(data_type)
                    y_ind = [0,0,0]
                    y_ind[ind] = 1
                    y.append(y_ind)
                    bins[ind] += 1
                    data_formatted = np.array(data_point['data'])
                    data_shape = data_formatted.shape
                    if (not(data_point['name'] == 'click') or len(data_shape) == 1):
                        data_formatted = np.expand_dims(data_formatted, 0) #add channel dimension
                    data_formatted = np.expand_dims(data_formatted, 0)
                    #data_encoded = model.data_heads[data_type].encoder.predict(data_formatted)
                    #history = model.autoencoder.encoder.predict(data_encoded)
                    history = model.encoders[data_type].encoder.predict(data_formatted)
                    X_data_type[data_type].append(history[0])

    X_train, X_test, y_train, y_test = [], [], [], []
    for data_type in data_types:
        if (len(X_data_type[data_type]) > 1):
            X_data_type_train, X_data_type_test = train_test_split(X_data_type[data_type], test_size=0.5, random_state = 42)
            X_train.extend(X_data_type_train)
            X_test.extend(X_data_type_test)
            temp = [0,0,0]
            ind = data_types.index(data_type)
            temp[ind] = 1
            y_train = y_train + [temp]*len(X_data_type_train)
            y_test = y_test + [temp]*len(X_data_type_test)
    print(bins)
    return X_train, X_test, y_train, y_test

def type_class(X_train, X_test, y_train, y_test):
    if (len(X_train) > 0 and len(X_test) > 0):
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        #print(X_train.shape)
        #print(X_test.shape)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        model = Sequential()
        model.add(Input(shape = (8,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20, batch_size=32)
        history = model.evaluate(X_test, y_test)
        #print(history)

def student_class(X_train, X_test, y_train, y_test):
    num_feats = len(y_test[0])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = Sequential()
    model.add(Input(shape=(8,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_feats, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    history = model.evaluate(X_test, y_test)
    print(history)
    
def class_pipeline(models, grades):
    students = [model.name for model in models]
    X = [[]] * len(students)
    y = [0] * len(students)
    X_reg = [[]] * len(students)

    for model in models:
        if (model.name in students):
            ind = students.index(model.name)
            for data_point in model.data:
                data_type = data_point['name']
                data_formatted = np.array(data_point['data'])
                data_shape = data_formatted.shape
                if (not(data_point['name'] == 'click') or len(data_shape) == 1):
                    data_formatted = np.expand_dims(data_formatted, 0) #add channel dimension
                    #print(data_formatted.shape)
                data_formatted = np.expand_dims(data_formatted, 0)
                #if (data_type in model.encoders.keys()):
                if (data_type in model.data_heads.keys()):
                    data_encoded = model.data_heads[data_type].encoder.predict(data_formatted)
                    history = model.autoencoder.encoder.predict(data_encoded)
                    #history = model.encoders[data_type].encoder.predict(data_formatted)
                    X[ind].append(history[0])
                    #print(history)
                    X_reg[ind].extend(history[0])

            for i in range(len(students)):
                if (int(students[i]) in grades.keys()):
                    y[i] = grades[int(students[i])]

    #print(y)
    print(len([i for i in y if y[i] == 0]))
    print(len([i for i in y if y[i] == 1]))
    return X, X_reg, y

def grades_class(X_train, X_test, X_reg_train, X_reg_test, y_train, y_test):
    print("classifier running")
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    max_seq_len = 0
    for i in range(len(X_train)):
        if (len(X_train[i]) > max_seq_len):
            max_seq_len = len(X_train[i])

    for i in range(len(X_test)):
        if (len(X_test[i]) > max_seq_len):
            max_seq_len = len(X_test[i])

    max_len = 0
    for i in range(len(X_reg_train)):
        if (len(X_reg_train[i]) > max_len):
            max_len = len(X_reg_train[i])

    print(max_len)
    for i in range(len(X_reg_test)):
        #print(len(X_reg_test[i]))
        if (len(X_reg_test[i]) > max_len):
            max_len = len(X_reg_test[i])

    #print(X_reg_test)
    for i in range(len(X_reg_train)):
        length = max_len - len(X_reg_train[i])
        for j in range(length):
            X_reg_train[i].append(0)

    for i in range(len(X_reg_test)):
        #print(max_len)
        #print(len(X_reg_test[i]))
        #print(X_reg_test[i])
        length = max_len - len(X_reg_test[i])
        #print("Length: {}".format(length))
        for j in range(length):
            #print(max_len-len(X_reg_test[i]))
            X_reg_test[i].append(0)

    #print(max_len)
    if (max_seq_len > 0):
        for i in range(len(X_train)):
            X_train[i] = X_train[i] + ([[0,0,0,0,0,0,0,0]]*(max_seq_len - len(X_train[i])))

        num_samples = len(X_train)
        X_train = np.array(X_train)
        X_train = X_train.reshape(num_samples, max_seq_len, 8)
        #print(max_seq_len)
        #print(X_train.shape)
        y_train = np.array(y_train)

        y_test = np.array(y_test)
        for i in range(len(X_test)):
            #print(X_test[i])
            X_test[i] = X_test[i] + ([[0,0,0,0,0,0,0,0]]*(max_seq_len - len(X_test[i])))
            #print(X_test[i])
            
        X_test = np.array(X_test)
        #print(X_test.shape)
        X_test = X_test.reshape(-1, max_seq_len, 8)
        
        model = Classifier(max_seq_len, 8, 1)
        model.model.fit(X_train, y_train, batch_size=32, epochs=20)
        history = model.model.evaluate(X_test, y_test)
        print(history)
        history = model.model.predict(X_test)
        
        model_2 = Sequential()
        model_2.add(Input(shape=(max_len,)))
        model_2.add(Dense(32, activation='relu'))
        model_2.add(Dense(1, activation='sigmoid'))
        model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #X_train2, X_test2, y_train2, y_test2 = train_test_split(X_reg, y, test_size=0.3)
        X_train2 = np.array(X_reg_train)
        #print(X_train2.shape)
        y_train2 = np.array(y_train)
        y_test2 = np.array(y_test)
        X_test2 = np.array(X_reg_test)
        #print(X_test2.shape)
        #model_2.fit(X_train2, y_train2, batch_size = 32, epochs=20)
        history = model_2.evaluate(X_test2, y_test2)
        #print(history)
        #history = model_2.predict(X_test2)
    
    return
    
 
class Classifier(Model):
    def __init__(self, timesteps, n_features, n_classes):
        super(Classifier, self).__init__()
        self.model = self.build_model(timesteps, n_features, n_classes)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def build_model(self, timesteps, n_features, n_classes):
        model = Sequential()
        model.add(Input(shape = (timesteps, n_features)))
        model.add(LSTM(32))
        model.add(Dense(n_classes, activation='sigmoid'))

        #print(model.summary())
        return model

    def call(self, x):
        return self.model(x)
