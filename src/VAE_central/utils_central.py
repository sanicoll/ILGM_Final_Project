from utils import *
from sklearn.model_selection import train_test_split

def run_sim(data, ep, max_seq_lens, code):
    formatted_data = []
    if (code == 'heads'):
        data_types = list(set([data_point['name'] for data_point in data]))
        data_heads = dict()
        formatted_data_by_type = dict()
        history_by_type = dict()
        intermediate_data = []
        for typ in data_types:
            formatted_data_by_type[typ] = []
        for data_point in data:
            v_pad = np.zeros((max_seq_lens[data_point['name']] - data_point['data'].shape[0], data_point['data'].shape[1]))
            data_formatted = np.vstack((data_point['data'], v_pad))
            data_formatted = np.expand_dims(data_formatted, axis = -1)
            formatted_data_by_type[data_point['name']].append(data_formatted)

        for typ in data_types:
            data_heads[typ] = VAE('conv', 'cce', (max_seq_lens[typ], formatted_data_by_type[typ][0].shape[1], 1), 64)
            data_heads[typ].compile(optimizer = 'adam')
            data_train = np.array(formatted_data_by_type[typ])
            print("Elements in data type {}: {}".format(typ, data_train.shape[0]))
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=8)
            history = data_heads[typ].fit(data_train, batch_size = min(100, data_train.shape[0]), epochs = int(ep/2), callbacks = [callback], verbose = False)
            history_by_type[typ] = history
            pred = data_heads[typ].encoder.predict(data_train)
            intermediate_data.extend(pred[2])
        intermediate_data = np.array(intermediate_data)
        print(intermediate_data.shape)

        model = VAE('fc', 'mse', (64,), 8)
        model.compile(optimizer = 'adam')
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = model.fit(intermediate_data, batch_size = 500, epochs = int(ep/2), callbacks = [callback])
        return history_by_type, history, data_heads, model
        
    elif (code == 'central'):
        max_features = max([data_point['data'].shape[1] for data_point in data])
        max_timesteps = max(max_seq_lens.values())
        model = VAE('conv', 'cce', (max_timesteps, max_features, 1), 8)
        for data_point in data:
            v_pad = np.zeros((max_timesteps - data_point['data'].shape[0], data_point['data'].shape[1]))
            data_formatted = np.vstack((data_point['data'], v_pad))
            h_pad = np.zeros((max_timesteps, max_features - data_point['data'].shape[1]))
            data_formatted = np.hstack((data_formatted, h_pad))
            data_formatted = np.expand_dims(data_formatted, axis=-1)
            formatted_data.append(data_formatted)
        model.compile(optimizer = 'adam')
        X_train, X_test = train_test_split(formatted_data, test_size = 0.3, random_state = 42, shuffle = True)
        X_train = np.array(X_train)
        print(X_train.shape)
        X_test = np.array(X_test)
        print(X_test.shape)
        print("Number of Data Samples: {}".format(len(formatted_data)))
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = model.fit(X_train, batch_size = 500, epochs = ep, callbacks = [callback])
        evalu8 = model.evaluate(X_test, X_test)
        return None, history, evalu8, model
    else:
        raise Exception("{} not recognized as a valid mode. Allowed modes: heads, central".format(code))

'''
    formatted_data_by_type = dict()
    for item in data_types:
        formatted_data_by_type[item] = []
        
    for data_point in data:
        data_formatted = data_point['data']
        if (data_point['name'] in encoders.keys()):
            formatted_data_by_type[data_point['name']].append(data_formatted)
        else:
            formatted_data_by_type[data_point['name']] = [data_formatted]
            data_types.append(data_point['name'])
            encoders[data_point['name']] = local_model(data_point['name'], input_shape[0], input_shape[1], 8)

        for data_type in formatted_data_by_type:
            print(data_type)
            data_test = np.array(formatted_data_by_type[data_type])
            history = encoders[data_type].model.fit(data_test, data_test, batch_size=128, epochs=10, verbose=0)
            encoders[data_type].losses_list.extend(history.history['loss'])
            encoders[data_type].acc_list.extend(history.history['accuracy'])
            
    return encoders
'''
