from classifier_central import *
from utils import *
from utils_central import *
import sys
import math
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import PCA

def QuickSort(arr):
    elements = len(arr)
    if elements < 2:
        return arr
    current_position = 0 #Position of the partitioning element
    for i in range(1, elements): #Partitioning loop
        if arr[i]['time'] <= arr[0]['time']:
            current_position += 1
            temp = arr[i]
            arr[i] = arr[current_position]
            arr[current_position] = temp
    temp = arr[0]
    arr[0] = arr[current_position]
    arr[current_position] = temp #Brings pivot to it's appropriate position
    left = QuickSort(arr[0:current_position]) #Sorts the elements to the left of pivot
    right = QuickSort(arr[current_position+1:elements]) #sorts the elements to the right of pivot
    arr = left + [arr[current_position]] + right #Merging everything together
    return arr

def data_parser(data_file):
    raw_data_df = pd.read_csv(data_file)
    student_list = list(set(raw_data_df['student']))
    #print(student_list)
    student_data = dict()
    for student in student_list:
        student_data[student] = []

    #reading in data from csv
    for index, row in raw_data_df.iterrows():
        student = row['student']
        #print(row['data'])
        student_data[student].append(dict(row))

    #removing students below a certain threshold number of actions
    n = 20 #tunable
    keys = [i for i in student_data.keys() if len(student_data[i]) < n]
    for key in keys:
        del student_data[key]

    #sorting by ascending time for each student
    for student in student_data:
        student_data[student] = QuickSort(student_data[student])
        student_data[student] = data_chunker(student_data[student])
    
    return student_data

def data_chunker(actions):
    #print(actions[0]['student'])
    #print(len(actions))
    new_student_data = []
    i = 0
    while i < len(actions):
        action_type = actions[i]['name']
        item = actions[i]['item']
        data = np.fromstring(actions[i]['data'].replace('[', '').replace(']', ''), sep = ',')
        #print(data)
        new_data = [data]
        j = i + 1
        while (j < len(actions) and actions[j]['name'] == action_type and actions[j]['item'] == item):
            new_data.append(np.fromstring(actions[j]['data'].replace('[', '').replace(']', ''), sep = ','))
            j += 1
        new_data = np.array(new_data)
        new_student_data.append({'name':action_type, 'item': item, 'time': actions[i]['time'], 'data': new_data})
        i = j

    #print([f['data'].shape for f in new_student_data])
    return new_student_data
        
def run_TSNE(X, color, label):
    #print(X)
    print(label)
    #X = np.concatenate(X, axis=0)
    X = np.array(X)
    print(X.shape)
    tsne = TSNE(n_components=2, random_state=0)
    Y= tsne.fit_transform(X)
    plt.clf()
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=color)
    plt.title("Encoding, {}".format(label))
    plt.savefig("heads_courseB_{}_VAE_central_latent8.png".format(label))
    return        

def main(data_file, epochs, typ_a):
    print("Starting Simulation")
    student_data = data_parser(data_file)

    #determining maximum length for each type of action
    seq_hist = {'event': [], 'post': [], 'click':[], 'assess':[]}
    max_seq_len = {'event': 0, 'post': 0, 'click':0, 'assess':0}
    for actions in student_data.values():
        for action in actions:
            action_type = action['name']
            seq_hist[action_type].append(action['data'].shape[0])

    #removing outliers (more than 3 std devs away)
    for action_type in max_seq_len:
        mean = float(sum(seq_hist[action_type])) / len(seq_hist[action_type])
        variance = float(sum([((x - mean) ** 2) for x in seq_hist[action_type]])) / len(seq_hist[action_type])
        res = variance ** 0.5
        max_seq_len[action_type] = math.ceil(mean + (3*res))

    #batching data
    data = []
    for student in student_data:
        student_data[student] = [x for x in student_data[student] if x['data'].shape[0] <= max_seq_len[x['name']]]
        data.extend(student_data[student])

    history_heads, history, evalu8, predict = run_sim(data, epochs, max_seq_len, typ_a)
    #history_heads_300, history_300, data_heads_300, model_300 = run_sim(data, 300, max_seq_len, 'heads')
    #history_heads_400, history_400, data_heads_400, model_400 = run_sim(data, 400, max_seq_len, 'heads')
    #history_heads_500, history_500, data_heads_500, model_500 = run_sim(data, 500, max_seq_len, 'heads')

    '''
    plt.clf()
    plt.figure()
    plt.plot(history_300.history['kl_loss'], label = '300 Epochs')
    plt.plot(history_400.history['kl_loss'], label = '400 Epochs')
    plt.plot(history_500.history['kl_loss'], label = '500 Epochs')
    plt.legend(loc='upper right')
    plt.title('Loss Curve - KL-Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.savefig('KL_train_loss_heads_courseB.png')

    plt.clf()
    plt.plot(history_300.history['reconstruction_loss'], label = '300 epochs')
    plt.plot(history_400.history['reconstruction_loss'], label = '400 epochs')
    plt.plot(history_500.history['reconstruction_loss'], label = '500 epochs')
    plt.legend(loc='upper right')
    plt.title('Loss Curve - Reconstruction')
    plt.xlabel('Epochs')
    plt.ylabel('loss (mse)')
    plt.savefig('reconstruction_train_loss_heads_courseB.png')

    plt.clf()
    for typ in history_heads_300.keys():
        plt.plot(history_heads_300[typ].history['reconstruction_loss'], label = typ)
    plt.legend(loc='upper right')
    plt.title('Loss Curve - Reconstruction')
    plt.xlabel('epochs')
    plt.ylabel('loss (cat xe)')
    plt.savefig('heads_reconstruction_train_loss_heads_courseB.png')

    plt.clf()
    for typ in history_heads_300.keys():
        plt.plot(history_heads_300[typ].history['reconstruction_loss'], label = typ)
    plt.legend(loc='upper right')
    plt.title('Loss Curve - KL')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('heads_KL_train_loss_heads_courseB.png')
    '''
    
    all_encodings_heads = []
    all_encodings_central = []

    max_len = max(max_seq_len.values())
    max_features = max([data_point['data'].shape[1] for data_point in data])
    formatted_data_by_type = {'click': [], 'assess': [], 'post': [], 'event': []}
    colors = {'click': 'b', 'assess': 'r', 'post': 'g', 'event': 'y'}
    #print("Encoding for LSTM started (data heads)")
    #X_train, X_test, y_train, y_test = lstm_pipeline(student_data, max_seq_len, model_300, data_heads_300) 
    #print("Prediction for LSTM beginning (data heads)")
    #lstm_predict(X_train, X_test, y_train, y_test)
    print("Encoding for LSTM started (central)")
    X_train, X_test, y_train, y_test = lstm_pipeline(student_data, max_seq_len, predict)
    print("Prediction for LSTM beginning (central)")
    lstm_predict(X_train, X_test, y_train, y_test)

    '''
    print("Running TSNE")
    run_TSNE(all_encodings_300, all_colors_300, '300 epochs')
    run_TSNE(all_encodings_400, all_colors_400, '400 epochs')
    run_TSNE(all_encodings_500, all_colors_500, '500 epochs')

    all_encodings_300 = np.array(all_encodings_300)
    all_encodings_400 = np.array(all_encodings_400)
    all_encodings_500 = np.array(all_encodings_500)
    pca = PCA(n_components = 2)
    Y= pca.fit_transform(all_encodings_300)
    Y_2 = pca.fit_transform(all_encodings_400)
    Y_3 = pca.fit_transform(all_encodings_500)
    plt.clf()
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=all_colors_300)
    plt.title("PCA (300 Epochs)")
    plt.savefig("heads_pca_300_central_courseB.png")
    plt.clf()
    plt.figure()
    plt.scatter(Y_2[:, 0], Y_2[:, 1], c=all_colors_400)
    plt.title("PCA (400 Epochs)")
    plt.savefig("heads_pca_400_central_courseB.png")
    plt.clf()
    plt.figure()
    plt.scatter(Y_3[:, 0], Y_3[:, 1], c=all_colors_500)
    plt.title("PCA (500 Epochs)")
    plt.savefig("heads_pca_500_central_courseB.png")
    '''
    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
