from utils_VAE import *
import math
from matplotlib.patches import Ellipse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys

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
    X = np.concatenate(X, axis=0)
    print(X.shape)
    #X = np.array(X)
    tsne = TSNE(n_components=2, random_state=0)
    Y= tsne.fit_transform(X)
    plt.clf()
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=color)
    plt.title("Encoding by {}".format(label))
    plt.savefig("courseB_{}_VAE_latent8.png".format(label))
    return

def confidence_ellipse(cov, ax, mean, n_std=1):
    ellipse = Ellipse((mean[0],mean[1]), width=cov[0] * (2*n_std), height=cov[1] * (2*n_std), alpha = 0.5)#, fill = False)
    ax.add_patch(ellipse)

    return ax

def plt_VAE(means, var_s):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    for i in range(len(means)):
        ax = confidence_ellipse(var_s[i], ax, means[i])
    plt.savefig("VAR_overlap.png")
        

def main(data_file, num_agg):
    print("Starting Simulation")
    student_data = data_parser(data_file)
    '''
    n, bins, patches = plt.hist(x, 100, range = (0, 250))
    plt.xlabel("Actions Per Student")
    plt.title("Distribution of Student Actions")
    plt.grid(True)
    plt.savefig('Activity_Dist.png')
    '''
    
    first_time = datetime.fromisoformat('2024-01-16 00:05:23.283')
    last_time = datetime.fromisoformat('2018-08-01 00:05:23.283')
    #period = 20
    seq_hist = {'event': [], 'post': [], 'click':[], 'assess':[]}
    max_seq_len = {'event': 0, 'post': 0, 'click':0, 'assess':0}
    for actions in student_data.values():
        for action in actions:
            time = datetime.fromisoformat(action['time'])
            action_type = action['name']
            seq_hist[action_type].append(action['data'].shape[0])
            if (time < first_time):
                first_time = time
            if (time > last_time):
                last_time = time

    #removing outliers (more than 3 std devs away)
    for action_type in max_seq_len:
        mean = float(sum(seq_hist[action_type])) / len(seq_hist[action_type])
        variance = float(sum([((x - mean) ** 2) for x in seq_hist[action_type]])) / len(seq_hist[action_type])
        res = variance ** 0.5
        max_seq_len[action_type] = math.ceil(mean + (3*res))
    print(max_seq_len)
    for student in student_data:
        student_data[student] = [x for x in student_data[student] if x['data'].shape[0] <= max_seq_len[x['name']]]
        student_data[student] = [x for x in student_data[student] if (datetime.fromisoformat(x['time']) >= first_time and datetime.fromisoformat(x['time']) < last_time)] 

    clients = []
    
    for student in student_data:
        clients.append(User(student, student_data[student], max_seq_len))
        
    #print(str(first_time) + " " + str(last_time))
    #test = Autoencoder('test', (128,128,1), 2)
    #print(test.model.summary())
    num_aggregations = num_agg
    print("Start: {}".format(first_time))
    print("End: {}".format(last_time))
    period = (last_time - first_time) / num_aggregations
    print("Period: {}".format(period))
    #period = (last_time - first_time)/50
    histo = [0] * (num_aggregations+1)
    for student in student_data:
        for i in range(len(student_data[student])):
            event_time = datetime.fromisoformat(student_data[student][i]['time'])
            if (event_time <= last_time and event_time >= first_time):
                event_time = int((event_time - first_time)/period)
                #print(event_time)
                histo[event_time] += 1
    print("Distribution of events: ")
    print(histo)
    
    globe = Server(num_aggregations, clients)
    for client in globe.clients:
        client.set_Server(globe)
    end_models = globe.run_sim(first_time, last_time)
    column_names = ["model", "losses", "accs", "weights"]
    output = pd.DataFrame(columns = column_names)

    min_loss_len = [100,100,100]
    max_loss_len = [0,0,0]
    max_loss = [[],[],[]]
    min_loss = [[],[],[]]
    med_loss = [[],[],[]]
    models_list = []
    model_colors = []
    encoding_list = []

    #run_TSNE(models_list, model_colors, 'model_weights')
    plt.clf()
    #for encoder in clients[0].encoders:
    #    plt.plot(clients[0].encoders[encoder].losses_list)
    #plt.savefig('example_loss.png')
    labels = ['discussion post', 'video watching', 'course access']

    means = []
    var_s = []

    rep_cliet = None
    max_len = 0
    for client in clients:
        if (max_len < len(client.data)):
            max_len = len(client.data)
            rep_client = client

    plt.clf()
    plt.figure()
    plt.plot(rep_client.KL, label = 'KL')
    plt.plot(rep_client.recon, label = 'Reconstruction')
    plt.plot(rep_client.losses['contrast'], label = 'Contrast')
    plt.legend(loc='upper right')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.savefig('total_loss_courseB.png')

    rep_encodings = []
    rep_colors = []
    
    post_encodings = []
    click_encodings = []
    event_encodings = []
    assess_encodings = []
    all_encodings = []
    all_colors = []
    post_colors = []
    event_colors = []
    click_colors = []
    assess_colors = []
    transition_colors = []
    i = 0
    data_types = ['assess', 'click', 'post', 'event']
    transitions = ['r', 'purple', 'orange', 'pink', 'purple', 'b', 'g', 'gray', 'orange', 'g', 'y', 'olive', 'pink', 'gray', 'olive', 'black']
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'purple', 'black']
    formatted_data_by_type = {'click': [], 'post': [], 'event': [], 'assess': []}

    for client in end_models:
        formatted_data_by_type = dict()
        for item in client.data_types:
            formatted_data_by_type[item] = []
            #means.append(client.autoencoder.encoder.mean)
            #var_s.append(client.autoencoder.encoder.var)
            #print(client.autoencoder.var)
        for d in range(len(client.data)-1):
            data_point = client.data[d]
            data_type = data_point['name']
            next_data_type = client.data[d+1]['name']
            data_formatted = data_point['data']
            #data_formatted = np.expand_dims(data_formatted, axis=0)
            max_len = client.max_seq_lens[data_type]
            data_formatted = np.concatenate((data_formatted, np.zeros((max_len - data_formatted.shape[0], data_formatted.shape[1]))), axis = 0)
            data_formatted = np.expand_dims(data_formatted, axis=0)
            ind = data_types.index(data_type)
            next_ind = data_types.index(next_data_type)
            col = transitions[ind * len(data_types) + next_ind]
            history_int = client.data_heads[data_type].encoder.predict(data_formatted)
            history = client.autoencoder.encoder.predict(history_int)
            means.append(history[0][0])
            var_s.append(history[1][0])
            #print("Mean: {}".format(history[0]))
            #print("Var: {}".format(history[1]))
            #print(len(history))
            all_encodings.append(history[2])
            transition_colors.extend([col])
            data_shape = data_formatted.shape
            #formatted_data_by_type[data_type].append(data_formatted)

            if (data_type == 'post'):
                post_encodings.append(history[2])
                all_colors.extend(['y'])
                post_colors.extend([colors[i%10]])
            elif (data_type == 'click'):
                click_encodings.append(history[2])
                all_colors.extend(['m'])
                click_colors.extend([colors[i%10]])
            elif (data_type == 'assess'):
                assess_encodings.append(history[2])
                all_colors.extend(['g'])
                assess_colors.extend([colors[i%10]])
            else:
                event_encodings.append(history[2])
                all_colors.extend(['c'])
                event_colors.extend([colors[i%10]])
        i += 1

    print("Running TSNE")
    #plot_VAE(means, var_s)
    run_TSNE(all_encodings, all_colors, 'all data types')
    run_TSNE(all_encodings, transition_colors, 'all data by transition')
    run_TSNE(click_encodings, click_colors, 'clickstream data')
    run_TSNE(post_encodings, post_colors, 'discussion post data')
    run_TSNE(event_encodings, event_colors, 'course access data')
    #print(means)
    #print(means[0])
    #plt_VAE(means, var_s)
    
    data_list = [click_encodings, post_encodings, event_encodings, rep_encodings]
    colors = [click_colors, post_colors, event_colors, rep_colors]
    labels = ['clickstream data', 'discussion post data', 'course access data', 'representative student']
    for i in range(len(data_list)):
        X = np.array(data_list[i])
        Y= pca.transform(X)
        plt.clf()
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], c=colors[i])
        plt.title("PCA by {} (All Encodings Model)".format(labels[i]))
        plt.savefig("pca_{}_perfed_contrast_subset.png".format(labels[i]))
    
    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
