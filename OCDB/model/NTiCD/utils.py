from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import torch.nn as nn


def preprocess(data,d,n,window_size):
    training_data = []
    y = []
    for var in range(d):
        # Normalize data column-wise
        sc = MinMaxScaler(feature_range=(0, 1))
        temp = sc.fit_transform(data[var,:,:]) #returns a 2D array
        # structuring the data 
        target = [] 
        in_seq = []
        for i in range(n-window_size-1):
            list1 = []
            for j in range(i,i+window_size):
                list1.append(temp[j])
            in_seq.append(list1)
            target.append(temp[j+1])
        #print(np.array(in_seq).shape)
        #print(np.array(target).shape)
        training_data.append(in_seq)
        y.append(target)
    # Permute the batch at axis=0 for Dataloader
    training_data = torch.tensor(np.array(training_data),dtype=torch.float64)
    y = torch.tensor(np.array(y),dtype=torch.float64)
    training_data = torch.permute(training_data,(1,0,2,3)) 
    y = torch.permute(y,(1,0,2)) 
    #print(training_data.shape)
    #print(y.shape)
    return (training_data, y)

def init_weights(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)

### Training the 3 models ###
