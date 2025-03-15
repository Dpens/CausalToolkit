import torch
import numpy as np
from ..BaseModel import BaseModel
from torch import nn
from .utils import *
from .module import timeseries
from torch.utils.data import DataLoader
import networkx as nx
torch.set_default_dtype(torch.float64)


class NTiCD(BaseModel):
    def __init__(self, device, num_variable, input_size, output_size, hidden_dim=128, n_layers=5):
        super(NTiCD, self).__init__(device)

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.num_variable = num_variable
        adj = torch.Tensor(1*np.zeros([num_variable,num_variable], dtype=np.double) + np.random.uniform(low=0, high=0.1, size=(num_variable,num_variable)))
        self.A = nn.Parameter(adj, requires_grad = True)  # Adjacency matrix as parameter to update during training
        self.gcn_lin_dim = hidden_dim

        #Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, self.hidden_dim, num_layers=n_layers, batch_first=True)  

        #GCN
        self.gcn_lin1 = nn.Sequential(
          nn.Linear(self.hidden_dim, self.gcn_lin_dim, bias=False))

        self.gcn_lin2 = nn.Sequential(
          nn.Linear(self.hidden_dim, self.gcn_lin_dim, bias=False))

        self.gcn_lin3 = nn.Sequential(
          nn.Linear(self.gcn_lin_dim, self.gcn_lin_dim, bias=False))
        
        self.relu = torch.nn.ReLU(inplace=False)  # element-wise

        # MLP layers
        self.fc = nn.Sequential(
          nn.Linear(self.gcn_lin_dim, output_size, bias=False),
          nn.Sigmoid())

        # print("\nInitial A:\n")
        # print(self.A.data)

    def forward(self, x):         
        batch_size = x.size(0)
        # LSTM
        h_LSTM = torch.empty((batch_size, self.hidden_dim, self.num_variable)).to(self.device)
        self.hidden = self.init_hidden(batch_size)
        for j in range(self.num_variable):
            #print('x input to LSTM: ', x[:,j,:,:].shape)
            lstm_out, (h_n,c_n)= self.lstm(x[:,j,:,:], self.hidden)
            #print('h_n shape inside forward: ', h_n.shape)
            h_LSTM[:, :, j] = torch.squeeze(h_n)[-1]  #shape = [batch, hidden_dim, num_variable]
        #print('after reshaping: ', h_LSTM.shape)

        # GAT
        self.A_prime = torch.sigmoid(self.A)
        H_times_A= torch.einsum('ikj,jl->ikl', h_LSTM, self.A_prime)   #shape = [batch, hidden_dim, num_variable]
        
        #GCN
        alpha = 0.9   #parameter for balance of self-information

        h_A = H_times_A.permute(0, 2, 1)  #shape = [batch, num_variable, hidden_dim]
        W1_h_A = self.gcn_lin1(h_A) #shape = [batch, num_variable, self.gcn_lin_dim]
        W1_h_A = W1_h_A.permute(0, 2, 1)   #shape = [batch, self.gcn_lin_dim, num_variable]   
        ################### updated GCN with alpha
        W2_h_A = self.gcn_lin2(h_A) #shape = [batch, num_variable, self.gcn_lin_dim]
        Relu_W2_h_A = self.relu(W2_h_A)
        W3_relu_HAW1 = self.gcn_lin3(Relu_W2_h_A) #shape = [batch, num_variable, self.gcn_lin_dim]
        W3_relu_HAW1 = W3_relu_HAW1.permute(0, 2, 1)  #shape = [batch, self.gcn_lin_dim, num_variable]
        AT_times_W3h= torch.einsum('ikj,jl->ikl', W3_relu_HAW1, self.A_prime)       #shape = [batch, self.gcn_lin_dim, num_variable]  

        h_out = (1-alpha) * AT_times_W3h + alpha * W1_h_A   #shape = [batch, num_variable, self.gcn_lin_dim]  # weighter summation version  

        # MLP
        out_MLP = torch.empty((batch_size, self.num_variable, self.output_size)).to(self.device)
        for k in range(self.num_variable):
            ma = h_out[:, :, k]
            out_MLP[:,k,:] = self.fc(ma)
        return (out_MLP)

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(self.device),
                            torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(self.device))
        return hidden
    
    def fit(self, feature, epochs=50, lr=1e-3, batch_size=128, window_size=5, regularization_param=2e-3, num_sequence=1):
        print('\nTraining started...\n')

        # Input-data
        data = feature
        # the loaded data is 2D, so need to convert to 3D
        data = data.reshape(data.shape[-1], -1, num_sequence)

        d = np.shape(data)[0]  # number of variables
        n = np.shape(data)[1]  # number of time-steps

        
        x_train, y_train = preprocess(data,d,n,window_size)
        dataset = timeseries(x_train,y_train, self.device)
        train_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size, drop_last=True)

        # define the model
        
        self.apply(init_weights)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # keep log using wandb
        #wandb.init(project='New Simultaneous Training')
        #wandb.watch(model, log='all')

        criterion = torch.nn.MSELoss()
        L = []

        # train to update A
        for epoch in range(epochs):
            total_loss = 0
            e = 0
            self.train()
            for i, (in_seq, target) in enumerate(train_loader):
                optimizer.zero_grad()

                out = self(in_seq) # returns the output from final MLP layer

                # Calculate error
                mse = criterion(out, target)
                l1_norm = torch.norm(self.A_prime)
                loss2 = regularization_param * l1_norm
                l3 =  regularization_param * torch.sum(torch.square(self.A_prime-torch.mean(self.A_prime,0)))
                loss = mse + loss2 - l3
                loss.backward(retain_graph=True)    # Does backpropagation and calculates gradients

                # Updates the weights accordingly
                optimizer.step() 

                # to keep a record of total loss per epoch
                total_loss += loss.item()
                e += 1

            avg_loss = total_loss/e

            if epoch%1 == 0:
                print('MSE: ', mse.item())
                print('l2_norm_groupwise: ', l1_norm.item())
                print('l3: ', l3.item())
                print('Epoch: {} .............'.format(epoch), end=' ')
                print("Loss: {:.4f}".format(avg_loss))
            L.append(avg_loss)

        A_pred = self.A_prime.detach().cpu().numpy()
        graph_thres = np.mean(A_pred,0) # columnwise mean to calculate different threshold for each variable
        A_pred[np.abs(A_pred) < graph_thres] = 0    
        A_pred[np.abs(A_pred) >= graph_thres] = 1
        for i in range(A_pred.shape[0]):
            A_pred[i, i] = 0
        self.DAG = A_pred

