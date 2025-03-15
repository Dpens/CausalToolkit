import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
from .utils import *

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .module import DepthwiseNet
import numpy as np
import torch.optim as _optim
import random
import heapq
from ..BaseModel import BaseModel


class TCDF(BaseModel):
    def __init__(self, device, num_variable, input_size, hidden_layers=2, kernel_size=256, dilation_c=4, optim="Adam", threshold=0.5):
        super(TCDF, self).__init__(device)
        # self.device = device

        self.optim = optim
        self.hidden_layers = hidden_layers
        self.dilation_c = dilation_c
        self.threshold = threshold
        self.dwns = nn.ModuleList([])
        for i in range(num_variable):
            self.dwns.append(DepthwiseNet(str(i), input_size, hidden_layers, kernel_size=kernel_size, dilation_c=dilation_c))
        
        self.pointwises = nn.ModuleList([])
        for i in range(num_variable):
            self.pointwises.append(nn.Conv1d(input_size, 1, 1))

        # self.fs_attentions = []
        # for i in range(num_variable):
        #     _attention = torch.ones(size=(input_size, 1)).to(device)
        #     fs_attention = torch.nn.Parameter(_attention, requires_grad=True)
        #     self.fs_attentions.append(fs_attention)
        self.fs_attentions = nn.ParameterList([torch.nn.Parameter(torch.ones(size=(input_size, 1)).to(device), requires_grad=True) for _ in range(num_variable)])
                  
    def init_weights(self):
        for pointwise in self.pointwises:
            pointwise.weight.data.normal_(0, 0.1)
        
    def forward(self, x, idx):
        y1 = self.dwns[idx](x * F.softmax(self.fs_attentions[idx], dim=0))
        y2 = self.pointwises[idx](y1)
        return y2.transpose(1,2)
        
        # self.dwn = self.dwn.to(device)
        # self.pointwise = self.pointwise.to(device)
        # self._attention = self._attention.to(device)
                  
    def init_weights(self):
        for pointwise in self.pointwises:
            pointwise.weight.data.normal_(0, 0.1)
        
    def forward(self, x, idx):
        y1 = self.dwns[idx](x*torch.softmax(self.fs_attentions[idx], dim=0))
        # y1 = self.dwns[idx](x * self.fs_attentions[idx])
        y2 = self.pointwises[idx](y1) 
        return y2.transpose(1,2)
    
    def fit(self, feature, epochs=10, log_interval=1, lr=0.001, seed=42, significance=0.5):
        df_data = pd.DataFrame(feature)
        allcauses = dict()
        alldelays = dict()
        allreallosses=dict()
        allscores=dict()

        columns = list(df_data)
        for c in columns:
            idx = df_data.columns.get_loc(c)
            causes, causeswithdelay, realloss, scores = self.findcauses(c, epochs=epochs, 
            log_interval=log_interval, lr=lr, seed=seed, 
            significance=significance, df_data=df_data)
            allscores[idx]=scores
            allcauses[idx]=causes
            alldelays.update(causeswithdelay)
            allreallosses[idx]=realloss

        G = nx.DiGraph()
        for c in columns:
            G.add_node(c)
        for pair in alldelays:
            p1,p2 = pair
            nodepair = (columns[p2], columns[p1])

            G.add_edges_from([nodepair],weight=alldelays[pair])
        
        DAG = np.array(nx.adjacency_matrix(G).todense())
        _DAG = np.zeros(shape=DAG.shape, dtype=int)
        _DAG[np.abs(DAG) > 0] = 1
        self.DAG = _DAG

    def _train(self, epoch, X, Y, optimizer, log_interval, epochs, targetidx):
        self.train()
        x, y = X[0:1], Y[0:1]
        optimizer.zero_grad()
        epochpercentage = (epoch/float(epochs))*100
        output = self(x, targetidx)
    
        attentionscores = self.fs_attentions[targetidx]
        # print("first", attentionscores.data)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
        # print("second", attentionscores.data)
        if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
            print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))
    
        return attentionscores.data, loss

    def findcauses(self, target, epochs, 
               log_interval, lr, seed, significance, df_data):
        """Discovers potential causes of one target time series, validates these potential causes with PIVM and discovers the corresponding time delays"""

        print("\n", "Analysis started for target: ", target)
        torch.manual_seed(seed)

        X_train, Y_train = preparedata(df_data, target)
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()

        targetidx = df_data.columns.get_loc(target)

        
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        optimizer = getattr(_optim, self.optim)(self.parameters(), lr=lr)    

        scores, firstloss = self._train(1, X_train, Y_train, optimizer,log_interval,epochs, targetidx)
        firstloss = firstloss.detach().cpu().data.item()
        for ep in range(2, epochs+1):
            scores, realloss = self._train(ep, X_train, Y_train, optimizer,log_interval,epochs, targetidx)
        realloss = realloss.detach().cpu().data.item()

        s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
        print(s)
        indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
        #attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
        if len(s)<=5:
            potentials = []
            for i in indices:
                if scores[i]> self.threshold:
                    potentials.append(i)
        else:
            potentials = []
            gaps = []
            for i in range(len(s)-1):
                if s[i] < self.threshold: #tau should be greater or equal to 1, so only consider scores >= 1
                    break
                gap = s[i]-s[i+1]
                gaps.append(gap)
            sortgaps = sorted(gaps, reverse=True)

            ind = -1
            for i in range(0, len(gaps)):
                largestgap = sortgaps[i]
                index = gaps.index(largestgap)
                ind = -1
                if index<((len(s)-1)/2): #gap should be in first half
                    if index>0:
                        ind=index #gap should have index > 0, except if second score <1
                        break
            if ind<0:
                ind = 0

            potentials = indices[:ind+1].tolist()
        print("Potential causes: ", potentials)
        validated = copy.deepcopy(potentials)

        #Apply PIVM (permutes the values) to check if potential cause is true cause
        for idx in potentials:
            random.seed(seed)
            X_test2 = X_train.clone().cpu().numpy()
            random.shuffle(X_test2[:,idx,:][0])
            shuffled = torch.from_numpy(X_test2)
            shuffled=shuffled.to(self.device)
            self.eval()
            output = self(shuffled, targetidx)
            testloss = F.mse_loss(output, Y_train)
            testloss = testloss.cpu().data.item()

            diff = firstloss-realloss
            testdiff = firstloss-testloss
            print(testdiff, diff)
            if testdiff>(diff*significance): 
                validated.remove(idx) 

        weights = []

        #Discover time delay between cause and effect by interpreting kernel weights
        for layer in range(self.hidden_layers):
            weight = self.dwns[targetidx].network[layer].net[0].weight.abs().view(self.dwns[targetidx].network[layer].net[0].weight.size()[0], self.dwns[targetidx].network[layer].net[0].weight.size()[2])
            weights.append(weight)

        causeswithdelay = dict()    
        for v in validated: 
            totaldelay=0    
            for k in range(len(weights)):
                w=weights[k]
                row = w[v]
                twolargest = heapq.nlargest(2, row)
                m = twolargest[0]
                m2 = twolargest[1]
                if m > m2:
                    index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                else:
                    #take first filter
                    index_max=0
                delay = index_max *(self.dilation_c**k)
                totaldelay+=delay
            if targetidx != v:
                causeswithdelay[(targetidx, v)]=totaldelay
            else:
                causeswithdelay[(targetidx, v)]=totaldelay+1
        print("Validated causes: ", validated)

        return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()



