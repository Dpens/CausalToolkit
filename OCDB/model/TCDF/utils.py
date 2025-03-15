import argparse
import pandas as pd
import networkx as nx 
import copy
import torch
from torch.autograd import Variable

def check_positive(value):
    """Checks if argument is positive integer (larger than zero)."""
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

def check_zero_or_positive(value):
    """Checks if argument is positive integer (larger than or equal to zero)."""
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

class StoreDictKeyPair(argparse.Action):
    """Creates dictionary containing datasets as keys and ground truth files as values."""
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def getextendeddelays(gtfile, columns):
    """Collects the total delay of indirect causal relationships."""
    gtdata = pd.read_csv(gtfile, header=None)

    readgt=dict()
    effects = gtdata[1]
    causes = gtdata[0]
    delays = gtdata[2]
    gtnrrelations = 0
    pairdelays = dict()
    for k in range(len(columns)):
        readgt[k]=[]
    for i in range(len(effects)):
        key=effects[i]
        value=causes[i]
        readgt[key].append(value)
        pairdelays[(key, value)]=delays[i]
        gtnrrelations+=1
    
    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    extendedreadgt = copy.deepcopy(readgt)
    
    for c1 in range(len(columns)):
        for c2 in range(len(columns)):
            paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) #indirect path max length 3, no cycles
            
            if len(paths)>0:
                for path in paths:
                    for p in path[:-1]:
                        if p not in extendedreadgt[path[-1]]:
                            extendedreadgt[path[-1]].append(p)
                            
    extendedgtdelays = dict()
    for effect in extendedreadgt:
        causes = extendedreadgt[effect]
        for cause in causes:
            if (effect, cause) in pairdelays:
                delay = pairdelays[(effect, cause)]
                extendedgtdelays[(effect, cause)]=[delay]
            else:
                #find extended delay
                paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2)) #indirect path max length 3, no cycles
                extendedgtdelays[(effect, cause)]=[]
                for p in paths:
                    delay=0
                    for i in range(len(p)-1):
                        delay+=pairdelays[(p[i+1], p[i])]
                    extendedgtdelays[(effect, cause)].append(delay)

    return extendedgtdelays, readgt, extendedreadgt

def evaluate(gtfile, validatedcauses, columns):
    """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
    FP=0
    FPdirect=0
    TPdirect=0
    TP=0
    FN=0
    FPs = []
    FPsdirect = []
    TPsdirect = []
    TPs = []
    FNs = []
    for key in readgt:
        for v in validatedcauses[key]:
            if v not in extendedreadgt[key]:
                FP+=1
                FPs.append((key,v))
            else:
                TP+=1
                TPs.append((key,v))
            if v not in readgt[key]:
                FPdirect+=1
                FPsdirect.append((key,v))
            else:
                TPdirect+=1
                TPsdirect.append((key,v))
        for v in readgt[key]:
            if v not in validatedcauses[key]:
                FN+=1
                FNs.append((key, v))
          
    print("Total False Positives': ", FP)
    print("Total True Positives': ", TP)
    print("Total False Negatives: ", FN)
    print("Total Direct False Positives: ", FPdirect)
    print("Total Direct True Positives: ", TPdirect)
    print("TPs': ", TPs)
    print("FPs': ", FPs)
    print("TPs direct: ", TPsdirect)
    print("FPs direct: ", FPsdirect)
    print("FNs: ", FNs)
    precision = recall = 0.

    if float(TP+FP)>0:
        precision = TP / float(TP+FP)
    print("Precision': ", precision)
    if float(TP + FN)>0:
        recall = TP / float(TP + FN)
    print("Recall': ", recall)
    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0.
    print("F1' score: ", F1,"(includes direct and indirect causal relationships)")

    precision = recall = 0.
    if float(TPdirect+FPdirect)>0:
        precision = TPdirect / float(TPdirect+FPdirect)
    print("Precision: ", precision)
    if float(TPdirect + FN)>0:
        recall = TPdirect / float(TPdirect + FN)
    print("Recall: ", recall)
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    print("F1 score: ", F1direct,"(includes only direct causal relationships)")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct

def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp=TPs[i]
        discovereddelay = alldelays[tp]
        gtdelays = extendedgtdelays[tp]
        for d in gtdelays:
            if d <= receptivefield:
                total+=1.
                error = d - discovereddelay
                if error == 0:
                    zeros+=1
                
            else:
                next
           
    if zeros==0:
        return 0.
    else:
        return zeros/float(total)
    

def preparedata(df_data, target):
    """Reads data from csv file and transforms it to two PyTorch tensors: dataset x and target time series y that has to be predicted."""
    # df_data = pd.read_csv(file)
    df_y = df_data.copy(deep=True)[[target]]
    df_x = df_data.copy(deep=True)
    df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
    df_yshift[target]=df_yshift[target].fillna(0.)
    df_x[target] = df_yshift
    data_x = df_x.values.astype('float32').transpose()    
    data_y = df_y.values.astype('float32').transpose()
    data_x = torch.from_numpy(data_x).double()
    data_y = torch.from_numpy(data_y).double()
    x, y = Variable(data_x), Variable(data_y)
    return x, y