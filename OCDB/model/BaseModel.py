import torch.nn as nn
import numpy as np
from ..metrics.Metrics import *


class BaseModel(nn.Module):
    def __init__(self, device):
        super(BaseModel, self).__init__()
        self.device = device
        self.DAG = None

    def fit(self, feature):
        # Please assign the learned DAG to self.DAG at the end.
        # self.DAG = learned_DAG
        pass

    def save_DAG(self, save_path):
        with open(save_path, 'w') as f:
            matG1 = np.matrix(self.DAG.data.clone().numpy())
            for line in matG1:
                np.savetxt(f, line, fmt='%.5f')

    def get_DAG(self):
        return self.DAG
    
    def _eval(self, ground_trues, metric="CSD"):
        if metric == "CSD":
            return CSD(ground_trues, self.DAG)
        elif metric == "SID":
            return SID(ground_trues, self.DAG)
        elif metric == "SHD-C":
            return SHD_CPDAG(ground_trues, self.DAG)
        elif metric == "CED":
            return CED(ground_trues, self.DAG, k=ground_trues.shape[0])
        else:
            return f1_score(ground_trues, self.DAG)
