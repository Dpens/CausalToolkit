import numpy as np
from .utils import *
from ..BaseModel import BaseModel
import pandas as pd
import networkx as nx
from tqdm import tqdm

class ADM4(BaseModel):

    def __init__(self, device, num_variable, graph_threshold=0.5, decay=3, max_iter=5, em_max_iter=50, rho=.1, threshold=.01, alpha=0.1, sparse=True, low_rank= False):
        super(ADM4, self).__init__(device=device)
        self.Z1 = np.zeros_like((num_variable, num_variable))
        self.U1 = np.zeros_like((num_variable, num_variable))
        self.mu = np.ones(num_variable)
        self.A = np.random.uniform(0.5, 0.9, (num_variable, num_variable))
        self.threshold = threshold
        self.graph_threshold = graph_threshold
        self.decay = decay
        self.max_iter = max_iter
        self.em_max_iter = em_max_iter
        self.num_variable = num_variable
        self.rho = rho
        self.sparse = sparse
        self.low_rank = low_rank
        self.alpha = alpha

    def fit(self, event_table: pd.DataFrame, columns):
        ids = [i for i in range(self.num_variable)]
        event_table = event_table.replace(columns, ids)
        time_seq = event_table["time_stamp"].to_numpy()
        mark_seq = event_table["event_type"].to_numpy()
        
        max_iter = self.max_iter
        em_max_iter = self.em_max_iter
        decay = self.decay
        num_variable = self.num_variable
        sparse = self.sparse
        rho = self.rho
        threshold = self.threshold
        alpha = self.alpha

        T_start = time_seq[0]
        T_stop = time_seq[-1] + 1

        dT = T_stop - time_seq
        GK = kernel_integration(dT, decay)

        Aest = self.A.copy()
        muest = self.mu.copy()

        gij = [np.array([0] * num_variable)]
        BmatA_raw = np.zeros_like(Aest)
        for i in range(len(time_seq)):
            time = time_seq[i]
            mark = mark_seq[i]
            BmatA_raw[:, mark] = BmatA_raw[:, mark] + [GK[i]] * num_variable
            if i > 0:
                tj = time_seq[i-1]
                uj = mark_seq[i-1]
                gij_last = gij[-1].copy()

                gij_now = (gij_last / decay) * decay * np.exp(-decay * (time - tj))
                gij_now[uj] += decay * np.exp(-decay*(time - tj))

                gij.append(gij_now)
        gij_raw = np.array(gij)

        if sparse:
            US = np.zeros_like(Aest)
            ZS = Aest.copy()

        for o in tqdm(range(max_iter)):
            rho = rho * (1.1 ** o)
            for n in range(em_max_iter):
                Amu = np.zeros_like(muest)
                Bmu = Amu.copy()

                CmatA = np.zeros_like(Aest)
                BmatA = BmatA_raw.copy()

                if sparse:
                    BmatA = BmatA + rho * (US - ZS)

                Amu = Amu + T_stop - T_start
                gij = Aest[mark_seq] * gij_raw
                gij = np.insert(gij, 0, values=muest[mark_seq], axis=1)
                pij = gij / np.sum(gij, axis=1)[:, np.newaxis]
                np.add.at(Bmu, mark_seq, pij[:, 0])
                np.add.at(CmatA, mark_seq, -pij[:, 1:])

                mu = Bmu / Amu
                if sparse:
                    A = (-BmatA + np.sqrt(np.square(BmatA) - 8 * rho * CmatA))/(4 * rho)
                else:
                    A = -CmatA / BmatA

                Err = np.sum(abs(A-Aest)) / np.sum(abs(Aest))
                Aest = A.copy()
                muest = mu.copy()
                self.A = Aest.copy()
                self.mu = muest.copy()

                if Err < threshold:
                    break

            if sparse:
                threshold = alpha / rho
                ZS = soft_thres_S(Aest+US, threshold)
                US = US + Aest - ZS
        DAG = np.zeros(shape=self.A.shape, dtype=int)
        DAG[np.abs(self.A) > self.graph_threshold] = 1
        # pred_G = nx.DiGraph(self.A)
        # self.DAG = pred_G
        for i in range(self.A.shape[0]):
            DAG[i, i] = 0
        self.DAG = DAG