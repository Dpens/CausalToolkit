import torch
import numpy as np
from tqdm import tqdm
from .utils import check_DAG
from ..BaseModel import BaseModel
from itertools import product
import pandas as pd
import networkx as nx

__MIN__ = -np.inf


class SHP(BaseModel):
    def __init__(self, device, num_variable, decay, time_interval=None,
                 penalty='BIC', seed=None, reg=3.0):
        '''
        :param event_table: A pandas.DataFrame of events with columns  ['seq_id', 'time_stamp', 'event_type']
        :param decay: The decay used in the exponential kernel
        :param init_structure: adj of causal structure of prior knowledge
        :param penalty: 'BIC' or 'AIC' penalty
        '''
        super(SHP, self).__init__(device=device)
        self.random_state = np.random.RandomState(seed)
        self.reg = reg
        self.time_interval = time_interval  # Delta t
        self.decay = decay
         # Initializing Structs
        self.init_structure = np.zeros([num_variable, num_variable])
        self.penalty = penalty
        self.num_variable = num_variable
        
    def fit(self, event_table: pd.DataFrame, columns, threshold, hill_climb=True):
        ids = [i for i in range(self.num_variable)]
        event_table = event_table.replace(columns, ids)
        event_table.insert(loc=0, column="seq_id", value=[0 for i in range(len(event_table))])
        self.event_table, self.event_names = self.get_event_table(event_table)
        for seq_id in np.unique(self.event_table["seq_id"].values):
            seq_index = self.event_table[self.event_table["seq_id"] == seq_id].index.tolist()
            self.event_table.loc[seq_index, "max_time_stamp"] = \
                self.event_table.loc[seq_index]["max_time_stamp"] - self.event_table.loc[seq_index]["time_stamp"].min()
            self.event_table.loc[seq_index, "time_stamp"] = \
                self.event_table.loc[seq_index]["time_stamp"] - self.event_table.loc[seq_index]["time_stamp"].min()

        # store the calculated likelihood
        self.hist_likelihood = dict()
        for i in range(len(self.event_names)):
            self.hist_likelihood[i] = dict()

        if self.penalty not in {'BIC', 'AIC'}:
            raise Exception('Penalty is not supported')

        self.n = len(self.event_names)  # num of event type
        self.T = self.event_table.groupby('seq_id').apply(lambda i: (i['time_stamp'].max())).sum()  # total time span
        self.T_each_seq = self.event_table.groupby('seq_id'). \
            apply(lambda i: (i['time_stamp'].max()))  # the last moment of each event sequence
        
        X_dict = dict()
        for seq_id, time_stamp, _, times, type_ind, _ in self.event_table.values:
            if (seq_id, time_stamp) not in X_dict:
                X_dict[(seq_id, time_stamp)] = [0] * self.n
            X_dict[(seq_id, time_stamp)][type_ind] = times
        self.X_df = pd.DataFrame(X_dict).T

        self.X = self.X_df.values
        self.sum_t_X_kappa, self.decay_effect_integral_to_T = self.calculate_influence_of_each_event()
        if hill_climb:
            linklihood, alpha, mu = self.Hill_Climb()
        else:
            linklihood, alpha, mu = self.EM_not_HC(np.ones([self.num_variable, self.num_variable]) - np.eye(self.num_variable, self.num_variable))
        print(np.min(np.abs(alpha)), np.max(np.abs(alpha)))
        alpha[np.abs(alpha) > threshold] = 1
        alpha[np.abs(alpha) <= threshold] = 0
        self.DAG = alpha

    # calculate the influence of each event
    def calculate_influence_of_each_event(self):
        sum_t_X_kappa = np.zeros_like(self.X, dtype='float64')
        decay_effect_integral_to_T = self.X_df.copy()

        for ind, (seq_id, time_stamp) in tqdm(enumerate(self.X_df.index)):
            # calculate the integral of decay function on time
            decay_effect_integral_to_T.iloc[ind] = \
                self.X_df.iloc[ind] * ((1 - np.exp(-self.decay * (self.T_each_seq[seq_id] - time_stamp))) / self.decay)

            start_ind = ind
            start_seq_id, start_time_stamp = self.X_df.index[start_ind]
            # the influence on subsequent timestamp when the event occurs
            next_ind = start_ind
            while start_seq_id == seq_id:  # the influence only spread on the same sequence
                kap = self.kappa(start_time_stamp - time_stamp)
                if kap < 0.0001:
                    break

                X_kappa = self.X[ind] * kap
                sum_t_X_kappa[next_ind] += X_kappa  # record the influence
                next_ind += 1
                if next_ind >= len(self.X):
                    break
                start_seq_id, start_time_stamp = self.X_df.index[next_ind]
        return sum_t_X_kappa, decay_effect_integral_to_T

    # decay function
    def kappa(self, t):
        y = np.exp(-self.decay * t)
        return y

    # transfer event table from continuous time domain to the discrete time domain
    def get_event_table(self, event_table: pd.DataFrame):
        event_table = event_table.copy()
        event_table.columns = ['seq_id', 'event_type', 'time_stamp']
        if self.time_interval is not None:
            event_table['time_stamp'] = (event_table['time_stamp'] / self.time_interval).astype(
                'int') * self.time_interval
        
        event_table['times'] = np.zeros(len(event_table))
        event_table = event_table.groupby(['seq_id', 'time_stamp', 'event_type']).count().reset_index()

        event_ind = event_table['event_type'].astype('category')
        event_table['type_ind'] = event_ind.cat.codes
        event_names = event_ind.cat.categories

        max_time = event_table.groupby('seq_id').apply(lambda i: i['time_stamp'].max())
        event_table = pd.merge(event_table, pd.DataFrame(max_time, columns=['max_time_stamp']).reset_index())

        event_table.sort_values(['seq_id', 'time_stamp', 'type_ind'])
        return event_table, event_names

    # EM module
    def EM(self, edge_mat):
        '''
        :param edge_mat:    Adjacency matrix
        :return:            Return (likelihood, alpha matrix, mu vector)
        '''
        if not check_DAG(edge_mat):
            return __MIN__, np.zeros([len(self.event_names), len(self.event_names)]), np.zeros(
                len(self.event_names))

        alpha = self.random_state.uniform(0, 1, [len(self.event_names), len(self.event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self.event_names))
        L = 0

        # calculate the likelihood for each event type i
        for i in (range(len(self.event_names))):
            Pa_i = tuple(np.where(edge_mat[:, i] == 1)[0])
            try:
                Li = self.hist_likelihood[i][tuple(Pa_i)][0]
                mu[i] = self.hist_likelihood[i][tuple(Pa_i)][2]
                for j in Pa_i:
                    alpha[j, i] = self.hist_likelihood[i][tuple(Pa_i)][1][j]
                L += Li
            except Exception as e:
                Li = __MIN__

                while 1:
                    # the first term of likelihood function
                    lambda_for_i = (self.sum_t_X_kappa * alpha[:, i]).sum(1) + mu[i]
                    # the second term of likelihood function
                    X_log_lambda = (self.X[:, i] * np.log(lambda_for_i)).sum()
                    lambda_i_sum = (((1 / self.decay) * self.X).sum(0) * alpha[:, i].T).sum() + mu[i] * self.T

                    # calculate the likelihood
                    new_Li = -lambda_i_sum + X_log_lambda

                    # Iteration termination condition
                    gain = new_Li - Li
                    if gain < 0.0085:
                        Li = new_Li
                        L += Li
                        Pa_i_alpha = dict()
                        for j in Pa_i:
                            Pa_i_alpha[j] = alpha[j, i]
                        self.hist_likelihood[i][tuple(Pa_i)] = (Li, Pa_i_alpha, mu[i])
                        break
                    Li = new_Li

                    # update mu
                    mu[i] = ((mu[i] / lambda_for_i) * self.X[:, i]).sum() / self.T
                    # update alpha
                    for j in Pa_i:
                        q_alpha = alpha[j, i] * self.sum_t_X_kappa[:, j] / lambda_for_i
                        upper = (q_alpha * self.X[:, i]).sum()
                        lower = self.decay_effect_integral_to_T.sum(0)[j] * self.time_interval
                        if lower == 0:
                            alpha[j, i] = 0
                            continue
                        alpha[j, i] = upper / lower

                i += 1
        if self.penalty == 'AIC':
            return L - (len(self.event_names) + edge_mat.sum()), alpha, mu
        if self.penalty == 'BIC':
            return L - (len(self.event_names) + edge_mat.sum()) * np.log(
                self.event_table['times'].sum()) * self.reg, alpha, mu

    # EM module without using hill climb
    def EM_not_HC(self, edge_mat):
        '''
        :param edge_mat:    Adjacency matrix
        :return:            Return (likelihood, alpha matrix, mu vector)
        '''

        alpha = self.random_state.uniform(0, 1, [len(self.event_names), len(self.event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self.event_names))
        L = 0

        for i in (range(len(self.event_names))):
            Pa_i = set(np.where(edge_mat[:, i] == 1)[0])

            try:
                Li = self.hist_likelihood[i][tuple(Pa_i)][0]
                mu[i] = self.hist_likelihood[i][tuple(Pa_i)][2]
                for j in Pa_i:
                    alpha[j, i] = self.hist_likelihood[i][tuple(Pa_i)][1][j]
                L += Li
            except Exception as e:
                Li = __MIN__

                while 1:
                    # the first term of likelihood function
                    lambda_for_i = (self.sum_t_X_kappa * alpha[:, i]).sum(1) + mu[i]
                    # the second term of likelihood function
                    X_log_lambda = (self.X[:, i] * np.log(lambda_for_i)).sum()
                    lambda_i_sum = (((1 / self.decay) * self.X).sum(0) * alpha[:, i].T).sum() + mu[i] * self.T

                    new_Li = -lambda_i_sum + X_log_lambda
                    # Iteration termination condition
                    gain = new_Li - Li

                    if gain <= 0.0085:
                        Li = new_Li
                        L += Li
                        Pa_i_alpha = dict()
                        for j in Pa_i:
                            Pa_i_alpha[j] = alpha[j, i]
                        self.hist_likelihood[i][tuple(Pa_i)] = (Li, Pa_i_alpha, mu[i])
                        break
                    Li = new_Li

                    # update mu
                    mu[i] = ((mu[i] / lambda_for_i) * self.X[:, i]).sum() / self.T
                    # update alpha
                    for j in Pa_i:
                        q_alpha = alpha[j, i] * self.sum_t_X_kappa[:, j] / lambda_for_i
                        upper = (q_alpha * self.X[:, i]).sum()
                        lower = self.decay_effect_integral_to_T.sum(0)[j] * self.time_interval
                        if lower == 0:
                            alpha[j, i] = 0
                            continue
                        alpha[j, i] = upper / lower

                i += 1
        if self.penalty == 'AIC':
            return L - (len(self.event_names) + edge_mat.sum())* self.reg, alpha, mu
        if self.penalty == 'BIC':
            return L - (len(self.event_names) + edge_mat.sum()) * np.log(
                self.event_table['times'].sum()) * self.reg, alpha, mu

    # the searching module for new edges
    def one_step_change_iterator(self, edge_mat):
        return map(lambda e: self.one_step_change(edge_mat, e),
                   product(range(len(self.event_names)),
                           range(len(self.event_names)), range(3)))

    def one_step_change(self, edge_mat, e):
        j, i = e[0], e[1]
        if j == i:
            return edge_mat
        new_graph = edge_mat.copy()
        if e[2] == 0:
            new_graph[j, i] = 0
            new_graph[i, j] = 0
            return new_graph
        elif e[2] == 1:
            new_graph[j, i] = 1
            new_graph[i, j] = 0
            return new_graph
        else:
            new_graph[j, i] = 0
            new_graph[i, j] = 1
            return new_graph

    def Hill_Climb(self):
        # Initialize the adjacency matrix
        edge_mat = self.init_structure
        result = self.EM(edge_mat)

        L = result[0]
        while 1:
            stop_tag = True
            for new_edge_mat in tqdm(list(self.one_step_change_iterator(edge_mat)), mininterval=5):
                new_result = self.EM(new_edge_mat)
                new_L = new_result[0]
                # Termination condition: no adjacency matrix with higher likelihood appears
                if new_L > L:
                    result = new_result
                    L = new_L
                    # if there is a new edge can be added, then set the stop_tag=False and continue searching
                    stop_tag = False
                    edge_mat = new_edge_mat

            if stop_tag:
                return result