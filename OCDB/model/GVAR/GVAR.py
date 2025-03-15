import numpy as np
import torch.nn as nn
import torch
from ..BaseModel import BaseModel
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
import networkx as nx
import random
from .utils import construct_training_dataset


class GVAR(BaseModel):
    def __init__(self, device: torch.device, num_variable: int, order=5, hidden_layer_size=50, num_hidden_layers=1,
                 method="OLS"):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.

        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        @param method: fitting algorithm (currently, only "OLS" is supported).
        """
        super(GVAR, self).__init__(device=device)

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()

        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_variable, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_variable**2)))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_variable = num_variable
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers

        self.method = method

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_variable]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        if self.method == "OLS":
            preds = torch.zeros((inputs.shape[0], self.num_variable)).to(self.device)
            for k in range(self.order):
                coeff_net_k = self.coeff_nets[k]
                coeffs_k = coeff_net_k(inputs[:, k, :])
                coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_variable, self.num_variable))
                if coeffs is None:
                    coeffs = torch.unsqueeze(coeffs_k, 1)
                else:
                    coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
                coeffs[:, k, :, :] = coeffs_k
                if self.method == "OLS":
                    preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        elif self.method == "BFT":
            NotImplementedError("Backfitting not implemented yet!")
        else:
            NotImplementedError("Unsupported fitting method!")

        return preds, coeffs
    
    def fit(self, feature: np.ndarray, lr=0.001, epochs=100, batch_size=256, 
            _lambda=0.0, gamma=0.0, seed=42, Q=20, alpha=0.5):
        self.lr= lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._lambda = _lambda
        self.gamma = gamma
        self.seed = seed
        self.alpha = alpha

        data_1 = feature
        data_2 = np.flip(feature, axis=0)

        print("-" * 25)
        print("Training model #1...")
        a_hat_1 = self._train(data=[data_1])
        print("Training model #2...")
        a_hat_2 = self._train(data=[data_2])
        a_hat_2 = np.transpose(a_hat_2)

        p = a_hat_1.shape[0]

        print("Evaluating stability...")
        alphas = np.linspace(0, 1, Q)
        qs_1 = np.quantile(a=a_hat_1, q=alphas)
        qs_2 = np.quantile(a=a_hat_2, q=alphas)
        agreements = np.zeros((len(alphas), ))
        for i in range(len(alphas)):
            a_1_i = (a_hat_1 >= qs_1[i]) * 1.0
            a_2_i = (a_hat_2 >= qs_2[i]) * 1.0
            # NOTE: we ignore diagonal elements when evaluating stability
            agreements[i] = (balanced_accuracy_score(y_true=a_2_i[np.logical_not(np.eye(a_2_i.shape[0]))].flatten(),
                                                     y_pred=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten()) +
                             balanced_accuracy_score(y_pred=a_2_i[np.logical_not(np.eye(a_2_i.shape[0]))].flatten(),
                                                     y_true=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten())) / 2
            # If only self-causal relationships are inferred, then set agreement to 0
            if np.sum(a_1_i) <= p or np.sum(a_2_i) <= p:
                agreements[i] = 0
            # If all potential relationships are inferred, then set agreement to 0
            if np.sum(a_1_i) == p**2 or np.sum(a_2_i) == p**2:
                agreements[i] = 0
        alpha_opt = alphas[np.argmax(agreements)]

        print("Max. stab. = " + str(np.round(np.max(agreements), 3)) + ", at α = " + str(alpha_opt))

        q_1 = np.quantile(a=a_hat_1, q=alpha_opt)
        a_hat = (a_hat_1 >= q_1) * 1.0
        # G_pred = nx.DiGraph(a_hat)
        # a_hat = a_hat.astype(np.int32)
        self.DAG = a_hat
    
    def _train(self, data):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        predictors, responses, time_idx = construct_training_dataset(data=data, order=self.order)

        # Optimiser
        optimiser = optim.Adam(params=self.parameters(), lr=self.lr)

        # Loss criterion
        criterion = nn.MSELoss()

        # Run the training and testing
        self.train()
        for epoch in range(self.epochs):
            self.running(epoch, optimiser, predictors, responses, time_idx, criterion)

        # Compute generalised coefficients & estimate causal structure
        self.eval()
        with torch.no_grad():
            coeffs = self.running(1, optimiser, predictors, responses, time_idx, criterion)
            causal_struct_estimate = torch.max(torch.median(torch.abs(coeffs), dim=0)[0], dim=0)[0].detach().cpu().numpy()

        return causal_struct_estimate
    

    def running(self, epoch: int, optimiser: optim, predictors: np.ndarray, responses: np.ndarray,
                time_idx: np.ndarray, criterion: torch.nn.modules.loss):
        if not self.training:
            coeffs_final = torch.zeros((predictors.shape[0], predictors.shape[1], predictors.shape[2],
                                        predictors.shape[2])).to(self.device)

        # Shuffle the data
        inds = np.arange(0, predictors.shape[0])
        if self.training:
            np.random.shuffle(inds)

        # Split into batches
        batch_split = np.arange(0, len(inds), self.batch_size)
        if len(inds) - batch_split[-1] < self.batch_size / 2:
            batch_split = batch_split[:-1]

        incurred_loss = 0
        incurred_base_loss = 0
        incurred_penalty = 0
        incurred_smoothness_penalty = 0
        for i in range(len(batch_split)):
            if i < len(batch_split) - 1:
                predictors_b = predictors[inds[batch_split[i]:batch_split[i + 1]], :, :]
                responses_b = responses[inds[batch_split[i]:batch_split[i + 1]], :]
                time_idx_b = time_idx[inds[batch_split[i]:batch_split[i + 1]]]
            else:
                predictors_b = predictors[inds[batch_split[i]:], :, :]
                responses_b = responses[inds[batch_split[i]:], :]
                time_idx_b = time_idx[inds[batch_split[i]:]]

            inputs = Variable(torch.tensor(predictors_b, dtype=torch.float64)).to(self.device)
            targets = Variable(torch.tensor(responses_b, dtype=torch.float64)).to(self.device)

            # Get the forecasts and generalised coefficients
            preds, coeffs = self(inputs=inputs)
            if not self.training:
                if i < len(batch_split) - 1:
                    coeffs_final[inds[batch_split[i]:batch_split[i + 1]], :, :, :] = coeffs
                else:
                    coeffs_final[inds[batch_split[i]:], :, :, :] = coeffs

            # Loss
            # Base loss
            base_loss = criterion(preds, targets)

            # Sparsity-inducing penalty term
            # coeffs.shape:     [T x K x p x p]
            penalty = (1 - self.alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                      self.alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

            # Smoothing penalty term
            next_time_points = time_idx_b + 1
            inputs_next = Variable(torch.tensor(predictors[np.where(np.isin(time_idx, next_time_points))[0], :, :],
                                                dtype=torch.float64)).to(self.device)
            preds_next, coeffs_next = self(inputs=inputs_next)
            penalty_smooth = torch.norm(coeffs_next - coeffs[np.isin(next_time_points, time_idx), :, :, :], p=2)

            loss = base_loss + self._lambda * penalty + self.gamma * penalty_smooth
            # Incur loss
            incurred_loss += loss.data.cpu().numpy()
            incurred_base_loss += base_loss.data.cpu().numpy()
            incurred_penalty += self._lambda * penalty.data.cpu().numpy()
            incurred_smoothness_penalty += self.gamma * penalty_smooth.data.cpu().numpy()

            if self.training:
                # Make an optimisation step
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        if self.training:
            print("Epoch " + str(epoch) + " : incurred loss " + str(incurred_loss) + "; incurred sparsity penalty " +
                  str(incurred_penalty) + "; incurred smoothness penalty " + str(incurred_smoothness_penalty))

        if not self.training:
            return coeffs_final