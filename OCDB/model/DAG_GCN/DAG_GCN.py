
import torch

import numpy as np
import networkx as nx

from ...metrics import Metrics
from ..BaseModel import BaseModel
from .DAG_GCN_module import DAG_GCN_Encoder, DAG_GCN_Decoder
from .DAG_GCN_utils import *


class DAG_GCN(BaseModel):
    """Directed Acyclic Causal Graph Discovery from Real World Data using Graph Convolutional Networks

    References
    ----------
    https://ieeexplore.ieee.org/document/10066790

    Parameters
    ----------
    device: torch.device("cpu") or torch.device("cuda")
    num_variable: int,
        the number of variables.
    x_dim: int,
        the last dimension of feature.
    hidden_dim_encoder: int, default: 64
        the hidden layer dimension of encoder.
    hidden_dim_decoder: int, default: 64
        the hidden layer dimension of decoder.
    output_dim: int, default equal to input dimension
        encoder output dimension
    k_max_iter: int, default: 100
        the max iteration number for searching lambda and c.
    epochs: int, default: 300
        train epochs
    lambda_A: float, default: 0.0
        coefficient for DAG constraint h(A).
    c_A: float, default: 1.0
        coefficient for absolute value h(A).
    eta: float, default: 10
        parameter of the augmented Lagrangian approach in DAG-GNN paper.
    gamma: float, default: 0.25
        parameter of the augmented Lagrangian approach in DAG-GNN paper.
    h_tolerance: float, default: 1e-8
        the tolerance of error of h(A) to zero.
    tau_A: float, default: 0.1
        coefficient for L-1 norm of A.
    graph_threshold: float, default: 0.3
        threshold for learned adjacency matrix binarization.
        greater equal to graph_threshold denotes has causal relationship.
    adj_high: float, default: 0.1
        The dense adjacency matrix is initialized with a random upper bound.
    adj_low: float, default: -0.1
        The dense adjacency matrix randomly initializes the lower bound.
    lr: float, default: 1e-3
        learning rate
    lr_decay: int, default: 100
        Period of learning rate decay.
    lr_gamma: float, default: 1.0
        Multiplicative factor of learning rate decay.
    """
    def __init__(self, device, num_variable, x_dim, hidden_dim_encoder=512, hidden_dim_decoder=512, output_dim=1,
                 k_max_iter=3, epochs=100, lambda_A=0.0, c_A=1, eta=10, gamma=0.25, h_tolerance=1e-8, tau_A=0.1,
                 graph_threshold=0.3, adj_high=0.1, adj_low=-0.1, lr=1e-3, lr_decay=100, lr_gamma=1.0, batch_size=256):
        super().__init__(device=device)

        self.dense_adj = None
        init_adj = np.random.uniform(low=adj_low,
                                     high=adj_high,
                                     size=(num_variable, num_variable))
        self.encoder = DAG_GCN_Encoder(in_channels=x_dim,
                                       H1=hidden_dim_encoder,
                                       H2=hidden_dim_encoder,
                                       H3=hidden_dim_encoder,
                                       out_channels=output_dim,
                                       activation=torch.nn.LeakyReLU(),
                                       batch_norm=True,
                                       num_features=num_variable,
                                       init=True)
        self.encoder.set_dense_adj(init_adj, init=True)
        self.decoder = DAG_GCN_Decoder(in_channels=output_dim,
                                       H1=hidden_dim_decoder,
                                       H2=hidden_dim_decoder,
                                       H3=hidden_dim_decoder,
                                       out_channels=x_dim,
                                       activation=torch.nn.LeakyReLU(),
                                       batch_norm=True,
                                       num_features=num_variable,
                                       init=True)

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay, gamma=lr_gamma)

        # hparams
        self.num_variable = num_variable
        self.k_max_iter = k_max_iter
        self.epochs = epochs
        self.lambda_A = lambda_A
        self.c_A = c_A
        self.eta = eta
        self.gamma = gamma
        self.h_tolerance = h_tolerance
        self.tau_A = tau_A
        self.graph_threshold = graph_threshold
        self.batch_size = batch_size

    def forward(self, X):
        Z, self.dense_adj = self.encoder(X)
        X_hat = self.decoder(Z, self.dense_adj)
        return Z, self.dense_adj, X_hat

    def fit(self, feature):
        print("feature.shape:", feature.shape)
        feature = feature.reshape(-1, self.num_variable, 1)
        train_loader = data_process(feature, self.batch_size)

        # optimizer step on hyparameters
        h_A_new = torch.tensor(1.)
        h_A_old = np.inf
        best_loss = np.inf
        best_epoch = 0
        best_loss_dag = None

        for step_k in range(self.k_max_iter):
            while self.c_A < 1e+20:
                for epoch in range(self.epochs):
                    for batch_idx, data in enumerate(train_loader):
                        self.optimizer.zero_grad()

                        loss = self.training_step(data, batch_idx, epoch)
                        # update best
                        if loss < best_loss:
                            best_loss = loss
                            best_epoch = epoch
                            best_loss_dag = self.dense_adj.detach().cpu().data.clone().numpy()
                        loss.backward()
                        self.optimizer.step()
                    # Reducing learning rate every epoch
                    self.scheduler.step()
                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))

                # update parameters
                A_new = self.dense_adj.data.clone()
                h_A_new = self._get_h_A(A_new)
                if h_A_new.item() > self.gamma * h_A_old:
                    self.c_A *= self.eta
                else:
                    break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            self.lambda_A += self.c_A * h_A_new.item()

            if h_A_new.item() <= self.h_tolerance:
                break

        graph = best_loss_dag
        graph[np.abs(graph) < self.graph_threshold] = 0
        graph[np.abs(graph) >= self.graph_threshold] = 1
        self.DAG = graph

    def eval(self, G_true, G):
        fdr = Metrics.fdr(G_true, G)
        tpr = Metrics.tpr(G_true, G)
        fpr = Metrics.fpr(G_true, G)
        shd = Metrics.shd(G_true, G)
        B = nx.to_numpy_array(G) != 0
        # linear index of nonzeros
        pred = np.flatnonzero(B)
        nnz = len(pred)
        return fdr, tpr, fpr, shd, nnz

    def training_step(self, data, batch_idx, epoch):
        X = data[0].to(self.device).float()
        Z, self.dense_adj, X_hat = self.forward(X)

        # print("forword done, start loss")
        loss, losses = self.loss_compute(X.squeeze(), Z.squeeze(), self.dense_adj, X_hat.squeeze())
        print('Epoch: {:03d}'.format(epoch),
              'batch: {:03d}'.format(batch_idx),
              'loss_train: {:.6f}'.format(loss.item()),
              'ELBO_loss: {:.6f}'.format(losses["ELBO_loss"]),
              'NLL_loss: {:.6f}'.format(losses["NLL_loss"]),
              'KL_loss: {:.6f}'.format(losses["KL_loss"]),
              'MSE_loss: {:.6f}'.format(losses["MSE_loss"]))
        return loss

    def loss_compute(self, X, Z, dense_adj, X_hat):
        # reconstruction accuracy loss
        NLL_loss = self._nll_gaussian(X, X_hat)
        # KL loss
        KL_loss = self._kl_gaussian_sem(Z)
        # ELBO loss
        ELBO_loss = KL_loss + NLL_loss
        # sparse loss
        sparse_loss = self._sparse_loss(dense_adj, self.tau_A)
        # compute h(A)
        h_A = self._get_h_A(dense_adj)
        loss = ELBO_loss + (
                self.lambda_A * h_A
                + 0.5 * self.c_A * h_A * h_A
                + 100 * torch.trace(dense_adj * dense_adj)
                + sparse_loss)
        losses = {
            "ELBO_loss": ELBO_loss.item(),
            "NLL_loss": NLL_loss.item(),
            "KL_loss": KL_loss.item(),
            "MSE_loss": torch.nn.MSELoss()(X, X_hat).item(),
        }
        # print(loss)
        return loss, losses

    @staticmethod
    def _nll_gaussian(X, X_hat, variance=0.0, add_const=False):
        # DAG-GNN paper equation (9)
        neg_log_p = variance + torch.div(torch.pow(X_hat - X, 2), 2.0 * np.exp(2.0 * variance))
        if add_const:
            const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
            neg_log_p += const
        return neg_log_p.sum() / (X.size(0))

    @staticmethod
    def _kl_gaussian_sem(Z):
        # DAG-GNN paper equation (8)
        return 0.5 * (torch.sum(Z ** 2) / Z.size(0))

    @staticmethod
    def _get_h_A(adj):
        # DAG-GNN paper (13)
        m = adj.shape[0]
        x = torch.eye(m).float().type_as(adj) + torch.div(adj * adj, m)
        matrix_poly = torch.matrix_power(x, m)
        expm_A = matrix_poly
        h_A = torch.trace(expm_A) - m
        return h_A

    @staticmethod
    def _sparse_loss(adj, tau):
        return tau * torch.sum(torch.abs(adj))