import torch
import torch_geometric as pyg

import copy
import numpy as np


class DAG_GCN_Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 H1: int,
                 H2: int,
                 H3: int,
                 out_channels: int,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 batch_norm: bool = False,
                 num_features: int = 11,
                 init: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        if init:
            self.init = 'kaiming_uniform'
        else:
            self.init = None

        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (pyg.nn.Linear(in_channels, H1,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, H2,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),  # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (pyg.nn.Linear(in_channels, H1,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, H2,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),  # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def set_dense_adj(self, dense_adj: np.ndarray, init: bool = False):
        self.dense_adj = torch.nn.Parameter(torch.tensor(dense_adj, dtype=torch.float), requires_grad=True)
        if init:
            torch.nn.init.kaiming_uniform_(self.dense_adj)

    def forward(self, x: torch.Tensor):
        x = x.double()
        dense_adj = torch.sinh(3.0 * self.dense_adj).double()
        # dense_adj = torch.nn.functional.leaky_relu(dense_adj)
        Z = self.model(x, dense_adj)
        return Z, dense_adj


class DAG_GCN_Decoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 H3: int,
                 H2: int,
                 H1: int,
                 out_channels: int,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 batch_norm: bool = False,
                 num_features: int = 11,
                 init: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        if init:
            self.init = 'kaiming_uniform'
        else:
            self.init = None
        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),  # TODO: Need or not?
                # MLP
                (pyg.nn.Linear(H2, H1,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, out_channels,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                copy.deepcopy(activation),  # TODO: Need or not?
                # MLP
                (pyg.nn.Linear(H2, H1,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, out_channels,
                               weight_initializer=self.init,
                               bias_initializer="zeros"), 'x -> x'),
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, dense_adj: torch.Tensor) -> torch.Tensor:
        return self.model(x, dense_adj)