from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import networkx as nx
import pandas as pd

from .utils import Exponential, AverageMeter, generate_sequence_mask, EventSeqDataset, convert_to_bucketed_dataloader, split_dataloader
from torch.utils.data import DataLoader
from ..BaseModel import BaseModel


class RPPN(BaseModel):
    def __init__(
        self,
        device, 
        num_variable,
        embedding_dim=128,
        hidden_size=256,
        lr=0.001, 
        batch_size=512,
        init_scale=10,
        activation=None,
        optim="Adam",
        graph_threshold=0.5
    ):
        super(RPPN, self).__init__(device=device)
        self.num_variable = num_variable
        self.lr = lr
        self.batch_size = batch_size
        self.optim = optim
        self.graph_threshold = graph_threshold

        self.embed = nn.Embedding(num_variable, embedding_dim)
        self.seq_encoder = nn.GRU(
            embedding_dim + 1, hidden_size, batch_first=True
        )
        self.attn_target = nn.Parameter(torch.Tensor(num_variable, hidden_size))

        self.baseline = nn.Parameter(-4 * torch.ones(num_variable))  # ~0.01
        self.activation = activation or (lambda x: F.elu(x) + 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.decay_kernels = Exponential(
            torch.full((num_variable,), init_scale), requires_grad=True
        )

        nn.init.xavier_uniform_(self.attn_target)

    def forward(self, batch, need_excitations=False, need_weights=False):
        """[summary]

        Args:
            batch (Tensor): size=[B, T, 2]

        Returns:
            intensities (Tensor): [B, T, n_types]
              conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
            excitations (Tensor): [B, T, n_types]
              excitation right after each event, starting at t0.
            unnormalized_weights (Tensor): [B, T, n_types]
              unnormalized attention weights for the predicitons at each event,
              starting at t0 (i.e., for the interval (t0, t1])
        """
        # (t0=0, t1, t2, ..., t_n)
        ts = F.pad(batch[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})
        type_feat = F.pad(self.embed(batch[:, :-1, 1].long()), (0, 0, 1, 0))

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        # [B, T, hidden_size]
        history_emb, *_ = self.seq_encoder(feat)

        # [B, T, n_types]
        unnormalized_weights = (
            (history_emb @ self.attn_target.t()).tanh().exp()
        )
        normalization = unnormalized_weights.cumsum(1) + 1e-10

        # [B, T, n_types]; apply fc to history_emb first; otherwise the
        # synthesized context embedding can be very large when both T and K are
        # large
        excitations = self.activation(
            (self.fc(history_emb) * unnormalized_weights)
            .cumsum(1)
            .div(normalization)
        )
        intensities = self.activation(self.baseline).add(
            excitations * self.decay_kernels.eval(dt[:, 1:, None])
        )

        ret = [intensities]

        if need_excitations:
            ret.append(excitations)

        if need_weights:
            ret.append(unnormalized_weights.squeeze(-1).detach())

        return ret[0] if len(ret) == 1 else tuple(ret)

    def _eval_nll(self, batch, intensity, excitation, mask):

        # sum log intensity of the corresponding event type
        loss_part1 = (
            -intensity.gather(dim=2, index=batch[:, :, 1:].long())
            .squeeze(-1)
            .log()
            .masked_select(mask)
            .sum()
        )

        # NOTE: under the assumption that CIFs are piece-wise constant
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1}); [B, T, 1]
        dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)

        loss_part2 = (
            (self.activation(self.baseline) * dt)
            .add(excitation * self.decay_kernels.integral(dt))
            .sum(-1)
            .masked_select(mask)
            .sum()
        )

        nll = (loss_part1 + loss_part2) / batch.size(0)
        return nll

    def fit(self, event_table: pd.DataFrame, columns, epochs, num_workers, save_path, split_ratio=8/9, removed=True):
        dataloader_args = {
                "batch_size": self.batch_size,
                "collate_fn": EventSeqDataset.collate_fn,
                "num_workers": num_workers,
        }
        ids = [i for i in range(self.num_variable)]
        event_table = event_table.replace(columns, ids)
        event_seq = event_table.loc[:, ["time_stamp", "event_type"]].to_numpy()
        feature = torch.reshape(torch.Tensor(event_seq), (-1, 2, event_seq.shape[1])).float()
        print(feature.shape)

        dataloader = DataLoader(
            EventSeqDataset(feature), **dataloader_args
        )

        train_dataloader, valid_dataloader = split_dataloader(dataloader, split_ratio)

        optimizer = getattr(torch.optim, self.optim)(
        self.parameters(), lr=self.lr
        )
        best_metric = 1e10
        for epoch in range(epochs):
            self.train()
            train_metrics = defaultdict(AverageMeter)

            for batch in train_dataloader:
                batch = batch.to(self.device)

                seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_lengths)
                intensity, excitation = self(batch, need_excitations=True)

                loss = nll = self._eval_nll(batch, intensity, excitation, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_metrics["nll"].update(nll, batch.size(0))
            if valid_dataloader:
                valid_metrics = self.evaluate(valid_dataloader)
            else:
                valid_metrics = None
            msg = f"[Training] Epoch={epoch}"
            for k, v in train_metrics.items():
                msg += f", {k}={v.avg:.4f}"
            print(msg)
            msg = f"[Validation] Epoch={epoch}"
            for k, v in valid_metrics.items():
                msg += f", {k}={v.avg:.4f}"
            print(msg)
            if valid_dataloader and valid_metrics["nll"].avg < best_metric:
                print(f"Found a better model at epoch {epoch}.")
                best_metric = valid_metrics["nll"].avg
                torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
        
        self.load_state_dict(torch.load(os.path.join(save_path, "model.pt")))
        if removed:
            os.remove(os.path.join(save_path, "model.pt"))
        
        dataloader = convert_to_bucketed_dataloader(dataloader, key_fn=len)
        infectivity = self.get_infectivity(dataloader)
        # print(np.min(infectivity), np.max(infectivity))
        # self.graph_threshold = np.max(np.abs(infectivity)) - 0.01
        self.graph_threshold = np.max(np.abs(infectivity)) - self.graph_threshold
        # print(self.graph_threshold)
        infectivity[np.abs(infectivity) < self.graph_threshold] = 0
        infectivity[np.abs(infectivity) >= self.graph_threshold] = 1
        # print(np.sum(infectivity))
        # pred_G = nx.DiGraph(infectivity)
        self.DAG = infectivity


    def evaluate(self, dataloader):
        self.eval()

        metrics = defaultdict(AverageMeter)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_lengths)
                intensity, excitation = self(batch, need_excitations=True)
                nll = self._eval_nll(batch, intensity, excitation, mask)

                metrics["nll"].update(nll, batch.size(0))

        return metrics
    
    def get_infectivity(self, dataloader):
        A = torch.zeros(self.num_variable, self.num_variable, device=self.device)
        type_count = torch.zeros(self.num_variable, device=self.device).long()
        self.eval()
        for batch in tqdm(dataloader):
            batch_size, T = batch.size()[:2]

            batch = batch.to(self.device)
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_lengths)

            _, unnormalized_weights = self(batch, need_weights=True)
            # both.size = [B, n_types, T]
            unnormalized_weights = unnormalized_weights.transpose(1, 2)
            inv_normalizations = 1 / (unnormalized_weights.cumsum(-1) + 1e-10)

            # cumulative the inverse normalization for all later positions.
            cum_inv_normalizations = (
                inv_normalizations.masked_fill(~mask[:, None, :], 0)
                .flip([-1])
                .cumsum(-1)
                .flip([-1])
            )

            # [K, B, T - 1]; drop t0
            event_scores = unnormalized_weights * cum_inv_normalizations
            event_scores = event_scores[:, :, 1:]

            types = batch[:, :, 1].long()
            A.scatter_add_(
                dim=1,
                index=types[:, :-1].reshape(1, -1).expand(self.num_variable, -1),
                src=event_scores.reshape(self.num_variable, -1),
            )

            valid_types = types.masked_select(mask).long()
            type_count.scatter_add_(
                0, index=valid_types, src=torch.ones_like(valid_types)
            )

        # plus one to avoid division by zero
        A /= type_count[None, :].float() + 1

        return A.detach().cpu().numpy()


