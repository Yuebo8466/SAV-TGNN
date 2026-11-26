#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ======================================================
# 多任务 GCN-GRU：Presence (classification) + Density (regression)
# ======================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class MultiTask_GCN_GRU_Temporal(nn.Module):
    """
    ### MOD:
      - 使用 GCN + GRU 提取节点时空 embedding
      - 使用两个独立 MLP 头：
          * presence_head: 分类 (2 类)
          * density_head:  回归 (1 输出)
    """

    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int = 64,
        gcn_out: int = 64,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gcn_encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=gcn_hidden,
            out_channels=gcn_out,
            dropout=dropout
        )

        self.gru = nn.GRU(
            input_size=gcn_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True
        )

        self.dropout = dropout

        # === 分类头 MLP ===
        self.cls_mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(gru_hidden, 2)  # presence / absence
        )

        # === 回归头 MLP ===
        self.reg_mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(gru_hidden, 5) #原来是1,改成5:5个density 等级: 0~4
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor):
        """
        x_seq: [T, N, F]
        edge_index: [2, E]
        返回:
          logits_cls: [N, 2]
          pred_reg:   [N]
        """
        T, N, Fdim = x_seq.shape

        spatial_list = []
        for t in range(T):
            h_t = self.gcn_encoder(x_seq[t], edge_index)
            spatial_list.append(h_t.unsqueeze(0))

        H_seq = torch.cat(spatial_list, dim=0)      # [T,N,gcn_out]
        H_seq = H_seq.permute(1, 0, 2).contiguous() # [N,T,gcn_out]

        gru_out, _ = self.gru(H_seq)
        h_last = gru_out[:, -1, :]                  # [N,gru_hidden]
        h_last = F.dropout(h_last, p=self.dropout, training=self.training)

        logits_cls = self.cls_mlp(h_last)           # [N,2]
     #  pred_reg   = self.reg_mlp(h_last).squeeze(-1)  # [N] 原来用的
        logits_reg = self.reg_mlp(h_last)           # [N,5] 等级 logits

        return logits_cls ,logits_reg  #, pred_reg

    @torch.no_grad()
    def forward_embed(self, x_seq, edge_index):
        """
        返回最终节点 embedding，方便画 t-SNE / UMAP
        """
        T, N, Fdim = x_seq.shape

        spatial_list = []
        for t in range(T):
            h_t = self.gcn_encoder(x_seq[t], edge_index)
            spatial_list.append(h_t.unsqueeze(0))

        H_seq = torch.cat(spatial_list, dim=0)
        H_seq = H_seq.permute(1, 0, 2).contiguous()

        gru_out, _ = self.gru(H_seq)
        h_last = gru_out[:, -1, :]

        return h_last
