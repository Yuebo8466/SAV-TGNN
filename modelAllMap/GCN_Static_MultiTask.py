import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MultiTask_GCN_Static(nn.Module):
    """
    静态图基线：只用单年特征做 GCN（无 GRU）
    输入: x [N,F], edge_index
    输出: logits_p [N,2], logits_d [N,5]
    """
    def __init__(self, in_channels, gcn_hidden=64, gcn_out=64, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, gcn_hidden)
        self.conv2 = GCNConv(gcn_hidden, gcn_out)
        self.drop = nn.Dropout(dropout)
        self.head_cls = nn.Linear(gcn_out, 2)
        self.head_reg = nn.Linear(gcn_out, 5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x); x = self.drop(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x); x = self.drop(x)
        return self.head_cls(x), self.head_reg(x)
