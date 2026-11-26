import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTask_MLP_Static(nn.Module):
    """
    无图无时序：仅对单年特征做 MLP
    """
    def __init__(self, in_channels, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_cls = nn.Linear(hidden, 2)
        self.head_reg = nn.Linear(hidden, 5)

    def forward(self, x):
        h = self.net(x)
        return self.head_cls(h), self.head_reg(h)
