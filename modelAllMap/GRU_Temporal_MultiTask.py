import torch
import torch.nn as nn

class MultiTask_GRU_Temporal(nn.Module):
    """
    无图深度时序基线：每个节点独立做 GRU
    输入:  x_seq [T,N,F]
    输出:  logits_p [N,2], logits_d [N,5]
    """
    def __init__(self, in_channels, gru_hidden=64, gru_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=False
        )
        self.drop = nn.Dropout(dropout)
        self.head_cls = nn.Linear(gru_hidden, 2)
        self.head_reg = nn.Linear(gru_hidden, 5)

    def forward(self, x_seq):
        out, _ = self.gru(x_seq)     # [T,N,H]
        h = self.drop(out[-1])       # [N,H]
        return self.head_cls(h), self.head_reg(h)
