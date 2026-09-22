
from __future__ import annotations
import torch
from torch import nn

class BlockScorer(nn.Module):
    def __init__(self,input_dim=11,hidden=48):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x): return self.net(x).squeeze(-1)
