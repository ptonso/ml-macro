# src/models/regression_model.py
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class RegConfig:
    n_features: int
    hidden_sizes: list = (64, 32)
    dropout: float = 0.1

class MLPRegressor(nn.Module):
    def __init__(self, cfg: RegConfig):
        super().__init__()
        layers = []
        in_dim = cfg.n_features
        for h in cfg.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
