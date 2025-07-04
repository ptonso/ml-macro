# src/models/forecast_model.py
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ForecastModelConfig:
    n_features: int
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    horizon: int = 1

class LSTMForecaster(nn.Module):
    def __init__(self, cfg: ForecastModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout
        )
        # for multi-step output, map hidden → horizon
        self.head = nn.Linear(cfg.hidden_size, cfg.horizon)

    def forward(self, x):
        # x: [B, k, n_features]
        out, _ = self.lstm(x)               # [B, k, H]
        last = out[:, -1, :]                # [B, H]
        return self.head(last)              # [B, horizon]
