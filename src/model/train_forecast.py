# src/trainers/forecast_trainer.py
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from typing import *

from src.preprocess.forecast_dataset import (
    ForecastConfig,
    ForecastDataset
)
from src.model.forecast import LSTMForecaster, ForecastModelConfig
from src.logger import setup_logger
import pandas as pd
from torch.utils.data import DataLoader

logger = setup_logger("train_forecast.log")


def train_forecast(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    feat_cols:    list[str],
    label_col:    str,
    window_size:  int,
    horizon:      int,
    val_frac:     float,
    test_frac:    float,
    batch_size:   int   = 64,
    max_epochs:   int   = 100,
    patience:     int   = 20,
    lr:           float = 1e-3,
) -> Tuple[
    LSTMForecaster, 
    Tuple[DataLoader, DataLoader, DataLoader]
    ]:
    """
    Trains an LSTM to predict the next `horizon` steps of `label_col`
    from a sliding window of the last `window_size` rows of `feat_cols`.
    Returns the trained model and the test DataLoader.
    """
    # 1) Build our ForecastConfig and loaders
    cfg = ForecastConfig(
        window_size=window_size,
        horizon=horizon,
        feat_cols=feat_cols,
        label_col=label_col,
        val_frac=val_frac,
        test_frac=test_frac,
        batch_size=batch_size
    )

    # 2) Instantiate model / optimizer / loss
    model_cfg = ForecastModelConfig(
        n_features=len(feat_cols),
        horizon=horizon
    )
    model = LSTMForecaster(model_cfg)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    best_val_loss = float('inf')
    counter       = 0

    # 3) Training loop with validation
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * xb.size(0)

        avg_train = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                total_val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        avg_val = total_val_loss / len(val_loader.dataset)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"{counter} epochs achieved without benefit, trigger patience!")
                break


        logger.info(
            f"Epoch {epoch}/{max_epochs} train={avg_train:.4f} val={avg_val:.4f}"
        )

    return model, (train_loader, val_loader, test_loader)
