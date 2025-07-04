# src/trainers/regression_trainer.py
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from src.preprocess.annual_dataset import make_annual_loaders, AnnualConfig
from src.model.regression import MLPRegressor, RegConfig
from src.logger import setup_logger

logger = setup_logger("train_regression.log")

def train_annual_regression(
    df, feat_cols, label_col,
    val_frac=0.2, test_frac=0.1,
    batch_size=16, epochs=50, lr=1e-3
):
    # 1) loaders
    ann_cfg = AnnualConfig(feat_cols, label_col, val_frac, test_frac, batch_size)
    train_loader, val_loader, test_loader = make_annual_loaders(df, ann_cfg)

    # 2) model + optimizer
    model = MLPRegressor(RegConfig(n_features=len(feat_cols)))
    opt   = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    # 3) train / val loop
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for X, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += loss_fn(model(X), y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        logger.info(f"E{ep}/{epochs} ✓ train={tr_loss:.4f}  val={val_loss:.4f}")

    return model, test_loader
