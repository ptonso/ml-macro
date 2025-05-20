# src/data/annual_dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AnnualConfig:
    feature_cols: List[str]
    label_col: str
    val_frac: float = 0.2
    test_frac: float = 0.1
    batch_size: int = 16

class AnnualDataset(Dataset):
    """
    Yearly-aggregated features → single-label regression.
    """
    def __init__(self, df: pd.DataFrame, cfg: AnnualConfig):
        df = df.copy()
        df['year'] = pd.to_datetime(df['date']).dt.year
        agg = {c: 'mean' for c in cfg.feature_cols}
        agg[cfg.label_col] = 'mean'
        yearly = df.groupby('year').agg(agg).reset_index()
        X = yearly[cfg.feature_cols].values
        y = yearly[cfg.label_col].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_annual_loaders(
    df: pd.DataFrame, cfg: AnnualConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = AnnualDataset(df, cfg)
    n = len(ds)
    n_test = int(n * cfg.test_frac)
    n_val  = int(n * cfg.val_frac)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    return (
      DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
      DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False),
      DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False),
    )
