import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

@dataclass
class ForecastConfig:
    # sliding‐window settings
    window_size: int               # k
    horizon:     int               # m
    feat_cols:   List[str]         # numeric features
    label_col:   str               # sliding‐window target
    # panel settings
    group_col:   str = 'country'
    date_col:    str = 'date'
    # splits & batching
    val_frac:    float = 0.1
    test_frac:   float = 0.1
    batch_size:  int   = 64
    # two separate scalers
    panel_scaler: Optional[StandardScaler] = None
    window_scaler: Optional[StandardScaler] = None


class ForecastDataset:
    """
    A single class handling both:
    
    1) panel → DataLoader over [B, C, F] via `make_panel_loader`
    2) sliding‐window → (train, val, test) over [N, k, F]→[N, m] via `make_sliding_loaders`
    """

    def __init__(self, df: pd.DataFrame, cfg: ForecastConfig):
        # build sliding‐window arrays
        Xs, ys = [], []
        for _, grp in df.groupby(cfg.group_col, sort=False):
            grp = grp.sort_values(cfg.date_col)
            arr = grp[cfg.feat_cols + [cfg.label_col]].to_numpy(dtype=float)

            # apply window_scaler if given (must match feat_cols+label)
            if cfg.window_scaler is not None:
                arr = cfg.window_scaler.transform(arr)

            k, m = cfg.window_size, cfg.horizon
            L = len(arr)
            if L < k + m:
                continue  # skip too‐short groups

            for i in range(L - k - m + 1):
                Xs.append(arr[i : i + k, :-1])
                ys.append(arr[i + k : i + k + m, -1])

        if not Xs:
            raise ValueError(f"No sliding‐window samples. "
                             f"Each group must have ≥ window_size+horizon rows.")

        self.X = torch.from_numpy(np.stack(Xs)).float()  # [N, k, F]
        self.y = torch.from_numpy(np.stack(ys)).float()  # [N, m]


    def __len__(self):
        # now len(ds) == number of samples
        return self.X.shape[0]

    @staticmethod
    def panel_to_array(
        df: pd.DataFrame,
        date_col:  str,
        group_col: str,
        feat_cols: List[str]
    ) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
        """
        Convert long‐form panel → np.ndarray [B, C, F],
        plus sorted time & group lists.
        """
        times  = sorted(df[date_col].unique())
        groups = sorted(df[group_col].unique())
        B, C, F = len(times), len(groups), len(feat_cols)

        arr = np.zeros((B, C, F), dtype=float)
        for i, t in enumerate(times):
            sub = df[df[date_col] == t]
            for j, g in enumerate(groups):
                row = sub[sub[group_col] == g]
                if not row.empty:
                    arr[i, j, :] = row[feat_cols].iloc[0].values
        return arr, times, groups

    @classmethod
    def make_panel_loader(
        cls, df: pd.DataFrame, cfg: ForecastConfig
    ) -> DataLoader:
        """
        DataLoader over the full panel as [B, C, F], batch-shuffled.
        """
        arr, _, _ = cls.panel_to_array(
            df, cfg.date_col, cfg.group_col, cfg.feat_cols
        )

        # apply panel_scaler if provided (must match feat_cols)
        if cfg.panel_scaler is not None:
            B, C, F = arr.shape
            flat = arr.reshape(B*C, F)
            flat = cfg.panel_scaler.transform(flat)
            arr = flat.reshape(B, C, F)

        tensor = torch.from_numpy(arr).float()

        class _PDataset(Dataset):
            def __init__(self, data: torch.Tensor):
                self.data = data
            def __len__(self):
                return self.data.shape[0]
            def __getitem__(self, i):
                return self.data[i]

        return DataLoader(
            _PDataset(tensor),
            batch_size=cfg.batch_size,
            shuffle=True
        )

    @classmethod
    def make_sliding_loaders(
        cls, df: pd.DataFrame, cfg: ForecastConfig
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Train/val/test DataLoaders for sliding-window forecasting.
        """
        fd = cls(df, cfg)  # builds self.X, self.y
        full_ds = TensorDataset(fd.X, fd.y)
        n = len(full_ds)
        n_test  = int(n * cfg.test_frac)
        n_val   = int(n * cfg.val_frac)
        n_train = n - n_val - n_test

        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test]
        )
        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False),
            DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False),
        )
