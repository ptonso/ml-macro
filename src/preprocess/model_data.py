
from pathlib import Path
import os
from torch.serialization import safe_globals  # PyTorch ≥2.6
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import DataLoader

from typing import *
from sklearn.preprocessing import StandardScaler
from src.preprocess.forecast_dataset import ForecastConfig, ForecastDataset
from src.model.train_forecast import train_forecast, test_result


@dataclass
class FreezeModelLoader:
    model        : DataLoader
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    feat_cols    : List[str]
    label_col    : str



class ModelLoaders:

    def __init__(
            self, 
            df: pd.DataFrame, 
            filepath: Optional[str] = None
        ):
        """
        df
        filepath: str = "data/20--model/LSTM.pth"
        """

        self.ml_ready = df

        if filepath is not None:
            self.package  = self.load(filepath)
            pass

    def train(self, label_col: str = "gdp_current_usd"):

        assert self.ml_ready.isna().sum().sum() == 0, "ml ready should not \
            contain missing values"  # should be 0

        label_col = "gdp_current_usd"
        num_cols  = self.ml_ready.select_dtypes(include="number").columns.tolist()
        feat_cols = [c for c in num_cols if c != label_col]

        assert len(feat_cols) > 0, f"no feature column with name {label_col} detected."


        panel_scaler = StandardScaler().fit(self.ml_ready[feat_cols].values)

        window_scaler = StandardScaler().fit(
            self.ml_ready[feat_cols + [label_col]].values
        )

        cfg = ForecastConfig(
            window_size   = 5,
            horizon       = 15,
            feat_cols     = feat_cols,
            label_col     = label_col,
            group_col     = "country",
            date_col      = "date",
            val_frac      = 0.1,
            test_frac     = 0.1,
            batch_size    = 64,
            panel_scaler  = panel_scaler,
            window_scaler = window_scaler,
        )

        panel_loader = ForecastDataset.make_panel_loader(self.ml_ready, cfg)
        train_loader, val_loader, test_loader = ForecastDataset.make_sliding_loaders(self.ml_ready, cfg)

        (model, (train_loader, val_loader, test_loader)
         ) = train_forecast(
            train_loader = train_loader, 
            val_loader   = val_loader, 
            test_loader  = test_loader,
            feat_cols    = feat_cols,
            label_col    = label_col,
            window_size  = cfg.window_size,
            horizon      = cfg.horizon,
            val_frac     = cfg.val_frac,
            test_frac    = cfg.test_frac,
            batch_size   = cfg.batch_size,
            max_epochs   = 10000,
            patience     = 30,
            lr           = 1e-4
        )

        self.package = FreezeModelLoader(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            feat_cols=feat_cols,
            label_col=label_col
        )
        
        return self.package

    
    def test(
            self, 
            package: FreezeModelLoader
        ) -> Tuple[np.ndarray, np.ndarray]:
        model = package.model
        test_loader = package.test_loader
        y_true, y_pred =  test_result(model, test_loader)
        return y_true, y_pred
    


    def save(self, filepath: str):
        """
        Atomically save the entire package (model + loaders) to disk.
        """
        path = Path(filepath).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(path.suffix + ".tmp")
        # pickle the whole dataclass
        torch.save(self.package, str(tmp))
        os.replace(str(tmp), str(path))
        print(f"[ModelLoaders] package saved to {path!r}")

    def load(self, filepath: str, map_location=None) -> FreezeModelLoader:
        """
        Reload the package you saved, safely allowing your FreezeModelLoader class.
        """
        path = Path(filepath).with_suffix(".pt")
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path!r}")

        with safe_globals([FreezeModelLoader]):
            pkg: FreezeModelLoader = torch.load(
                str(path),
                map_location=map_location,
                weights_only=False    # must disable weights_only to unpickle entire object
            )

        self.package = pkg
        print(f"[ModelLoaders] package loaded from {path!r}")
        return pkg

