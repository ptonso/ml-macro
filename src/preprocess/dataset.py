import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from functools import reduce

import pandas as pd

from src.utils import get_data_dir
from src.preprocess.result import ResultData, Metadata
from src.logger import setup_logger, setup_logging


@dataclass
class DatasetConfig:
    """Configuration for data processing."""
    data_dir: Path = get_data_dir()
    csv_paths: List[Path] = None
    overall_start_date: Optional[str] = None  # YYYY-MM-DD
    overall_end_date:   Optional[str] = None  # YYYY-MM-DD


class Dataset:
    """High-level class for dataset handling."""

    def __init__(self) -> None:
        self.config = DatasetConfig()
        setup_logging()
        self.logger = setup_logger("dataset.log")

        # discover all CSV files once
        self.config.csv_paths = list(self.config.data_dir.rglob("*.csv"))

        # internal caches
        self._path_names_dict: Optional[Dict[Path, str]]     = None
        self._data_dict:      Optional[Dict[str, pd.DataFrame]] = None
        self._ml_ready:       Optional[pd.DataFrame]         = None
        self._metadata:       Optional[Metadata]             = None

        self._overall_start_date = self._parse_date(self.config.overall_start_date)
        self._overall_end_date   = self._parse_date(self.config.overall_end_date)

    def get(self, result: ResultData = ResultData()) -> ResultData:
        if result.path_names_dict is None:
            result.path_names_dict = self.path_names_dict
        if result.datadict:
            result.datadict = self.data_dict
        if result.ml_ready:
            result.ml_ready = self.ml_ready
        if result.metadata:
            result.metadata = self.metadata
        return result

    @property
    def path_names_dict(self) -> Dict[Path, str]:
        if self._path_names_dict is None:
            mapping: Dict[Path, str] = {}
            for p in self.config.csv_paths:
                mapping[p] = self._extract_name(p.name)
            self._path_names_dict = mapping
        return self._path_names_dict

    @property
    def data_dict(self) -> Dict[str, pd.DataFrame]:
        if self._data_dict is None:
            loaded: Dict[str, pd.DataFrame] = {}
            for path, name in self.path_names_dict.items():
                try:
                    df = pd.read_csv(path, parse_dates=["date"])
                    df.set_index("date", inplace=True)
                    loaded[name] = df.astype("float32")
                except Exception as e:
                    self.logger.error(f"Failed to load {path}: {e}")
            self._data_dict = loaded
        return self._data_dict

    @property
    def ml_ready(self) -> pd.DataFrame:
        if self._ml_ready is None:
            long_frames: List[pd.DataFrame] = []
            for name, df in self.data_dict.items():
                melted = (
                    df
                    .reset_index()
                    .melt(id_vars="date", var_name="country", value_name=name)
                )
                long_frames.append(melted)
            if not long_frames:
                self._ml_ready = pd.DataFrame()
            else:
                merged = reduce(
                    lambda L, R: pd.merge(L, R, on=["date", "country"], how="outer"),
                    long_frames
                )
                self._ml_ready = merged.sort_values(["date", "country"]).reset_index(drop=True)
        return self._ml_ready

    @property
    def metadata(self) -> Metadata:
        if self._metadata is None:
            self._metadata = Metadata(
                category_dict      = self._build_category_dict(),
                countries          = self._load_countries(),
                indicators         = self._load_indicators(),
                overall_start_date = self._format_date(self._overall_start_date),
                overall_end_date   = self._format_date(self._overall_end_date),
            )
        return self._metadata

    def _extract_name(self, filename: str) -> str:
        base = Path(filename).stem
        return base.replace("_world_bank", "").lower()

    def _build_category_dict(self) -> Dict[str, str]:
        cats: Dict[str, str] = {}
        for path, name in self.path_names_dict.items():
            cat = path.parent.name
            if cat not in cats:
                cats[cat] = name
        return cats

    def _load_countries(self) -> Dict[str, Any]:
        p = self.config.data_dir / "10--metadata" / "countries.json"
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading countries.json: {e}")
            return {}

    def _load_indicators(self) -> Dict[str, Any]:
        p = self.config.data_dir / "10--metadata" / "indicators.json"
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading indicators.json: {e}")
            return {}

    def _parse_date(self, date_str: Optional[str]) -> Optional[pd.Timestamp]:
        if not date_str:
            return None
        try:
            return pd.to_datetime(date_str, format="%Y-%m-%d")
        except Exception as e:
            self.logger.error(f"Error parsing date '{date_str}': {e}")
            return None

    def _format_date(self, ts: Optional[pd.Timestamp]) -> Optional[str]:
        return ts.strftime("%Y-%m-%d") if ts is not None else None



if __name__ == "__main__":
    dataset = Dataset()
    ml_df = dataset.get_ml_ready()
    print(ml_df)
