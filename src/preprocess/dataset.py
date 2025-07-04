import json
from pathlib import Path
from typing import *
from dataclasses import dataclass
from functools import reduce
import pandas as pd

from src.utils import *
from src.preprocess.result import ResultData, Metadata
from src.logger import setup_logger, setup_logging


@dataclass
class DatasetConfig:
    """Configuration for data processing."""
    data_dir:   Path = get_data_dir()
    raw_dir:    Path = get_raw_dir()
    filter_dir: Path = get_filter_dir()
    clean_dir: Path = get_clean_dir()
    csv_paths: List[Path] = None
    overall_start_date: Optional[str] = None  # YYYY-MM-DD
    overall_end_date:   Optional[str] = None  # YYYY-MM-DD
    type: Literal["raw", "filter", "clean"] = "clean"


class Dataset:
    """High-level class for dataset handling."""

    def __init__(self, config:Optional[DatasetConfig] = None) -> None:
        if config is None:
            config = DatasetConfig()
        self.config = config 

        setup_logging()
        self.logger = setup_logger("dataset.log")

        if self.config.type == "raw":
            self.data_dir = self.config.raw_dir
        elif self.config.type == "filter":
            self.data_dir = self.config.filter_dir
        elif self.config.type == "clean":
            self.data_dir = self.config.clean_dir
        # discover all CSV files once
        self.config.csv_paths = list(self.data_dir.rglob("*.csv"))

        # internal caches
        self._path_names_dict: Optional[Dict[Path, str]]     = None
        self._data_dict:      Optional[Dict[str, pd.DataFrame]] = None
        self._ml_ready:       Optional[pd.DataFrame]         = None
        self._metadata:       Optional[Metadata]             = None

        self._overall_start_date = self._parse_date(self.config.overall_start_date)
        self._overall_end_date   = self._parse_date(self.config.overall_end_date)

    def get(
            self, 
            datadict: bool = True,
            ml_ready: bool = True,
            metadata: bool = True,
        ) -> ResultData:
        
        result = ResultData(
            path_names_dict=None,
            datadict=datadict,
            ml_ready=ml_ready,
            metadata=metadata
            )
        if result.path_names_dict is None:
            result.path_names_dict = self.path_names_dict
        if result.datadict is not False:
            result.datadict = self.data_dict
        if result.ml_ready is not False:
            result.ml_ready = self.ml_ready
        if result.metadata is not False:
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
            countries_meta = self._load_countries()
            alias_map = {
                alias: canon
                for canon, info in countries_meta.items()
                for alias in info["aliases"]
            }
            canonical = list(countries_meta.keys())

            loaded: Dict[str, pd.DataFrame] = {}
            for path, name in self.path_names_dict.items():
                try:
                    # read & parse
                    df = (
                        pd.read_csv(path, parse_dates=["date"])
                        .set_index("date")
                    )
                    df.columns = (
                        df.columns
                            .str.strip()
                            .str.lower()
                            .str.replace(r"[^\w]+", "_", regex=True)
                            .str.replace(r"_+", "_", regex=True)
                            .str.strip("_")
                        )

                    df = df.rename(columns=alias_map)

                    df = df.T.groupby(level=0).first().T

                    # drop non-canonical columns, then reindex in canonical order
                    df = df.loc[:, df.columns.isin(canonical)]
                    df = df.reindex(columns=canonical)

                    # cast & sort
                    df = df.astype("float32").sort_index()
                    loaded[name] = df

                except Exception as e:
                    self.logger.error(f"Failed to load {path}: {e}")
            
            if not loaded:
                self._data_dict = {}
                return self._data_dict
            
            # compute full union date range
            all_starts  = [df.index.min() for df in loaded.values()]
            all_ends    = [df.index.max() for df in loaded.values()]
            union_start = min(all_starts)
            union_end   = max(all_ends)

            full_idx = pd.date_range(union_start, union_end, freq="YS")
            full_idx.name = "date"

            for name, df in loaded.items():
                loaded[name] = df.reindex(full_idx)
            
            self._data_dict = loaded

            if not self.config.type == "raw":
                pruned: Dict[str, pd.DataFrame] = {}
                for name, df in loaded.items():
                    df2 = df.reindex(full_idx)
                    # drop any country-column entirely NaN
                    df2 = df2.dropna(axis=1, how="all")
                    # only keep features that have at least one country
                    if df2.shape[1] > 0:
                        pruned[name] = df2
                self._data_dict = pruned

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

    def get_country_data(self, country_name: str) -> pd.DataFrame | None:
        """
        return all numeric variables of a country, indexed by year, from the year 2000 onwards.
        """
        df = self.ml_ready
        if df.empty:
            return None

        country_df = df.loc[df["country"].str.lower() == country_name.lower()].copy()
        if country_df.empty:
            return None

        country_df["year"] = country_df["date"].dt.year

        if country_df.empty:
            return None

        country_df.set_index("year", inplace=True)
        country_df.drop(columns=["date", "country"], inplace=True, errors="ignore")
        country_df.reset_index(inplace=True)

        return country_df


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
            cats[name] = cat
        return cats

    def _load_countries(self) -> Dict[str, Any]:
        p = get_metadata_dir() / "countries.json"
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading countries.json: {e}")
            return {}

    def _load_indicators(self) -> Dict[str, Any]:
        p = get_metadata_dir() / "indicators.json"
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
