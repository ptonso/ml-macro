import zipfile
import tempfile
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Any, List

from src.dashboard.config import WorldTimeConfig
from src.dashboard.utils import safe_log_transform, get_iso3
from src.preprocess.dataset import Dataset
from src.preprocess.result import ResultData

class DataManager:
    def __init__(self, cfg: WorldTimeConfig):
        self.cfg     = cfg
        self.dataset = Dataset()

        # 1) load metadata & raw tables once
        res = self.dataset.get(ResultData(datadict=True, metadata=True))
        self._all_raw: Dict[str, pd.DataFrame] = res.datadict or {}
        self.category_dict: Dict[str, str] = res.metadata.category_dict

        # 2) load + simplify shapefile once
        self.world: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.load_shapefile()

        # placeholders for the currently‐selected indicator
        self.years: List[str] = []
        # for each year: {"values": [...], "table": DataFrame}
        self.precomputed: Dict[str, Dict[str, Any]] = {}

    def load_shapefile(self) -> None:
        shp_dir  = self.cfg.shapefile_dir / "ne_110m_admin_0_countries"
        shp_path = shp_dir / "ne_110m_admin_0_countries.shp"
        if not shp_path.exists():
            url = (
                "https://naturalearth.s3.amazonaws.com/110m_cultural/"
                "ne_110m_admin_0_countries.zip"
            )
            resp = requests.get(url, stream=True); resp.raise_for_status()
            shp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(resp.content)
                with zipfile.ZipFile(tmp.name, 'r') as z:
                    z.extractall(shp_dir)
            Path(tmp.name).unlink()

        self.world = gpd.read_file(shp_path)
        self.world["geometry"] = self.world.geometry.simplify(tolerance=0.01)

    def fetch_and_prepare(self, indicator: str) -> None:
        """
        Called once when the user picks an indicator:
        - melts raw[indicator]
        - log/decile
        - for each year builds:
           * a list of normalized_value floats in world order
           * the sorted DataFrame table
        """
        raw = self._all_raw.get(indicator)
        if raw is None:
            raise ValueError(f"No data for indicator '{indicator}'")

        # melt & iso3
        df = (
            raw.reset_index()
               .melt(id_vars="date", var_name="country", value_name="value")
               .assign(iso_a3=lambda d: d["country"].map(get_iso3))
        )
        # log + decile
        df["log_value"] = df["value"].apply(safe_log_transform)
        vals = df["log_value"].dropna()
        decs = np.nanpercentile(vals, np.arange(0,101,10)) if len(vals)>0 else [0]
        def decile(v: float) -> float:
            if pd.isna(v): return np.nan
            i = np.searchsorted(decs, v, side="right") - 1
            return min(max(i, 0), 9)/9.0
        df["normalized_value"] = df["log_value"].apply(decile)

        # years list
        years_int = sorted({pd.to_datetime(d).year for d in raw.index})
        self.years = [str(y) for y in years_int]

        # per-year value arrays + tables
        self.precomputed.clear()
        for yr in self.years:
            yint    = int(yr)
            dfy     = df[df["date"].dt.year==yint]
            df_geo  = dfy[["iso_a3","normalized_value"]]
            merged  = self.world.merge(
                         df_geo,
                         left_on="ADM0_A3", right_on="iso_a3",
                         how="left")
            values  = merged["normalized_value"].fillna(np.nan).tolist()
            table   = dfy.sort_values("value", ascending=False)
            self.precomputed[yr] = {
                "values": values,
                "table":  table
            }
