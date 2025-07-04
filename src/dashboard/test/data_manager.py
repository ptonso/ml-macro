import zipfile
import tempfile
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set

from src.dashboard.config import WorldTimeConfig
from src.dashboard.utils import safe_log_transform, get_iso3
from src.preprocess.dataset import Dataset
from src.preprocess.result import ResultData

class DataManager:
    def __init__(self, cfg: WorldTimeConfig):
        self.cfg = cfg
        self.dataset = Dataset()
        res = self.dataset.get(ResultData(datadict=True, metadata=True))
        self._all_raw: Dict[str, pd.DataFrame] = res.datadict or {}
        self.category_dict: Dict[str, str] = res.metadata.category_dict
        self.world: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.load_shapefile()
        self.years: List[str] = []
        self.precomputed: Dict[str, Dict[str, Any]] = {}
        self.country_data_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def load_shapefile(self) -> None:
        shp_dir = self.cfg.shapefile_dir / "ne_110m_admin_0_countries"
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
        raw = self._all_raw.get(indicator)
        if raw is None:
            raise ValueError(f"No data for indicator '{indicator}'")
        
        df = (
            raw.reset_index()
               .melt(id_vars="date", var_name="country", value_name="value")
               .assign(iso_a3=lambda d: d["country"].map(get_iso3))
        )
        
        df["display_name"] = df["country"].apply(lambda x: x.title() if isinstance(x, str) else x)
        
        # Apply the safe_log_transform to all values
        df["log_value"] = df["value"].apply(safe_log_transform)
        
        # Get non-NaN log values for normalization
        non_nan_logs = df["log_value"].dropna()
        
        if len(non_nan_logs) > 0:
            deciles = np.nanpercentile(non_nan_logs, np.arange(0, 101, 10))
            
            def assign_decile(row):
                value = row["log_value"]
                
                if pd.isna(value):
                    return np.nan
                
                decile_index = np.searchsorted(deciles, value, side='right') - 1
                return min(max(decile_index, 0), 9) / 9.0
            
            df["normalized_value"] = df.apply(assign_decile, axis=1)
        else:
            df["normalized_value"] = df["value"].apply(
                lambda x: 0.5 if not pd.isna(x) else np.nan
            )

        years_int = sorted({pd.to_datetime(d).year for d in raw.index})
        self.years = [str(y) for y in years_int]

        self.precomputed.clear()
        self.country_data_cache.clear()
        
        for yr in self.years:
            yint = int(yr)
            dfy = df[df["date"].dt.year==yint]
            
            df_geo = dfy[["iso_a3", "normalized_value", "value", "display_name"]]
            
            merged = self.world.merge(
                df_geo,
                left_on="ADM0_A3", right_on="iso_a3",
                how="left"
            )
            
            normalized_values = merged["normalized_value"].fillna(np.nan).tolist()
            raw_values = merged["value"].fillna(np.nan).tolist()
            display_names = merged["display_name"].fillna("Unknown").tolist()
            
            table = dfy.sort_values("value", ascending=False)
            
            self.precomputed[yr] = {
                "values": normalized_values,
                "raw_values": raw_values,
                "display_names": display_names,
                "table": table
            }
    
    def get_country_data(self, country: str, indicator: str) -> Tuple[List[str], List[float]]:
        cache_key = f"{country}_{indicator}"
        if cache_key in self.country_data_cache:
            return self.country_data_cache[cache_key]["years"], self.country_data_cache[cache_key]["values"]
            
        yrs, vals = [], []
        for yr in self.years:
            tbl = self.precomputed.get(yr, {}).get("table")
            if tbl is None:
                continue
                
            row = tbl[tbl["country"] == country]
            if not row.empty:
                yrs.append(yr)
                vals.append(float(row["value"].iloc[0]))
                
        self.country_data_cache[cache_key] = {
            "years": yrs,
            "values": vals
        }
        
        return yrs, vals
    
    def get_filtered_countries(self, g20_only: bool = True) -> List[str]:
        G20_COUNTRIES = [
            "Argentina", "Australia", "Brazil", "Canada", "China", 
            "France", "Germany", "India", "Indonesia", "Italy", 
            "Japan", "Mexico", "Russia", "Saudi Arabia", "South Africa", 
            "South Korea", "Turkey", "United Kingdom", "United States", "European Union"
        ]
        
        all_countries = set()
        for yr in self.years:
            tbl = self.precomputed.get(yr, {}).get("table")
            if tbl is not None:
                countries = tbl["country"].unique().tolist()
                all_countries.update(countries)
        
        countries_list = list(all_countries)
        
        if g20_only:
            countries_list = [c for c in countries_list 
                             if c.lower() in [g.lower() for g in G20_COUNTRIES]]
            
        countries_list.sort()
        return countries_list