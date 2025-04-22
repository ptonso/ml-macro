import os
import json
import inspect
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.utils import get_metadata_dir
from src.logger import setup_logger


class BaseAPI:
    def __init__(
            self, 
            indicators: Dict,
            data_path: Path
            ):

        self.api_name = type(self).__name__
        script_path = (type(self).__module__).replace(".", "/")

        self.logger = setup_logger(
            script_path=script_path,
            category=self.api_name
            )

        self.indicators = indicators
        self.data_path = data_path
        self.countries = self._load_countries()

        self.alias_map: Dict[str, str] = {
            alias.lower(): std_name
            for std_name, meta in self.countries.items()
            for alias in meta.get("aliases", [])
        }


    def create_csv_for_indicators(
            self, 
            start_year: int, 
            end_year: int, 
            indicator_list: Optional[list[str]] = None
        ) -> None:
        """
        Create CSV files for the specified indicators.
        :param indicator_list: List of indicator IDs to fetch data for. If None, all indicators will be fetched.
        """
    
        if indicator_list is None:
            indicator_list = list(self._ensemble_api_indicators().keys())
        all_data = {}
        for indicator_id in indicator_list:
            api_indicator_id = self._ensemble_api_indicators().get(indicator_id)
            if api_indicator_id:
                data = self.fetch_data(api_indicator_id, indicator_id, start_year, end_year)
                if data:
                    processed_data = self._process_data(data)
                    if processed_data is not None:
                        all_data[indicator_id] = processed_data
        
        self.logger.info(f"Fetched data for {len(all_data)} indicators.")
        self._save_to_csv(all_data, start_year)


    def _ensemble_api_indicators(self):
        api_indicators: Dict[str, str] = {}
        key = getattr(self, "source_key", self.api_name)
        for category, indicators in self.indicators.items():
            for indicator_id, indicator_info in indicators.items():
                if key in indicator_info:
                    api_indicators[indicator_id] = indicator_info[key]
        return api_indicators

    def _load_countries(self):
        countries_json = get_metadata_dir() / "countries.json"
        with open(countries_json, 'r') as file:
            return json.load(file)


    def _standardize(self, country: str) -> str:
        lower = country.lower()
        for std_name, variants in self.countries.items():
            if lower in (v.lower() for v in variants):
                return std_name
        return lower

    def fetch_data(self, indicator, indicator_name, start_year, end_year):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_time_series(self, start_year, end_year):
        raise NotImplementedError("This method should be implemented by subclasses")

    def _process_data(self, data):
        df = pd.DataFrame(data)
        if {'country', 'date', 'value'}.issubset(df.columns):
            df = df[['country', 'date', 'value']].dropna()
            df['country'] = df['country'].apply(lambda x: self._standardize(x['value']))
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date', columns='country', values='value')
            df = df.sort_index()
            return df
        else:
            self.logger.error("Expected columns are missing in the data.")
            self.logger.error("Available columns:", df.columns)
            return None
        

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        to_rename: Dict[str, str] = {}
        to_drop: list[str] = []

        for col in df.columns:
            std = self.alias_map.get(col.lower())
            if std:
                to_rename[col] = std
            else:
                to_drop.append(col)

        if to_rename:
            df = df.rename(columns=to_rename)
        to_drop.append("regions")
        if to_drop:
            df = df.drop(columns=to_drop)
            self.logger.warning(f"Dropped {len(to_drop)} columns: {to_drop}")
        return df

    def _save_to_csv(self, data, start_year):
        for indicator, df in data.items():
            df = df[df.index.year >= start_year]
            category = next((cat for cat, indics in self.indicators.items() if indicator in indics), "misc")
                        
            api_name = getattr(self, "source_key", self.api_name)
            csv_path = self.data_path / category / f"{indicator}_{api_name}.csv"
            os.makedirs(csv_path.parent, exist_ok=True)
            df.to_csv(csv_path, index_label="date")
