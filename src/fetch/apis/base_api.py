import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict


class BaseAPI:
    def __init__(
            self, 
            indicators: Dict, 
            standardization: Dict, 
            data_path: Path, 
            api_name: str
            ):
        self.indicators = indicators
        self.standardization = standardization
        self.api_name = api_name
        self.data_path = data_path

    def ensemble_api_indicators(self, api_name):
        api_indicators = {}
        for category, indicators in self.indicators.items():
            for indicator_id, indicator_info in indicators.items():
                if api_name in indicator_info:
                    api_indicators[indicator_id] = indicator_info[api_name]
        return api_indicators

    def standardize_country_name(self, country):
        for standard_name, variants in self.standardization['countries'].items():
            if country.lower() in [v.lower() for v in variants]:
                return standard_name
        return country.lower()

    def fetch_data(self, indicator, indicator_name, start_year, end_year):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_time_series(self, start_year, end_year):
        raise NotImplementedError("This method should be implemented by subclasses")

    def process_data(self, data):
        df = pd.DataFrame(data)
        if {'country', 'date', 'value'}.issubset(df.columns):
            df = df[['country', 'date', 'value']].dropna()
            df['country'] = df['country'].apply(lambda x: self.standardize_country_name(x['value']))
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date', columns='country', values='value')
            df = df.sort_index()
            return df
        else:
            print("Expected columns are missing in the data.")
            print("Available columns:", df.columns)
            return None

    def save_to_csv(self, data, start_year):
        for indicator, df in data.items():
            df = df[df.index.year >= start_year]
            category = next((cat for cat, indics in self.indicators.items() if indicator in indics), "misc")
                        
            csv_path = self.data_path / category / f"{indicator}_{self.api_name}.csv"
            os.makedirs(csv_path.parent, exist_ok=True)
            df.to_csv(csv_path, index_label="date")


    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as file:
            return json.load(file)

    def create_csv_for_indicators(self, start_year, end_year, indicator_list=None):
        if indicator_list is None:
            indicator_list = list(self.ensemble_api_indicators(self.api_name).keys())
        all_data = {}
        for indicator_id in indicator_list:
            api_indicator_id = self.ensemble_api_indicators(self.api_name).get(indicator_id)
            if api_indicator_id:
                data = self.fetch_data(api_indicator_id, indicator_id, start_year, end_year)
                if data:
                    processed_data = self.process_data(data)
                    if processed_data is not None:
                        all_data[indicator_id] = processed_data
        self.save_to_csv(all_data, start_year)
