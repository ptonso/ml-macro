import requests
from src.fetch.apis.base_api import BaseAPI
from src.logger import setup_logger

class IMFAPI(BaseAPI):
    source_key = "imf"
    BASE_URL = "https://www.imf.org/external/api"

    def __init__(self, indicators, data_path):
        super().__init__(indicators, data_path)

    def fetch_data(self, indicator, indicator_name, start_year, end_year):
        url = f"{self.BASE_URL}/data/{indicator}?start={start_year}&end={end_year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()['data']  # Adjust based on actual JSON structure
                return data
            except KeyError:
                self.logger.error(f"No data found for indicator: {indicator_name} (ID: {indicator})")
                return None
        else:
            self.logger.error(f"Failed to fetch data for indicator: {indicator_name} (ID: {indicator})")
            return None

    def get_time_series(self, start_year, end_year):
        all_data = {}
        for indicator_id, api_indicator_id in self.ensemble_api_indicators(self.api_name).items():
            data = self.fetch_data(api_indicator_id, indicator_id, start_year, end_year)
            if data:
                processed_data = self.process_data(data)
                if processed_data is not None:
                    all_data[indicator_id] = processed_data
        return all_data
