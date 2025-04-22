import requests
from src.fetch.apis.base_api import BaseAPI
from src.logger import setup_logger


class AlphaVantageAPI(BaseAPI):
    source_key = "alpha_vantage"
    BASE_URL = "https://www.alphavantage.co/query"
    API_KEY = "your_alpha_vantage_api_key"  # Replace with your Alpha Vantage API key

    def __init__(self, indicators, data_path):
        self.api_name = "alpha_vantage"
        super().__init__(indicators, data_path)

    def fetch_data(self, indicator, indicator_name, start_year, end_year):
        url = f"{self.BASE_URL}?function={indicator}&apikey={self.API_KEY}&datatype=json"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()['Time Series (Daily)']  # Adjust based on actual JSON structure
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
