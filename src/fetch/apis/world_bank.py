import requests
from src.fetch.apis.base_api import BaseAPI


class WorldBankAPI(BaseAPI):
    BASE_URL = "http://api.worldbank.org/v2"

    def __init__(self, indicators, standardization, data_path):
        super().__init__(indicators, standardization, data_path, "world_bank")

    def fetch_data(self, indicator, indicator_name, start_year, end_year):
        url = f"{self.BASE_URL}/country/all/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=20000"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()[1]  # The actual data is in the second element of the response JSON
                return data
            except IndexError:
                print(f"No data found for indicator: {indicator_name} (ID: {indicator})")
                return None
        else:
            print(f"Failed to fetch data for indicator: {indicator_name} (ID: {indicator})")
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
