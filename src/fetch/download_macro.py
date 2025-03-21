import json
from src.utils import get_metadata_dir, get_data_dir

from src.fetch.apis.world_bank import WorldBankAPI
from src.fetch.apis.imf import IMFAPI
from src.fetch.apis.oecd import OECDAPI
from src.fetch.apis.alpha_vantage import AlphaVantageAPI

from src.logger import setup_logger

if __name__ == "__main__":

    logger = setup_logger("download_macro.log")

    metadata_dir = get_metadata_dir()
    indicator_json = metadata_dir / "indicator.json"
    standardization_json = metadata_dir / "standardization.json"

    output_path = get_data_dir() / "00--raw" / "macro"

    with open(indicator_json, 'r') as file:
        indicators = json.load(file)
    
    with open(standardization_json, 'r') as file:
        standardization = json.load(file)

    start_year = 1900
    end_year = 2030

    api_class_list = [WorldBankAPI, IMFAPI, OECDAPI, AlphaVantageAPI]

    for api_class in api_class_list:
        logger.info(f"Starting download from {api_class.__name__}")
        try:
            api = api_class(indicators, standardization, data_path=output_path)
            api.create_csv_for_indicators(start_year, end_year)
        except Exception as e:
            logger.error(f"downloading from {api_class.__name__}: {e}")

    logger.info(f"data fetching and saving completed.")
