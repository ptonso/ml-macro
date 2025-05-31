import sys
import json
import traceback
from src.utils import get_metadata_dir, get_data_dir

from src.fetch.apis.world_bank import WorldBankAPI
from src.fetch.apis.imf import IMFAPI
from src.fetch.apis.fao import FAOAPI
from src.fetch.apis.oecd import OECDAPI
from src.fetch.apis.alpha_vantage import AlphaVantageAPI

from src.logger import setup_logger


logger = setup_logger()

def main():
    metadata_dir = get_metadata_dir()
    
    start_year = 1900
    end_year = 2030

    indicators_json = metadata_dir / "indicators.json"
    with open(indicators_json, 'r') as file:
        indicators = json.load(file)

    output_path = get_data_dir() / "00--raw" / "macro"
    api_classes = [WorldBankAPI, IMFAPI, OECDAPI] # , FAOAPI]

    for API in api_classes:
        logger.start(f"download from {API.__name__}")
        try:
            api = API(indicators, data_path=output_path)
            api.create_csv_for_indicators(start_year, end_year)
            
            logger.success(f"download data from {API.__name__}.")
        except Exception as e:
            logger.error(f"downloading from {API.__name__}:")
            logger.error(traceback.format_exc())
            sys.exit(1)

    logger.success(f"data fetching and saving completed.")



if __name__ == "__main__":
    main()



    
