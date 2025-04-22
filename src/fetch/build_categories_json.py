import os
import json

from src.utils import get_metadata_dir, get_data_dir
from src.logger import setup_logger

def main():

    logger = setup_logger()
    metadata_path = get_metadata_dir()
    data_dir = get_data_dir()

    os.makedirs(metadata_path, exist_ok=True)

    categories_json = metadata_path / "categories.json"

    with open(categories_json, 'w') as file:
        json.dump(categories, file, indent=4)
        logger.success(f"categories.json saved to: data/{categories_json.parent.relative_to(data_dir)}")



categories = {
    1: {
        "title": "demographic",
        "classes": ["East Asia", "Eastern Europe", "Latin America", "Middle East and North Africa", "South and South East Asia", "Sub Saharan Africa", "Western Europe", "Western offshoot"]
    },
    2: {
        "title": "income",
        "classes": ["low", "lower-middle", "upper-middle", "high"]
    }    
}

if __name__ == "__main__":
    main()