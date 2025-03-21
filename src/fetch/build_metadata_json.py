import json
import os
from src.utils import get_metadata_dir

def main():
    jsonfile = get_metadata_dir() / "metadata.json"
    create_metadata(jsonfile)


def create_metadata(jsonfile: str):
    metadata = {
        "eua": ["united states", "united states of america", "usa", "america"],
        "brasil": ["brazil", "brasil"],
        "china": ["china", "people's republic of china", "prc"],
        "india": ["india", "republic of india"],
        "south_africa": ["south africa", "republic of south africa"]
        # Add more standardized names and their alternatives here
    }

    with open(jsonfile, 'w') as file:
        json.dump(metadata, file, indent=4)

    print("Metadata JSON file created successfully.")

if __name__ == "__main__":
    main()
