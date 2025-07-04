# re download everything
import os
import sys
import importlib
import traceback

import subprocess
from typing import Dict

from src.logger import setup_logging, setup_logger
from src.utils import get_data_dir, get_project_dir

setup_logging(level="DEBUG")
logger = setup_logger()


def build_data_dir():
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    structure: Dict[str, list] = {
        "00--raw": ["macro"],
        "01--clean": [],
        "10--metadata": [],
        "11--manual": [],
    }

    for parent, children in structure.items():
        (data_dir / parent).mkdir(parents=True, exist_ok=True)
        for c in children:
            (data_dir / parent / c).mkdir(exist_ok=True)


def main():
    project_root = get_project_dir()
    sys.path.insert(0, project_root)

    build_data_dir()

    scripts = [
    "build_indicators_json",
    "build_countries_json",
    "build_categories_json",
    "download_macro",
    ]

    for script in scripts:
        logger.start(f"Running {script}.py...")

        try:
            mod = importlib.import_module(f"src.fetch.{script}")
            mod.main()
            logger.success(f"{script}.py completed successfully.")

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.fail(f"{script}.py failed: {e}")
            break



if __name__ == "__main__":
    main()