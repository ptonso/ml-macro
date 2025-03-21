# re download everything
import os
import sys
import subprocess
from typing import Dict

from src.logger import setup_logger
from src.utils import get_data_dir, get_project_dir

logger = setup_logger("script_runner.log")

def main():
    project_root = get_project_dir()
    sys.path.insert(0, project_root)

    scripts = [
    "build_indicator_json",
    "build_metadata_json",
    "build_standarization_json",
    "download_macro",
    ]

    build_data_dir()

    for script in scripts:
        logger.info(f"[INFO] Running {script}.py...")

        process = subprocess.Popen(
            ["python3", "-m", f"src.fetch.{script}"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, 
            )
        
        for line in iter(process.stdout.readline, ''):
            cleaned_line = ' - '.join(line.split(' - ')[1:]).strip()
            logger.info(f"[INFO] - [{script}] - {cleaned_line}")

        for line in iter(process.stderr.readline, ''):
            cleaned_line = ' - '.join(line.split(' - ')[1:]).strip()
            logger.error(f"[ERROR] - [{script}] - {cleaned_line}")
        
        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()

        if return_code == 0:
            logger.info(f"[SUCCESS] {script} - Completed successfully.")
        else:
            logger.error(f"[FAIL] {script} - Failed with exit code {return_code}")
            break


def build_data_dir():
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    structure : Dict[str, list] = {
        "00--raw" : ["macro"],
        "01--clean" : [],
        "10--metadata": [],
        "11--manual": [],
    }
    
    for parent, child_list in structure.items():
        parent_path = data_dir / parent
        os.makedirs(parent_path, exist_ok=True)

        for child in child_list:
            child_path = parent_path / child
            os.makedirs(child_path, exist_ok=True)


if __name__ == "__main__":
    main()