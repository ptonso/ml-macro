# src/dashboard/config.py
from pathlib import Path
from dataclasses import dataclass

@dataclass
class WorldTimeConfig:
    shapefile_dir: Path
    year: str
    title: str = "Dashboard"
    log_file: str = "api.log"
