# src/dashboard/utils.py
import pandas as pd
import numpy as np
import pycountry
from typing import List, Tuple, Optional

def format_option_name(name: str) -> str:
    if name.endswith("_world_bank"):
        name = name[:-11]
    return name.replace("_", " ").replace("-", " ").title()

def _hex_to_rgb(hex_color: str) -> Tuple[int,int,int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def make_blue_red_transparent_palette(
    blue_hex: str = "#324376",
    red_hex: str = "#6E0D25",
    steps: int = 10
) -> List[str]:
    half = steps // 2
    br, bg, bb = _hex_to_rgb(blue_hex)
    rr, rg, rb = _hex_to_rgb(red_hex)
    palette: List[str] = []
    for i in range(half):
        alpha = 1.0 - (i / (half - 1)) * 0.8 if half > 1 else 1.0
        palette.append(f"rgba({br},{bg},{bb},{alpha:.2f})")
    for i in range(half):
        alpha = 0.2 + (i / (half - 1)) * 0.8 if half > 1 else 0.2
        palette.append(f"rgba({rr},{rg},{rb},{alpha:.2f})")
    return palette

def safe_log_transform(value: float) -> float:
    if pd.isna(value):
        return np.nan
    if value > 0:
        return np.log1p(value)
    if value < 0:
        return -np.log1p(abs(value))
    return 0.0

def get_iso3(name: str) -> Optional[str]:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None
