import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# give a instance with true in desire formats
@dataclass
class ResultData:
    path_names_dict: Dict[Path, str] = None # {"path/to/economic_activity.csv": "economic_activity", ...}

    datadict:   bool = False # Optional[Dict[str, pd.DataFrame]]
    ml_ready:   bool = False # Optional[pd.DataFrame]
    metadata:   bool = False # Optional["Metadata"]


@dataclass
class Metadata:
    category_dict      :  Dict[str, str]       # {"demography": "eonomic_activity", ...}
    countries          :  Dict[str, Any]       # consult `data/10--metadata/countries.json`
    indicators         :  Dict[str, Any]       # consult `data/10--metadata/indicator.json`
    overall_start_date :  str                  # <YYYY-MM-DD>
    overall_end_date   :  str                  # <YYYY-MM-DD>


