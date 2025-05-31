import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from src.fetch.apis.base_api import BaseAPI
from src.logger import setup_logger

class FAOAPI(BaseAPI):
    """
    Fetch FAO data for crops & livestock via the FENIX JSON POST endpoint,
    filtering each commodity code (e.g. 'POTA') over all countries and years.
    """
    source_key = "fao"
    API_BASE = "https://faostat3.fao.org"      # ← use HTTPS, not HTTP
    DOMAIN_CODE = "QC"                         # Crops & Livestock domain

    def __init__(
        self,
        indicators: Dict[str, Any],
        data_path: Path
    ) -> None:
        super().__init__(indicators, data_path)
        self.logger = setup_logger(category=self.api_name)

    def create_csv_for_indicators(
        self,
        start_year: int,
        end_year: int,
        indicator_list: Optional[List[str]] = None
    ) -> None:
        if indicator_list is None:
            # gather only those with a "fao" key
            indicator_list = [
                ind
                for category in self.indicators.values()
                for ind, info in category.items()
                if self.source_key in info
            ]

        all_data: Dict[str, pd.DataFrame] = {}
        for ind in indicator_list:
            code = self._ensemble_api_indicators().get(ind)
            if not code:
                continue
            df = self.fetch_data(code, ind, start_year, end_year)
            if df is not None and not df.empty:
                df = self._standardize_columns(df)
                all_data[ind] = df

        self.logger.info(f"Fetched data for {len(all_data)} FAO commodities.")
        self._save_to_csv(all_data, start_year)

    def fetch_data(
        self,
        indicator: str,
        indicator_name: str,
        start_year: int,
        end_year: int
    ) -> Optional[pd.DataFrame]:
        item_code = indicator.split(".")[-1]
        years = [str(y) for y in range(start_year, end_year + 1)]
        payload = {
            "datasource": "faostatdb",
            "domainCode": self.DOMAIN_CODE,
            "lang": "E",
            "list1Codes": [item_code],  # item code like 'POTA'
            "list2Codes": ["all"],      # all countries
            "list3Codes": [],
            "list4Codes": [],
            "list5Codes": years,
            "nullValues": False,
            "limit": 0
        }

        try:
            resp = requests.post(
                f"{self.API_BASE}/wds/rest/procedures/data",
                json={"payload": json.dumps(payload)},
                timeout=30
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"FAO data request for '{indicator_name}' failed: {e}")
            return None

        js = resp.json()
        records = js.get("value") or js.get("data") or []
        if not records:
            self.logger.error(f"No records for {indicator_name} in {start_year}-{end_year}")
            return None

        df = pd.DataFrame(records)
        df = df.rename(columns={
            "geoAreaName": "country",
            "year": "date",
            "value": indicator_name
        })[["country", "date", indicator_name]]
        df["date"] = pd.to_datetime(df["date"], format="%Y")
        pivot = df.pivot(index="date", columns="country", values=indicator_name)
        return pivot.sort_index()
