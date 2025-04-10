import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from src.utils import get_data_dir


class Dataset:

    """High-level class for Dataset Handling"""

    def __init__(self, data_dir: Path = get_data_dir()):

        self.data_dir = data_dir
        self.csv_paths = self._get_csv_list()

        self._countries_list = None
        self._overall_start_date = None
        self._overall_end_date = None


    def get_datadict(
            self, 
            dataset_names: Optional[List[str]] = None,
            ) -> Dict[str, pd.DataFrame]:
        
        """
        Args:
            dataset_names: List of dataset names
            start_date: str "YYYY-MM-DD"
            end_date:   str "YYYY-MM-DD"
        
        Returns:
            dict of datasets: {<dataset_name>: Dataframe}
        """
        datasets = {}

        if dataset_names is None:
            dataset_names = self.get_dataset_names()

        for dataset_name in dataset_names:
            try:
                csv_path = self._get_csv_from_name(dataset_name)
                dataframe = self._load_dataset(csv_path)
                dataframe = self._clean_dataframe(dataframe)
                datasets[dataset_name] = dataframe
            except Exception as e:
                print(f"dataset name: {dataset_name} not in path. Skipping...")
        
        return datasets
    
    def get_ml_ready(self) -> pd.DataFrame:
        """
        Returns a merged long-format DataFrame with all datasets joined on (date, country),
        inserting NaN where data is missing.
        """
        datasets = self.get_datadict()
        long_dfs = []

        for name, df in datasets.items():
            df = df.copy()

            df = df.reset_index()

            # Melt wide → long
            df_long = df.melt(
                id_vars="date",
                var_name="country",
                value_name=name  # Feature name from dataset
            )

            long_dfs.append(df_long)

        # Merge all datasets on ['date', 'country'] using outer join
        from functools import reduce
        ml_df = reduce(lambda left, right: pd.merge(left, right, on=["date", "country"], how="outer"), long_dfs)

        ml_df = ml_df.sort_values(by=["date", "country"]).reset_index(drop=True)

        return ml_df



    def get_dataset_names(self) -> List[str]:
        """Return list of dataset names"""
        dataset_names = []
        for csv_path in self.csv_paths:
            dataset_name = csv_path.name.split(".csv")[0]
            dataset_names.append(dataset_name)
        return dataset_names


    def get_category_names(self) -> List[str]:
        """Return list o categories names"""
        categories_list = []
        for csv_path in self.csv_paths:
            csv_dir = csv_path.parent.name
            if csv_dir not in categories_list:
                categories_list.append(csv_dir)
        return categories_list


    def _load_dataset(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df = df.astype('float32')
        return df

    def _clean_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe

    def _get_csv_from_name(self, dataset_name) -> Path:
        for path in self.csv_paths:
            if path.name.split(".csv")[0] == dataset_name:
                return path
        return None

    def _get_csv_list(self) -> List[Path]:
        """Return list of csv paths in data dir"""
        csv_list = []
        for file in self.data_dir.rglob("*.csv"):
            csv_list.append(Path(file))
        return csv_list
    


if __name__ == "__main__":
    dataset = Dataset()

    ml_df = dataset.ml_ready()

    print(ml_df)