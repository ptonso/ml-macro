import pandas as pd
from typing import Dict


def clean_datadict(
    datadict: Dict[str, pd.DataFrame],
    start_year: int = 2000,
    feat_missing_thresh: float = 0.3,
    country_missing_thresh: float = 0.3
) -> Dict[str, pd.DataFrame]:
    """
    1) Keep only index.year >= start_year
    2) Drop features whose global NaN‐fraction > feat_missing_thresh
    3) Drop countries whose NaN‐fraction (across the remaining features) > country_missing_thresh
    4) For each remaining feature‐DataFrame:
         • linear interpolate each country‐series
         • then forward‐ and backward‐fill the ends
         • drop any country column still all-NaN
    5) Finally, restrict every feature‐DataFrame to the SAME set of countries
       (the intersection) so that no NaNs can reappear in ml_ready.
    """
    # 1) year filter
    filtered = {
        name: df.loc[df.index.year >= start_year]
        for name, df in datadict.items()
    }

    # 2) drop features by overall NaN‐rate
    feat_nan_frac = {
        name: df.isna().stack().mean()
        for name, df in filtered.items()
    }
    keep_feats = [n for n, frac in feat_nan_frac.items()
                  if frac <= feat_missing_thresh]
    filtered = {n: filtered[n] for n in keep_feats}

    if not filtered:
        return {}

    # 3) drop countries by overall NaN‐rate (across features)
    any_df = next(iter(filtered.values()))
    n_years = any_df.shape[0]
    n_feats = len(filtered)

    country_nan_frac = {}
    for country in any_df.columns:
        total_missing = sum(df[country].isna().sum()
                            for df in filtered.values())
        country_nan_frac[country] = total_missing / (n_years * n_feats)

    keep_countries = [c for c, frac in country_nan_frac.items()
                      if frac <= country_missing_thresh]
    filtered = {n: df.loc[:, keep_countries]
                for n, df in filtered.items()}

    # 4) interpolate + ffill + bfill + drop-all-NaN
    interp_dict = {}
    for name, df in filtered.items():
        df_i = (
            df
            .interpolate(method="linear", axis=0, limit_direction="both")
            .ffill(axis=0)
            .bfill(axis=0)
        )
        # drop any column that still has no valid data
        df_i = df_i.dropna(axis=1, how="all")
        interp_dict[name] = df_i

    # 5) align to common country set
    common_countries = set.intersection(
        *(set(df.columns) for df in interp_dict.values())
    )
    # sort for consistency
    common_countries = sorted(common_countries)

    cleaned = {
        name: df.loc[:, common_countries]
        for name, df in interp_dict.items()
    }
    return cleaned


if __name__ == "__main__":

    import pandas as pd
    from typing import *
    from src.preprocess.dataset import Dataset, DatasetConfig
    from src.preprocess.result import ResultData

    result_data = ResultData(
        datadict = True, # Optional[Dict[str, pd.DataFrame]]
        ml_ready = True, # Optional[pd.DataFrame]
        metadata = True  # Optional["Metadata"]
        )

    dataset = Dataset(DatasetConfig(
        use_raw=True
    ))

    result_data = dataset.get(result_data)

    datadict = result_data.datadict

    cleaned_dd = clean_datadict(datadict)

    for name, df in cleaned_dd.items():
        print(f"{name}: {df.isna().sum().sum()}")