import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


DPI:  int = 150
CMAP: str = "cividis" # ["viridis", "cividis", "magma", "inferno", "YlGnBu", "rocket"]

def compute_feature_yearly_missing(
    datadict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Returns a DataFrame M with
      – index = feature names
      – columns = sorted years (ints)
      – M.loc[f, y] = proportion of missing values in feature f for year y
                     (averaged over all countries).
    """
    sample_df = next(iter(datadict.values()))
    if not pd.api.types.is_datetime64_any_dtype(sample_df.index):
        try:
            sample_df.index = pd.to_datetime(sample_df.index, format="%Y")
        except:
            sample_df.index = pd.to_datetime(sample_df.index)
    years = sorted(sample_df.index.year.unique())

    heatmap = pd.DataFrame(index=datadict.keys(), columns=years, dtype=float)

    for feature, df in datadict.items():
        # copy and fix index
        df2 = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df2.index):
            try:
                df2.index = pd.to_datetime(df2.index, format="%Y")
            except:
                df2.index = pd.to_datetime(df2.index)

        missing_flags = df2.isna()
        yearly_country = missing_flags.groupby(df2.index.year).mean()
        heatmap.loc[feature, yearly_country.index] = yearly_country.mean(axis=1).values
    return heatmap



def plot_feature_missing_heatmap(heatmap: pd.DataFrame):
    plt.figure(
        figsize=(len(heatmap.columns)*0.5+1, len(heatmap.index)*0.25+1),
        dpi=DPI
    )
    img = plt.imshow(heatmap.values, aspect="auto", origin="lower", cmap=CMAP)
    plt.xticks(np.arange(len(heatmap.columns)), heatmap.columns, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(heatmap.index)), heatmap.index, fontsize=8)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.title("Missingness by Feature & Year", fontsize=12)
    plt.colorbar()
    plt.tight_layout(pad=0.5)
    plt.show()


def plot_country_missing_for_feature(
    datadict: Dict[str, pd.DataFrame],
    feature: str
):
    """
    For a single feature name, plots a binary‐map of missingness:
      – x axis = year
      – y axis = country
      – cell = 1 if missing, 0 if present
    """
    if feature not in datadict:
        raise KeyError(f"Feature {feature!r} not in your datadict")

    df = datadict[feature].copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, format="%Y")
        except:
            df.index = pd.to_datetime(df.index)

    mat = df.isna().T.astype(int)
    years = mat.columns.year if hasattr(mat.columns, "year") else mat.columns
    countries = mat.index

    plt.figure(
        figsize=(len(years) * 0.3 + 2,
                len(countries) * 0.2 + 2),
                dpi=DPI
                )
    img = plt.imshow(mat.values, aspect="auto", origin="lower", cmap=CMAP)
    plt.xticks(np.arange(len(years)), years, rotation=90)
    plt.yticks(np.arange(len(countries)), countries)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.title(f"Missingness for feature {feature!r}", fontsize=12)
    plt.colorbar()
    plt.tight_layout(pad=0.5)
    plt.show()



def compute_country_yearly_missing(
    datadict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Returns a DataFrame C with
      – index = country names
      – columns = sorted years (ints)
      – C.loc[c, y] = proportion of features that are missing
                     for country c in year y.
    """
    per_feature = []
    for df in datadict.values():
        df2 = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df2.index):
            try:
                df2.index = pd.to_datetime(df2.index, format="%Y")
            except:
                df2.index = pd.to_datetime(df2.index)
        missing = df2.isna()
        yearly = missing.groupby(df2.index.year).mean()
        per_feature.append(yearly)

    total = sum(per_feature)
    avg    = total / len(per_feature)

    heatmap = avg.T

    years     = sorted(avg.index)
    countries = sorted(heatmap.index)
    heatmap   = heatmap.loc[countries, years]

    return heatmap


def plot_country_missing_heatmap(heatmap: pd.DataFrame):
    """
    Given the DataFrame from compute_country_yearly_missing,
    plots a heatmap with
      – x axis = year
      – y axis = country
      – color = proportion of features missing
    """
    plt.figure(
        figsize=(
            len(heatmap.columns) * 0.5 + 2,
            len(heatmap.index)   * 0.2 + 2,
        ),
        dpi=DPI
    )
    data = heatmap.values.astype(float)
    img  = plt.imshow(data, aspect="auto", origin="lower", cmap=CMAP)
    plt.xticks(
        np.arange(len(heatmap.columns)),
        heatmap.columns,
        rotation=90
    )
    plt.yticks(
        np.arange(len(heatmap.index)),
        heatmap.index
    )

    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Country", fontsize=10)
    plt.title("Proportion of Features Missing\nby Country and Year", fontsize=12)
    plt.tight_layout(pad=0.5)
    plt.colorbar()
    plt.show()


