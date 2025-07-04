import numpy as np
import pandas as pd
from typing import Dict, Any

def health_check_datadict(datadict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Runs a suite of checks on your dict of DataFrames.

    Returns a dict containing:
      - start_dates, end_dates: per-feature min/max index
      - columns: set of columns per feature
      - index_dtype: inferred dtype of each index
      - duplicate_index: count of duplicate index entries per feature
      - duplicate_columns: count of duplicate column names per feature
      - non_numeric: count of entries that aren’t int/float/NaN
      - summary: high-level pass/fail for the first two checks
    """
    results = {
        "start_dates": {},
        "end_dates": {},
        "columns": {},
        "index_dtype": {},
        "duplicate_index": {},
        "duplicate_columns": {},
        "non_numeric": {},
        "summary": {}
    }

    # 1) Gather per‐feature stats
    for feature, df in datadict.items():
        # — ensure index is datetime if possible
        idx = df.index
        if not pd.api.types.is_datetime64_any_dtype(idx):
            try:
                idx = pd.to_datetime(idx, errors="raise", format="%Y")
                results["index_dtype"][feature] = "datetime (year)"
            except Exception:
                results["index_dtype"][feature] = f"{type(idx).__name__}"
        else:
            results["index_dtype"][feature] = "datetime"

        # record start / end
        results["start_dates"][feature] = idx.min()
        results["end_dates"][feature]   = idx.max()

        # record column set
        colset = set(df.columns)
        results["columns"][feature] = colset

        # duplicate checks
        results["duplicate_index"][feature]   = int(idx.duplicated().sum())
        results["duplicate_columns"][feature] = int(df.columns.duplicated().sum())

        # non‐numeric checks
        # anything that is not NaN and not an int/float
        def _is_bad(x):
            return not (pd.isna(x) or isinstance(x, (int, float, np.floating, np.integer)))
        bad_count = df.map(_is_bad).sum().sum()
        results["non_numeric"][feature] = int(bad_count)

    # 2) Summary for uniformity
    unique_starts = set(results["start_dates"].values())
    unique_ends   = set(results["end_dates"].values())
    all_cols      = list(results["columns"].values())

    same_dates = (len(unique_starts) == 1 and len(unique_ends) == 1)
    same_cols  = all(colset == all_cols[0] for colset in all_cols)

    results["summary"]["uniform_start_end"]   = same_dates
    results["summary"]["uniform_columns"]     = same_cols
    if not same_dates:
        results["summary"]["mismatched_starts"] = unique_starts
        results["summary"]["mismatched_ends"]   = unique_ends
    if not same_cols:
        # show which features differ
        common = set.intersection(*all_cols)
        results["summary"]["extra_columns"]   = {
            f: results["columns"][f] - common for f in datadict
        }
        results["summary"]["missing_columns"] = {
            f: common - results["columns"][f] for f in datadict
        }

    return results

def print_health_anomalies(report):
    # 1) start/end date mismatches
    if not report["summary"]["uniform_start_end"]:
        ss = report["summary"]["mismatched_starts"]
        es = report["summary"]["mismatched_ends"]
        print(f"❌ Date range mismatch:")
        print(f"   starts: {ss}")
        print(f"   ends:   {es}")

    # 2) column mismatches
    if not report["summary"]["uniform_columns"]:
        extra = report["summary"]["extra_columns"]
        missing = report["summary"]["missing_columns"]
        print(f"❌ Column set mismatch:")
        for feature in extra:
            if extra[feature]:
                print(f"   {feature!r} has extra cols:   {extra[feature]}")
            if missing[feature]:
                print(f"   {feature!r} missing cols:     {missing[feature]}")

    # 3) duplicate‐index or duplicate‐column anomalies
    for feature, dup_idx in report["duplicate_index"].items():
        if dup_idx > 0:
            print(f"❌ {feature!r}: {dup_idx} duplicate index entries")
    for feature, dup_col in report["duplicate_columns"].items():
        if dup_col > 0:
            print(f"❌ {feature!r}: {dup_col} duplicate column names")

    # 4) non‐numeric entries
    for feature, bad in report["non_numeric"].items():
        if bad > 0:
            print(f"❌ {feature!r}: {bad} non-numeric cells")

