import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute       import IterativeImputer
from sklearn.ensemble     import HistGradientBoostingRegressor


def input_regression_by_group(
    df: pd.DataFrame,
    group_col: str = 'country',
    random_state: int = 123,
    max_iter: int = 10
) -> pd.DataFrame:
    """
    expect df as ml_ready tables:
    date, country, feature1, feature2
    ...
    where we apply date X country cross product.


    Impute each country's time‐series separately, but first ensure
    no feature is entirely missing in that country by pre‐filling
    with the global mean.

    - Leaves 'date' and `group_col` untouched.
    - Imputes all other columns via IterativeImputer + HGBRegressor.
    """
    non_num_cols = ['date', group_col]
    # All numeric columns
    numeric = df.drop(columns=non_num_cols)
    # Compute global means (used to pre‐fill group‐wide NaNs)
    global_means = numeric.mean()

    imputed_parts = []
    for _, grp in df.groupby(group_col, sort=False):
        grp = grp.sort_values('date')
        ids = grp[non_num_cols]
        nums = grp.drop(columns=non_num_cols).copy()

        # 1) pre-fill any column that is ALL-NaN in this group
        all_missing = nums.columns[nums.isna().all()]
        for col in all_missing:
            nums[col] = global_means[col]

        # 2) now all columns have at least one value → safe to impute
        imp = IterativeImputer(
            estimator=HistGradientBoostingRegressor(),
            random_state=random_state,
            max_iter=max_iter,
            initial_strategy='mean'
        )
        nums_imp = pd.DataFrame(
            imp.fit_transform(nums),
            columns=nums.columns,
            index=grp.index
        )

        imputed_parts.append(pd.concat([ids, nums_imp], axis=1))

    result = pd.concat(imputed_parts).sort_index()
    return result[df.columns]

