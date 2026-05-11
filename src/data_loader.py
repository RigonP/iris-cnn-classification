"""
I use only dataset/iris.data (provided on Moodle, which contains missing
values encoded as 'NaN' or '?') as my single data source. I import this
module from all my other scripts instead of using sklearn.datasets.load_iris()
or any pre-cleaned file.

I impute the missing values with the column-wise median, computed exclusively
from iris.data itself, so I never reference any external dataset.
"""
import os
import numpy as np
import pandas as pd

# I keep these canonical names here so all my scripts stay consistent.
FEATURE_NAMES = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)',
]
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']
CLASS_MAPPING = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
}

# I resolve the path relative to this file so the loader works from any directory.
_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'dataset', 'iris.data'
)


def load_iris_dataframe(path: str = _DATASET_PATH) -> pd.DataFrame:
    """I load iris.data, treat 'NaN' and '?' as missing, impute with the
    column-wise median, and return a clean DataFrame with the 'class' label
    and an integer 'target' column."""

    # I tell pandas to treat both 'NaN' and '?' as missing values.
    columns = FEATURE_NAMES + ['class']
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        na_values=['?', 'NaN'],
        skip_blank_lines=True,
    )

    df = df.dropna(subset=['class']).reset_index(drop=True)

    # I count missing values so I can report them, then replace each NaN with
    # the column-wise median computed only from iris.data itself.
    n_missing_before = int(df[FEATURE_NAMES].isna().sum().sum())
    df[FEATURE_NAMES] = df[FEATURE_NAMES].apply(
        lambda col: col.fillna(col.median())
    )

    df['target'] = df['class'].map(CLASS_MAPPING).astype(int)

    print(f"[data_loader] Loaded {len(df)} samples from {path}")
    print(f"[data_loader] Imputed {n_missing_before} missing values "
          f"with column-wise median (computed from iris.data only).")
    return df


def load_iris_arrays(path: str = _DATASET_PATH):
    """I return numpy arrays directly. I use this when I need plain arrays
    instead of a DataFrame.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4), dtype=float
    y : np.ndarray, shape (n_samples,), dtype=int   (integer-encoded targets)
    feature_names : list[str]
    target_names : list[str]
    """
    df = load_iris_dataframe(path)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df['target'].to_numpy(dtype=int)
    return X, y, list(FEATURE_NAMES), list(TARGET_NAMES)


# Smoke test I run with `python data_loader.py` to verify the loader works.
if __name__ == "__main__":
    X, y, feature_names, target_names = load_iris_arrays()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"NaN values remaining in X: {int(np.isnan(X).sum())}")
