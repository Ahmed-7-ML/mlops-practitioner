# Feature Engineering

import pandas as pd
from typing import List
from sklearn.feature_extraction import DictVectorizer

categorical_features: List[str] = ["PULocationID", "DOLocationID", "store_and_fwd_flag"]


def engineer_features(
    df: pd.DataFrame, dv: DictVectorizer = None, fit: bool = False
) -> (pd.DataFrame, DictVectorizer):
    """
    Transform Categorical Features using DictVectorizer.
    Returns (X, dv).
    """
    # Take a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    df[categorical_features] = df[categorical_features].astype(str)
    dicts = df[categorical_features].to_dict(orient="records")

    if fit:
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(dicts)
    else:
        if dv is None:
            raise ValueError("DictVectorizer must be provided when fit=False.")
        X = dv.transform(dicts)
    return X, dv
