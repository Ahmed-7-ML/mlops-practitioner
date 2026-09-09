# Feature Engineering

import pandas as pd
from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from prodml.config import config


def engineer_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple:
    """
    Transform Categorical Features using DictVectorizer.
    Returns (X_train, X_val, y_train, y_val, dv).
    """
    categorical = ["PU_DO"]
    numerical = ["trip_distance", "passenger_count"]
    target_column = config.target_column
    dv = DictVectorizer(sparse=False)

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train, y_val = df_train[target_column].values, df_val[target_column].values

    return X_train, X_val, y_train, y_val, dv
