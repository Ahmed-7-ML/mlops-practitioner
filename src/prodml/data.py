# Loading Parquet, Train/Val Split

from typing import Tuple
import pandas as pd
from prodml.logging_conf import setup_logging

logger = setup_logging("prodml.data")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the data from the specified path in the configuration.

    Args:
        config (Config): Configuration object containing data path.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # 1. Load the data
    df = pd.read_parquet(file_path)

    # 2. Calculate trip duration in minutes
    df["trip_duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # 3. Filter out trips with duration less than 1 minute or greater than 60 minutes
    # Filter out trips with distance less than or equal to 0
    df = df[(df["trip_duration"] >= 1) & (df["trip_duration"] <= 60)]
    df = df[df["trip_distance"] > 0]

    # 4. Create categorical PU_DO feature
    categorincal_features = ["PULocationID", "DOLocationID"]
    df[categorincal_features] = df[categorincal_features].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    # Drop Unnecessary Columns
    df = df.drop(columns=["lpep_pickup_datetime", "lpep_dropoff_datetime", "ehail_fee"])
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspect the DataFrame by printing its shape, columns, info, description, and head.

    Args:
        df (pd.DataFrame): DataFrame to inspect.
    """
    logger.info(f"Data Shape: {df.shape}")
    logger.info(f"Data Columns: {df.columns.tolist()}")
    logger.info("Data Info:\n")
    df.info()
    logger.info(f"Data Description:\n{df.describe()}")
    logger.info(f"Data Head:\n{df.head()}")


def split_data(
    df: pd.DataFrame, train_size: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        config (Config): Configuration object containing test size and random state.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    cutoff = int(len(df) * train_size)
    train_df = df.iloc[:cutoff]
    val_df = df.iloc[cutoff:]
    return train_df, val_df
