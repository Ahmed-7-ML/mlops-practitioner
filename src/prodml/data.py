# Loading Parquet, Train/Val Split

from typing import Tuple
import pandas as pd
import logging
from prodml.config import Config
from sklearn.model_selection import train_test_split

config = Config()
logger = logging.getLogger(__name__)


def load_clean_data(config: Config = config) -> pd.DataFrame:
    """
    Load and clean the data from the specified path in the configuration.

    Args:
        config (Config): Configuration object containing data path.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info(f"Loading Parquet File from: {config.data_path}")
    df = pd.read_parquet(config.data_path)

    # Calculate trip duration in minutes
    df["trip_duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Filter Duration and Distance
    df = df[(df["trip_duration"] >= 1) & (df["trip_duration"] <= 60)]
    df = df[df["trip_distance"] > 0]

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
    df: pd.DataFrame, config: Config = config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        config (Config): Configuration object containing test size and random state.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    train_df, val_df = train_test_split(
        df, test_size=config.test_size, random_state=config.random_state
    )
    return train_df, val_df
