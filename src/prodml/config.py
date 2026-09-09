# All Paths, Hyperparameters & Ports

# --> Imports
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Paths (Relative to Project Root)
    data_path: str = "data/raw/green_tripdata_2026-05.parquet"
    model_path: str = "models/model.pkl"
    output_path: str = "data/output/predictions.csv"
    onnx_model_path: str = "models/model.onnx"

    # Target Column
    target_column: str = "trip_duration"

    # Hyperparameters
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    # max_depth: int = 10
    # min_samples_split: int = 2
    # min_samples_leaf: int = 1

    # Ports
    api_port: int = 8000

    class Config:
        env_prefix = "PRODML_"
