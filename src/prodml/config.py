# All Paths, Hyperparameters & Ports
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Paths (Relative to Project Root)
    data_path: str = "data/raw/green_tripdata_2026-05.parquet"
    pkl_model_path: str = "models/baseline.pkl"
    onnx_model_path: str = "models/model.onnx"

    # Target Column
    target_column: str = "trip_duration"

    # Ports
    api_port: int = 8000
    api_host: str = "0.0.0.0"


config = Config()
