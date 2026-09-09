# Model Fit + Persist

import numpy as np
import os
import pickle

from prodml.logging_conf import setup_logging
from prodml.data import load_data, inspect_data, split_data
from prodml.features import engineer_features
from prodml.utils import timed
from prodml.config import Config
from prodml.export import export_onnx

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

config = Config()
logger = setup_logging("prodml.train")


@timed
def train_evaluate_model(model_name: str = "Linear Regression") -> None:
    """
    Train a machine learning model.
    """
    setup_logging()
    logger.info(f"Starting {model_name} training pipeline")

    # 1. Load Clean Data and Inspect
    df = load_data(config)
    logger.info(
        "[1]Loaded and cleaned data",
        extra={"extra": {"rows": len(df), "columns": len(df.columns)}},
    )
    inspect_data(df)

    # 2. Split Data into Train and Test Sets
    df_train, df_val = split_data(df, config)
    logger.info(
        "[2]Split data into train and validation sets",
        extra={"extra": {"train_rows": len(df_train), "val_rows": len(df_val)}},
    )

    # 3. Define Training and Validation Features and Target
    X_train, X_val, y_train, y_val, dv = engineer_features(df_train, df_val)
    logger.info(
        "[3]Engineered features",
        extra={
            "extra": {
                "train_features": len(X_train.columns),
                "val_features": len(X_val.columns),
            }
        },
    )

    # 4. Train Model
    model = LinearRegression()
    logger.info(f"[4]Training {model_name} model")
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    logger.info(f"[5]Evaluating {model_name} model")
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    logger.info(f"Train Score: {train_score:.4f}")
    logger.info(f"Validation Score: {val_score:.4f}")
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    logger.info(
        "Validation metrics",
        extra={
            "extra": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 2),
            }
        },
    )

    # 6. Persist model + DictVectorizer together
    os.makedirs(os.path.dirname(config.pkl_model_path), exist_ok=True)
    with open(config.pkl_model_path, "wb") as f_out:
        pickle.dump({"dict_vectorizer": dv, "model": model}, f_out)
    logger.info(f"[6]Model saved to {config.pkl_model_path}")

    num_features = len(dv.get_feature_names_out())
    export_onnx(model, num_features)

    logger.info("Training Complete.")


if __name__ == "__main__":
    train_evaluate_model()
