# Model Fit + Persist

import numpy as np
import pickle
import logging
from pathlib import Path

from prodml.logging_conf import setup_logging
from prodml.data import load_clean_data, inspect_data, split_data
from prodml.features import engineer_features
from prodml.utils import timed
from prodml.config import Config

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

config = Config()
logger = logging.getLogger(__name__)


@timed
def train_evaluate_model(
    config: Config = config, model_name: str = "Random Forest"
) -> None:
    """
    Train a machine learning model.
    """
    setup_logging()
    logger.info(f"Starting {model_name} training pipeline")

    # 1. Load Clean Data and Inspect
    df = load_clean_data(config)
    inspect_data(df)
    logger.info("Loaded and cleaned data", extra={"extra_data": {"rows": len(df)}})

    # 2. Split Data into Train and Test Sets
    train_df, val_df = split_data(df, config)

    # 3. Define Features & Target
    X_train, dv = engineer_features(train_df, fit=True)
    y_train = train_df[config.target_column].values

    X_val, _ = engineer_features(val_df, dv=dv, fit=False)
    y_val = val_df[config.target_column].values

    # 4. Train Model
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        # max_depth=config.max_depth,
        # min_samples_split=config.min_samples_split,
        # min_samples_leaf=config.min_samples_leaf,
        # random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Model
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
            "extra_data": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 2),
            }
        },
    )

    # 6. Persist model + DictVectorizer together
    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config.model_path, "wb") as f_out:
        pickle.dump({"dict_vectorizer": dv, "model": model}, f_out)
    logger.info(f"Model saved to {config.model_path}")


if __name__ == "__main__":
    train_evaluate_model()
