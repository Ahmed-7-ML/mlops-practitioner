# Load Model + Single/Batch Prediction

import pandas as pd
import pickle
from typing import Any
from prodml.config import config
from prodml.utils import timed
from prodml.logging_conf import setup_logging

logger = setup_logging("prodml.predict")

CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance", "passenger_count"]


class DurationPredictor:
    """
    Load the trained model and DictVectorizer, and provide methods for single and batch predictions.
    """

    def __init__(self, model_path: str = None):
        self.pkl_model_path = model_path or config.pkl_model_path
        self.model = None
        self.dv = None

    def load_pkl_model(self) -> "DurationPredictor":
        """
        Load the trained model and DictVectorizer from the specified path.
        """
        try:
            with open(self.pkl_model_path, "rb") as f_in:
                artifacts = pickle.load(f_in)
            self.model = artifacts["model"]
            self.dv = artifacts["dict_vectorizer"]
            logger.info(f"Model loaded from {self.pkl_model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.pkl_model_path}: {e}")
            raise
        return self

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using the fitted DictVectorizer."""
        if self.dv is None:
            raise RuntimeError("Model not loaded. Call .load_pkl_model() first.")

        df = df.copy()

        # Build PU_DO if not present
        if "PU_DO" not in df.columns:
            if "PULocationID" in df.columns and "DOLocationID" in df.columns:
                df["PULocationID"] = df["PULocationID"].astype(str)
                df["DOLocationID"] = df["DOLocationID"].astype(str)
                df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
            else:
                raise ValueError("Need either PU_DO or (PULocationID + DOLocationID)")

        # Ensure required columns exist
        for col in NUMERICAL:
            if col not in df.columns:
                df[col] = 0.0

        dicts = df[CATEGORICAL + NUMERICAL].to_dict(orient="records")
        X = self.dv.transform(dicts)
        return X

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """
        Predict the trip duration for a single set of features.

        Args:
            features (dict): A dictionary containing feature values.
        Returns:
            float: Predicted trip duration.
        """
        if self.model is None or self.dv is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded. Call .load_pkl_model() first.")

        trip_distance = features.get("trip_distance")
        if trip_distance is not None and float(trip_distance) > 100:
            logger.warning(
                "Input outside training range",
                extra={"extra": {"trip_distance": trip_distance}},
            )

        logger.debug("Feature vector", extra={"extra": {"features": features}})

        df = pd.DataFrame([features])
        X = self._prepare_X(df)
        prediction = float(self.model.predict(X)[0])
        logger.info(
            "Prediction served",
            extra={"extra": {"prediction": round(prediction, 2)}},
        )
        return prediction

    @timed
    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        """
        Predict the trip durations for a batch of feature sets.

        Args:
            features_list (list): A list of dictionaries, each containing feature values.
        Returns:
            list: Predicted trip durations.
        """
        if self.model is None or self.dv is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded. Call .load() first.")

        df = pd.DataFrame(features_list)
        X = self._prepare_X(df)
        predictions = [float(pred) for pred in self.model.predict(X)]
        logger.info(
            "Batch prediction served",
            extra={"extra": {"batch_size": len(predictions)}},
        )
        return predictions
