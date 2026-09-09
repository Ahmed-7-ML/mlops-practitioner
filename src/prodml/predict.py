# Load Model + Single/Batch Prediction

import pandas as pd
import pickle
import logging
from typing import Any
from prodml.features import engineer_features
from prodml.config import Config
from prodml.utils import timed

config = Config()
logger = logging.getLogger(__name__)


class DurationPredictor:
    """
    Load the trained model and DictVectorizer, and provide methods for single and batch predictions.
    """

    def __init__(self, model_path: str = None, config: Config = config):
        self.config = config
        self.model_path = model_path or self.config.model_path
        self.model = None
        self.dv = None

    def load_model(self) -> "DurationPredictor":
        """
        Load the trained model and DictVectorizer from the specified path.
        """
        try:
            with open(self.model_path, "rb") as f_in:
                artifacts = pickle.load(f_in)
            self.model = artifacts["model"]
            self.dv = artifacts["dict_vectorizer"]
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
        return self

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
            raise RuntimeError("Model not loaded. Call .load() first.")
        df = pd.DataFrame([features])
        X, _ = engineer_features(df, dv=self.dv, fit=False)
        prediction = self.model.predict(X)[0]
        logger.info(
            "Prediction served",
            extra={"extra_data": {"prediction": round(prediction, 2)}},
        )
        return float(prediction)

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
        X, _ = engineer_features(df, dv=self.dv, fit=False)
        predictions = self.model.predict(X)
        logger.info(
            "Batch prediction served",
            extra={"extra_data": {"batch_size": len(predictions)}},
        )
        return [float(pred) for pred in predictions]
