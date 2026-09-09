import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import Config
from prodml.data import load_clean_data, split_data
from prodml.features import engineer_features
from prodml.logging_conf import setup_logging
from prodml.predict import DurationPredictor

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: Any, n_features: int, onnx_path: str = "models/model.onnx"
) -> None:
    """Export a scikit-learn model to ONNX with dynamic batch axis."""
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
    )
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"ONNX model saved to {onnx_path}")


def benchmark_latency(
    predict_fn: Callable, X: np.ndarray, n_runs: int = 30
) -> dict[str, float]:
    """Measure mean, median(p50), p95 and p99 latency over multiple runs (in milliseconds)."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X)
        times.append((time.perf_counter() - start) * 1000)

    times_arr = np.array(times)
    return {
        "mean_ms": round(float(times_arr.mean()), 2),
        "p50_ms": round(float(np.percentile(times_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(times_arr, 99)), 2),
    }


def run_parity_and_benchmark(
    n_samples: int = 500, atol: float = 1e-4, n_runs: int = 30
) -> dict[str, Any]:
    """Compare pickle vs ONNX predictions and and measure latency (mean + p95)."""
    setup_logging()
    config = Config()
    logger.info("Starting parity check and benchmark")

    # 1. Load data
    df = load_clean_data(config)
    _, val_df = split_data(df, config)
    sample_df = val_df.head(n_samples)

    # 2. Load pickle model
    predictor = DurationPredictor().load_model()
    X, _ = engineer_features(sample_df, dv=predictor.dv, fit=False)
    X = X.astype(np.float32)

    # 3. Export to ONNX
    onnx_path = config.onnx_model_path
    export_to_onnx(predictor.model, n_features=X.shape[1], onnx_path=onnx_path)

    # 4. Parity check (single run)
    pred_pkl = predictor.model.predict(X)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    pred_onnx = session.run(None, {input_name: X})[0].ravel()

    max_diff = float(np.abs(pred_pkl - pred_onnx).max())
    is_close = bool(np.allclose(pred_pkl, pred_onnx, atol=atol))

    logger.info(
        "Parity check",
        extra={
            "extra_data": {
                "max_diff": round(max_diff, 6),
                "allclose": is_close,
                "atol": atol,
            }
        },
    )

    if not is_close:
        raise AssertionError(
            f"Predictions differ by more than {atol}. Max diff = {max_diff}"
        )

    # 5. Benchmark (mean + median + p95 + p99)
    def pkl_predict(data: np.ndarray) -> np.ndarray:
        return predictor.model.predict(data)

    def onnx_predict(data: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: data})[0].ravel()

    pkl_stats = benchmark_latency(pkl_predict, X, n_runs=n_runs)
    onnx_stats = benchmark_latency(onnx_predict, X, n_runs=n_runs)

    results = {
        "n_samples": n_samples,
        "max_diff": round(max_diff, 6),
        "pickle_mean_ms": pkl_stats["mean_ms"],
        "pickle_median_ms": pkl_stats["p50_ms"],
        "pickle_p95_ms": pkl_stats["p95_ms"],
        "pickle_p99_ms": pkl_stats["p99_ms"],
        "onnx_mean_ms": onnx_stats["mean_ms"],
        "onnx_median_ms": onnx_stats["p50_ms"],
        "onnx_p95_ms": onnx_stats["p95_ms"],
        "onnx_p99_ms": onnx_stats["p99_ms"],
    }

    logger.info("Benchmark results", extra={"extra_data": results})
    return results


if __name__ == "__main__":
    run_parity_and_benchmark()
