from fastapi import FastAPI, HTTPException
import onnxruntime as ort
from prodml.config import Config
import numpy as np
import time
from .schemas import PredictionRequest, PredictionResponse
from prodml.predict import DurationPredictor

config = Config()
app = FastAPI(
    title="ProdML API",
    description="ProdML API for managing production machine learning models.",
    version="1.0.0",
)

predictor = DurationPredictor().load_model()

# Load the model once at startup using FastAPI's lifespan context manager — never inside the request handler.
# Loading per-request is the most common beginner mistake and it costs 100× in latency.
session = ort.InferenceSession(config.onnx_model_path)
input_name = session.get_inputs()[0].name


@app.get("/health")
def health():
    """
    Returns a 200 status code only if the model object is loaded in memory
    — not just "the process is alive
    """
    if session is None:
        return {"status": "unhealthy"}
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """
    Returns metadata about the API.
    Model version, training date, feature names, framework, artifact hash
    """
    return {"title": app.title, "description": app.description, "version": app.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionRequest) -> PredictionResponse:
    """
    Accepts input data and returns predictions from the model.
    """
    try:
        # Convert input data to the appropriate format for ONNX model
        dict_data = input_data.model_dump()
        dict_data.pop("correlation_id", None)  # Remove correlation_id if present

        X = predictor.dv.transform([dict_data])
        input_array = np.array(X, dtype=np.float32)

        start = time.perf_counter()
        predictions = session.run(None, {input_name: input_array})[0]
        latency = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            prediction=float(predictions.flat[0]),
            model_version="1.0.0",
            correlation_id=getattr(input_data, "correlation_id", "N/A"),
            latency_ms=latency,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    batch_input_data: list[PredictionRequest],
) -> list[PredictionResponse]:
    """
    Accepts a batch of input data and returns predictions from the model.
    """
    try:
        dict_list = []
        correlation_ids = []

        for input_data in batch_input_data:
            dict_data = input_data.model_dump()
            dict_data.pop("correlation_id", None)  # Remove correlation_id if present
            dict_list.append(dict_data)
            correlation_ids.append(getattr(input_data, "correlation_id", "N/A"))

        X = predictor.dv.transform(dict_list)
        input_array = np.array(X, dtype=np.float32)

        start = time.perf_counter()
        predictions = session.run(None, {input_name: input_array})[0]
        total_latency = (time.perf_counter() - start) * 1000

        avg_latency_per_sample = total_latency / len(batch_input_data)

        response_list = []
        for pred, corr_id in zip(predictions.flat, correlation_ids):
            response_list.append(
                PredictionResponse(
                    prediction=float(pred),
                    model_version="1.0.0",
                    correlation_id=corr_id,
                    latency_ms=avg_latency_per_sample,
                )
            )
        return response_list
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch model inference failed: {str(e)}"
        )


"""
{
  "trip_distance": 4.08,
  "VendorID": 2,
  "store_and_fwd_flag": "N",
  "RatecodeID": 1.0,
  "PULocationID": 198,
  "DOLocationID": 56,
  "passenger_count": 3,
  "trip_type": 1.0
}
"""
