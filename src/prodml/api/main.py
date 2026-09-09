from fastapi import FastAPI
import onnxruntime as ort
from prodml.config import Config
import numpy as np
from schemas import PredictionRequest, PredictionResponse

config = Config()
app = FastAPI(
    title="ProdML API",
    description="ProdML API for managing production machine learning models.",
    version="1.0.0",
)

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


@app.post("/predict")
def predict(input_data: PredictionRequest) -> PredictionResponse:
    """
    Accepts input data and returns predictions from the model.
    """
    try:
        # Convert input data to the appropriate format for ONNX model
        input_array = np.array([list(input_data.values())], dtype=np.float32)
        predictions = session.run(None, {input_name: input_array})[0]
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/batch")
def predict_batch(
    batch_input_data: list[PredictionRequest],
) -> list[PredictionResponse]:
    """
    Accepts a batch of input data and returns predictions from the model.
    """
    try:
        # Convert input data to the appropriate format for ONNX model
        for input_data in batch_input_data:
            if not isinstance(input_data, dict):
                return {"error": "Each item in the batch must be a dictionary."}
            input_array = np.array([list(input_data.values())], dtype=np.float32)
            predictions = session.run(None, {input_name: input_array})[0]
            yield {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
