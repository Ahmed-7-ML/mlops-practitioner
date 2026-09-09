from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

import time
import uuid
from contextlib import asynccontextmanager

from prodml.api.schemas import BatchTripInput, BatchTripResponse, TripInput, TripOutput
from prodml.predict import DurationPredictor
from prodml.logging_conf import correlation_id_var, setup_logging


logger = setup_logging("prodml.api")
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model at startup rather than per request, which is a common beginner mistake that costs 100× in latency."""
    global predictor
    logger.info("Starting up: Loading model artifact into memory...")

    try:
        predictor = DurationPredictor().load_model()
    except Exception as e:
        logger.error("Startup model load failure", extra={"extra": {"error": str(e)}})

    yield
    # Cleanup code if needed
    logger.info("Shutting down: Releasing model resources...")
    predictor = None


app = FastAPI(
    title="ProdML API",
    description="ProdML API for managing production machine learning models.",
    version="1.0.0",
)

# predictor = DurationPredictor().load_model()

# # Load the model once at startup using FastAPI's lifespan context manager — never inside the request handler.
# # Loading per-request is the most common beginner mistake and it costs 100× in latency.
# session = ort.InferenceSession(config.onnx_model_path)
# input_name = session.get_inputs()[0].name


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Maintains your ContextVar-based correlation ID tracking."""
    req_id = str(uuid.uuid4())
    token = correlation_id_var.set(req_id)

    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    correlation_id_var.reset(token)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Validation rejection", extra={"extra": {"errors": exc.errors()}})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
def health():
    """
    Returns a 200 status code only if the model object is loaded in memory
    — not just "the process is alive
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded in memory.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/metadata")
def metadata():
    """
    Returns metadata about the API.
    Model version, training date, feature names, framework, artifact hash
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded in memory.",
        )
    return predictor.get_metadata()


@app.post("/predict", response_model=TripOutput)
def predict(trip: TripInput) -> TripOutput:
    """
    Accepts input data and returns predictions from the model.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model unavailable")

    if trip.trip_distance > 100:
        logger.warning(
            "trip_distance > 100", extra={"extra": {"distance": trip.trip_distance}}
        )

    features = {
        "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
        "trip_distance": trip.trip_distance,
        "passenger_count": trip.passenger_count,
    }
    logger.debug("feature vector", extra={"extra": {"features": features}})

    start = time.perf_counter()
    prediction = predictor.predict_one(features)
    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "prediction served with latency",
        extra={"extra": {"latency_sec": latency, "prediction": prediction}},
    )
    return TripOutput(
        prediction=prediction,
        latency_ms=latency,
        model_version="1.0.0",
        correlation_id=correlation_id_var.get(),
    )


@app.post("/predict/batch", response_model=BatchTripResponse)
def predict_batch(
    batch_input_data: BatchTripInput,
) -> BatchTripResponse:
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model unavailable")

    predictions = []
    start = time.perf_counter()
    for trip in batch_input_data.inputs:
        features = {
            "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
            "trip_distance": trip.trip_distance,
            "passenger_count": trip.passenger_count,
        }
        logger.debug("feature vector", extra={"extra": {"features": features}})
        prediction = predictor.predict_one(features)
        predictions.append(
            TripOutput(
                prediction=prediction,
                latency_ms=0,  # Individual latency not tracked in batch
                model_version="1.0.0",
                correlation_id=correlation_id_var.get(),
            )
        )
    total_latency = (time.perf_counter() - start) * 1000
    logger.info(
        "batch prediction served with total latency",
        extra={
            "extra": {"total_latency_sec": total_latency},
            "batch_size": len(batch_input_data.inputs),
        },
    )
    return BatchTripResponse(
        predictions=predictions,
        latency_ms=total_latency,
        model_version="1.0.0",
        correlation_id=correlation_id_var.get(),
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
