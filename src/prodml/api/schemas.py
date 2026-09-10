from pydantic import BaseModel, Field, ConfigDict


# PredictionRequest and PredictionResponse schemas for the API --> Single Prediction
class PredictionRequest(BaseModel):
    DOLocationID: int = Field(..., gt=0, description="The ID of the drop-off location.")
    PULocationID: int = Field(..., gt=0, description="The ID of the pick-up location.")
    passenger_count: int = Field(
        default=1, gt=0, lt=10, description="The number of passengers in the trip."
    )
    trip_distance: float = Field(
        ..., gt=0, lt=200, description="The distance of the trip in miles."
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "DOLocationID": 1,
                "PULocationID": 2,
                "passenger_count": 3,
                "trip_distance": 10.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(
        ..., description="The predicted duration of the trip in minutes."
    )
    latency_ms: float = Field(
        ..., description="The latency of the prediction in milliseconds."
    )
    model_version: str = Field(
        ..., description="The version of the model used for prediction."
    )
    correlation_id: str = Field(
        ...,
        description="A unique identifier for the request, useful for tracing and debugging.",
    )


# BatchPredictionRequest and BatchPredictionResponse schemas for the API --> Batch Prediction
class BatchPredictionRequest(BaseModel):
    inputs: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        description="A list of prediction requests for batch processing.",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {
                        "DOLocationID": 1,
                        "PULocationID": 2,
                        "passenger_count": 3,
                        "trip_distance": 10.5,
                    },
                    {
                        "DOLocationID": 3,
                        "PULocationID": 4,
                        "passenger_count": 1,
                        "trip_distance": 5.0,
                    },
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(
        ...,
        description="A list of prediction responses corresponding to the input requests.",
    )
    latency_ms: float = Field(
        ..., description="Total Batch Inference Latency in milliseconds."
    )
    model_version: str = Field(
        ..., description="The version of the model used for prediction."
    )
    correlation_id: str = Field(
        ...,
        description="A unique identifier for the request, useful for tracing and debugging.",
    )
