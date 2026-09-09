from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    PredictionRequest is a Pydantic model that represents the input data for making predictions.
    It includes fields for the input features required by the prediction model.
    """

    # model_config
    trip_distance: float = Field(
        ...,
        gt=0,
        lt=200,
        description="The distance of the trip in miles. Must be greater than 0 and less than 200.",
    )
    VendorID: int = Field(
        ..., description="The ID of the vendor providing the service."
    )
    store_and_fwd_flag: str = Field(
        ...,
        description="A flag indicating whether the trip record was stored and forwarded.",
    )
    RatecodeID: int = Field(..., description="The ID of the rate code for the trip.")
    PULocationID: int = Field(
        ..., description="The ID of the pickup location for the trip."
    )
    DOLocationID: int = Field(
        ..., description="The ID of the dropoff location for the trip."
    )
    passenger_count: int = Field(
        ...,
        gt=0,
        description="The number of passengers in the trip. Must be greater than 0.",
    )
    trip_type: int = Field(
        ...,
        description="The type of trip. This field indicates the nature of the trip.",
    )
    # fare_amount
    # extra
    # mta_tax
    # tip_amount
    # tolls_amount
    # improvement_surcharge
    # total_amount
    # payment_type
    # congestion_surcharge
    # cbd_congestion_fee
    # correlation_id


class PredictionResponse(BaseModel):
    """
    PredictionResponse is a Pydantic model that represents the output data from the prediction model.
    It includes fields for the predicted values and any additional information related to the prediction.
    """

    prediction: float = Field(..., description="The predicted value from the model.")
    model_version: str = Field(
        ..., description="The version of the model used for making the prediction."
    )
    correlation_id: str = Field(
        ...,
        description="A unique identifier for the prediction request, useful for tracking and debugging.",
    )
    latency_ms: float = Field(
        ..., description="The time taken to generate the prediction in milliseconds."
    )
