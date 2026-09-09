# Testing DurationPredictor class

from prodml.predict import DurationPredictor

# 1. Load the model
predictor = DurationPredictor().load_model()

# Single prediction
sample = {
    "PULocationID": "66",
    "DOLocationID": "262",
    "store_and_fwd_flag": "N",
}

pred = predictor.predict_one(sample)
print(f"Single prediction: {pred:.2f} minutes")

# Batch prediction
batch = [
    {"PULocationID": "66", "DOLocationID": "262", "store_and_fwd_flag": "N"},
    {"PULocationID": "198", "DOLocationID": "56", "store_and_fwd_flag": "N"},
    {"PULocationID": "95", "DOLocationID": "160", "store_and_fwd_flag": "N"},
]

preds = predictor.predict_batch(batch)
print("Batch predictions:")
for i, p in enumerate(preds, 1):
    print(f"  Sample {i}: {p:.2f} minutes")
