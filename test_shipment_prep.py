
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

# Step 1: Load CSV files
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

print("Original raw rows:", len(raw))

# Step 2: Clean shipment data
clean, medians = clean_shipments(raw)

print("\nCleaned rows:", len(clean))
print("Historical medians:", medians)

# Step 3: Build features
features = build_features(clean, hubs)

print("\nRows after hub merge:", len(features))

# Step 4: Check service types
print("\nService types:")
print(clean["service_type"].value_counts())

# Step 5: Check missing values
print("\nMissing values in important columns:")
check_columns = [
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr",
    "promised_hours",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "log_declared_value",
    "delivery_hours"
]

print(features[check_columns].isna().sum())

# Step 6: Final checks
assert len(clean) == 1200
assert len(features) == 1200
assert features[check_columns].isna().sum().sum() == 0

print("\nSUCCESS: All basic tests passed!")
