
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments


# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# 1. Load final configuration
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

if str(config["max_depth"]).lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(config["max_depth"])


# 2. Load modelling data
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]


# 3. Recreate exact Part C split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# 4. Build final pipeline
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])


# 5. Train using training rows only
final_model.fit(X_train, y_train)


# 6. Save model
joblib.dump(final_model, "models/delay_model.pkl")


# 7. Save threshold
pd.DataFrame({
    "threshold": [threshold]
}).to_csv("models/delay_threshold.csv", index=False)


# 8. Save feature order
pd.DataFrame({
    "feature": FEATURES
}).to_csv("models/feature_order.csv", index=False)


# 9. Calculate and save cleaning medians
raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)

impute_df = pd.DataFrame([
    {"column": column, "median": medians[column]}
    for column in [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]
])

impute_df.to_csv("models/impute_values.csv", index=False)


# 10. Reload saved model and compare probabilities
in_memory_probabilities = final_model.predict_proba(X_test)[:, 1]

loaded_model = joblib.load("models/delay_model.pkl")

reloaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

max_difference = abs(
    in_memory_probabilities - reloaded_probabilities
).max()


# 11. Save reload verification
reload_check = pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {"metric": "reload_max_abs_probability_diff", "value": max_difference},
    {"metric": "threshold", "value": threshold}
])

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)


print("PART D MODEL PACKAGING COMPLETED")
print("--------------------------------")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Selected features:", selector_k)
print("Threshold:", threshold)
print("Reload probability difference:", max_difference)
print("\nSaved model files:")
print("- models/delay_model.pkl")
print("- models/delay_threshold.csv")
print("- models/feature_order.csv")
print("- models/impute_values.csv")
print("- outputs/part_d_reload_check.csv")
