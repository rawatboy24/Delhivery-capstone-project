
import numpy as np
import pandas as pd
import joblib

from src.shipment_prep import clean_shipments, build_features


# ============================================================
# STEP 1: LOAD SAVED FILES
# ============================================================

new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

model = joblib.load("models/delay_model.pkl")

threshold = pd.read_csv(
    "models/delay_threshold.csv"
)["threshold"].iloc[0]

feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()

impute_df = pd.read_csv("models/impute_values.csv")

medians = dict(zip(
    impute_df["column"],
    impute_df["median"]
))

hub_segments = pd.read_csv("outputs/hub_segments.csv")


# ============================================================
# STEP 2: CLEAN NEW SHIPMENTS
# ============================================================

cleaned_shipments, _ = clean_shipments(
    new_shipments,
    medians=medians
)


# ============================================================
# STEP 3: BUILD FEATURES
# ============================================================

featured_shipments = build_features(
    cleaned_shipments,
    hubs
)

# Keep exactly the same feature order used during training
X_new = featured_shipments[feature_order]


# ============================================================
# STEP 4: PREDICT DELAY PROBABILITY
# ============================================================

probabilities = model.predict_proba(X_new)[:, 1]

# Use full precision for threshold decision
flagged = (probabilities >= threshold).astype(int)


# ============================================================
# STEP 5: ATTACH HUB SEGMENTS
# ============================================================

segment_map = hub_segments.set_index("hub_id")["segment"]

segments = featured_shipments["hub_id"].map(segment_map)


# ============================================================
# STEP 6: DECIDE OPERATIONAL ACTION
# ============================================================

actions = np.where(
    (flagged == 1) & (segments == "Escalate"),
    "Escalate to hub manager",
    np.where(
        flagged == 1,
        "Proactive customer call",
        "No action"
    )
)


# ============================================================
# STEP 7: CREATE FINAL OUTPUT
# ============================================================

result = pd.DataFrame({
    "shipment_id": featured_shipments["shipment_id"],
    "hub_id": featured_shipments["hub_id"],
    "segment": segments,
    "delay_probability": np.round(probabilities, 4),
    "flagged": flagged,
    "action": actions
})


# ============================================================
# STEP 8: SAVE AND VERIFY
# ============================================================

result.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)

print("NEW SHIPMENT PREDICTIONS COMPLETED")
print("----------------------------------")
print("Total shipments:", len(result))
print("Unique shipment IDs:", result["shipment_id"].nunique())
print("Missing values:", int(result.isna().sum().sum()))
print("Flagged shipments:", int(result["flagged"].sum()))

print("\nAction counts:")
print(result["action"].value_counts().to_string())

print("\nFirst 10 predictions:")
print(result.head(10).to_string(index=False))

print("\nSaved: outputs/new_shipment_actions.csv")
