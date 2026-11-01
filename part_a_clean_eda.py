
import pandas as pd
import numpy as np

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES
)

# --------------------------------------------------
# 1. Load historical shipment data and hubs
# --------------------------------------------------

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

raw_rows = len(raw)
duplicate_rows = raw.duplicated().sum()

# --------------------------------------------------
# 2. Clean shipment data
# --------------------------------------------------

clean, medians = clean_shipments(raw)

# --------------------------------------------------
# 3. Build features
# --------------------------------------------------

df = build_features(clean, hubs)

# Create target variable
df["is_delayed"] = (
    df["delivery_hours"] > df["promised_hours"]
).astype(int)

# --------------------------------------------------
# 4. Save modelling table
# --------------------------------------------------

df.to_csv("data/modelling_table.csv", index=False)

# --------------------------------------------------
# 5. Data quality checks
# --------------------------------------------------

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows": duplicate_rows,
    "clean_rows": len(clean),
    "rows_after_hub_merge": len(df),
    "missing_distance_before": raw["distance_km"].isna().sum(),
    "missing_weight_before": raw["parcel_weight_kg"].isna().sum(),
    "missing_declared_value_before": raw["declared_value"].isna().sum(),
    "missing_payment_mode_before": raw["payment_mode"].isna().sum(),
    "delayed_shipments": df["is_delayed"].sum(),
    "delay_rate": df["is_delayed"].mean(),
    "missing_features_and_target": df[FEATURES + ["is_delayed"]].isna().sum().sum()
}

checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
)

checks_df["value"] = checks_df["value"].round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# --------------------------------------------------
# 6. Delay analysis by service type
# --------------------------------------------------

def delay_summary(data, dimension, groups=None):
    if groups is None:
        groups = data.groupby(dimension)

    result = groups.agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    result = result.rename(columns={dimension: "group"})
    result.insert(0, "dimension", dimension)

    return result


service_summary = delay_summary(df, "service_type")
hub_summary = delay_summary(df, "hub_tier")

# Distance bands
df["distance_band"] = pd.cut(
    df["distance_km"],
    bins=[0, 200, 400, 600, np.inf],
    labels=["0-200", "201-400", "401-600", "601+"]
)

distance_summary = delay_summary(df, "distance_band")

delay_groups = pd.concat(
    [service_summary, hub_summary, distance_summary],
    ignore_index=True
)

delay_groups["delay_rate"] = delay_groups["delay_rate"].round(4)
delay_groups["mean_hours"] = delay_groups["mean_hours"].round(4)

delay_groups.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

# --------------------------------------------------
# 7. Correlation analysis
# --------------------------------------------------

correlations = df.select_dtypes(include="number").corr()

correlations.round(4).to_csv(
    "outputs/part_a_correlations.csv",
    index=True
)

# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("PART A COMPLETED SUCCESSFULLY")
print("--------------------------------")
print("Raw rows:", raw_rows)
print("Duplicate rows:", duplicate_rows)
print("Clean rows:", len(clean))
print("Rows after hub merge:", len(df))
print("Delayed shipments:", df["is_delayed"].sum())
print("Delay rate:", round(df["is_delayed"].mean(), 4))
print("\nHistorical medians:")
print(medians)

print("\nAll feature and target missing values:")
print(df[FEATURES + ["is_delayed"]].isna().sum())

print("\nDelay by service type:")
print(service_summary)

print("\nDelay by hub tier:")
print(hub_summary)

print("\nCreated files:")
print("data/modelling_table.csv")
print("outputs/part_a_checks.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")

