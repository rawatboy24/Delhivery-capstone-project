

import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops", "rain_flag",
    "sale_event_flag", "is_cod", "is_weekend",
    "dispatch_hour", "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no",
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]


def clean_shipments(raw, medians=None):
    """
    Clean raw shipment data.

    Parameters:
        raw: Raw shipment DataFrame
        medians: Historical median values (optional)

    Returns:
        clean: Cleaned DataFrame
        medians: Dictionary of median values
    """

    # Step 1: Make a copy of raw data
    clean = raw.copy()

    # Step 2: Remove exact duplicate rows
    clean = clean.drop_duplicates().copy()

    # Step 3: Clean service_type
    clean["service_type"] = (
        clean["service_type"]
        .str.strip()
        .str.title()
    )

    # Step 4: Fill missing payment_mode
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # Step 5: Convert declared_value into numeric values
    clean["declared_value_inr"] = (
        clean["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value_inr"],
        errors="coerce"
    )

    clean["declared_value_inr"] = clean["declared_value_inr"].astype(float)

    # Remove original declared_value column
    clean = clean.drop(columns=["declared_value"])

    # Step 6: Convert dispatch_time into datetime
    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"],
        errors="coerce"
    )

    # Step 7: Convert parcel weight above 100 into missing values
    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # Step 8: Calculate historical medians
    columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:
        medians = {
            col: clean[col].median()
            for col in columns
        }

    # Step 9: Fill missing numeric values using medians
    for col in columns:
        clean[col] = clean[col].fillna(medians[col])

    return clean, medians


def build_features(clean, hubs):
    """
    Merge hub information and create ML features.
    """

    # Step 1: Make a copy
    df = clean.copy()

    # Step 2: Merge hub information
    df = df.merge(
        hubs,
        on="hub_id",
        how="left"
    )

    # Step 3: Create promised_hours
    df["promised_hours"] =    df["service_type"].map(PROMISED_HOURS)

    # Step 4: Create is_cod
    df["is_cod"] = (
        df["payment_mode"] == "COD"
    ).astype(int)

    # Step 5: Create is_weekend
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Step 6: Extract dispatch hour
    df["dispatch_hour"] = (
        df["dispatch_time"].dt.hour
    )

    # Step 7: Log transformation of declared value
    df["log_declared_value"] = np.log1p(
        df["declared_value_inr"]
    )

    return df
