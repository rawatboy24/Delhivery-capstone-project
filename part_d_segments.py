
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

# 1. Load modelling data
df = pd.read_csv("data/modelling_table.csv")

# 2. Create hub profile
hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_profile.to_csv("outputs/hub_profile.csv", index=False)

# 3. Scale the four profile columns
PROFILE_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = hub_profile[PROFILE_FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Separation evidence for K = 2, 3, 4, 5
separation_rows = []

for k in [2, 3, 4, 5]:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_scaled)

    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(X_scaled)), 2):
        distance = np.linalg.norm(X_scaled[i] - X_scaled[j])

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    ratio = np.mean(across_distances) / np.mean(within_distances)

    sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True
    )

    separation_rows.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(map(str, sizes))
    })

separation = pd.DataFrame(separation_rows)
separation.to_csv("outputs/part_d_separation.csv", index=False)

# 5. Final K-Means model with K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(X_scaled)

# 6. Name clusters according to their mean delay rate
cluster_delay = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

segment_names = {
    cluster_delay.index[0]: "Standard",
    cluster_delay.index[1]: "Watch",
    cluster_delay.index[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(segment_names)

# 7. Create hub segments output
hub_segments = hub_profile[["hub_id", "segment"]].merge(
    df[["hub_id", "hub_city", "hub_tier"]].drop_duplicates(),
    on="hub_id",
    how="left"
)

hub_segments = hub_segments.sort_values("hub_id").reset_index(drop=True)

hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# 8. Create segment profile
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values("segment")
segment_profile["segment"] = segment_profile["segment"].astype(str)

segment_profile.to_csv("outputs/segment_profile.csv", index=False)

# 9. Find tier-versus-segment mismatches
expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(expected_segment)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
][["hub_id", "segment", "hub_city", "hub_tier"]]

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)

# 10. Print results
print("\nHUB PROFILE")
print(hub_profile.drop(columns=["cluster", "segment"]).round(4).to_string(index=False))

print("\nSEPARATION RESULTS")
print(separation.round(3).to_string(index=False))

print("\nSEGMENT PROFILE")
print(segment_profile.round(4).to_string(index=False))

print("\nHUB SEGMENTS")
print(hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]].to_string(index=False))

print("\nTIER VS SEGMENT MISMATCHES")
print(mismatches.to_string(index=False))

print("\nPART D SEGMENTATION COMPLETED!")
