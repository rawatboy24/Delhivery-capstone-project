import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES

# Step 1: Load modelling table
df = pd.read_csv("data/modelling_table.csv")

# Step 2: Select input features and target
X = df[FEATURES]
y = df["delivery_hours"]

# Step 3: First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Step 4: Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

# Step 5: Fit scaler ONLY on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same fitted scaler
X_test_scaled = scaler.transform(X_test)

# Step 6: Calculate scaling checks
checks = {
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": abs(
        X_train_scaled.mean(axis=0)
    ).max(),
    "test_scaled_max_abs_mean": abs(
        X_test_scaled.mean(axis=0)
    ).max()
}

# Step 7: Save checks
checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
)

checks_df["value"] = checks_df["value"].round(10)

checks_df.to_csv(
    "outputs/part_b_checks.csv",
    index=False
)

# Step 8: Print results
print("PART B - TRAIN / VALIDATION / TEST SPLIT")
print("----------------------------------------")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\nScaling checks:")
print(checks_df)

print("\nSUCCESS: Data split and scaling completed!")

# STEP 3: BASELINE MODEL

# Training data ka average delivery time
baseline_prediction = y_train.mean()

# Har validation row ke liye same prediction
y_val_pred_baseline = np.full(len(y_val), baseline_prediction)

# Evaluation metrics
baseline_mae = mean_absolute_error(y_val, y_val_pred_baseline)
baseline_mse = mean_squared_error(y_val, y_val_pred_baseline)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_val, y_val_pred_baseline)

print("\nBASELINE MODEL RESULTS")
print("----------------------")
print("Baseline Prediction:", round(baseline_prediction, 2))
print("MAE:", round(baseline_mae, 2))
print("MSE:", round(baseline_mse, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2 Score:", round(baseline_r2, 4))

# STEP 4: DECISION TREE DEPTH EXPERIMENT

from sklearn.tree import DecisionTreeRegressor

depth_values = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in depth_values:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    tree_results.append({
        "max_depth": depth if depth is not None else "unlimited",
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

tree_depth_df = pd.DataFrame(tree_results)

tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False
)

print("\nDECISION TREE DEPTH RESULTS")
print("---------------------------")
print(tree_depth_df.round(3).to_string(index=False))

best_tree_row = tree_depth_df.loc[tree_depth_df["val_rmse"].idxmin()]

print("\nBest Tree Depth:", best_tree_row["max_depth"])
print("Best Validation RMSE:", round(best_tree_row["val_rmse"], 3))

# STEP 5: COMPARE FOUR MODELS

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

# Linear Regression
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Decision Tree
decision_model = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

# Random Forest
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Gradient Boosting
gradient_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

models = {
    "linear_regression": linear_model,
    "decision_tree": decision_model,
    "random_forest": random_forest_model,
    "gradient_boosting": gradient_model
}

model_results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    model_results.append({
        "model": name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

metrics_df = pd.DataFrame(model_results)

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("\nFOUR MODEL COMPARISON")
print("---------------------")
print(metrics_df.round(3).to_string(index=False))

# STEP 6: GRADIENT BOOSTING LEARNING RATE EXPERIMENT

learning_rates = [0.01, 0.1, 1.0]

gb_results = []

for lr in learning_rates:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    predictions = gb_model.predict(X_val)

    val_rmse = np.sqrt(mean_squared_error(y_val, predictions))
    val_r2 = r2_score(y_val, predictions)

    gb_results.append({
        "learning_rate": lr,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

gb_df = pd.DataFrame(gb_results)

gb_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)

print("\nGRADIENT BOOSTING LEARNING RATE RESULTS")
print("---------------------------------------")
print(gb_df.round(3).to_string(index=False))

# STEP 7: CROSS VALIDATION

from sklearn.model_selection import KFold, cross_val_score

# Training + Validation data combine
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Same KFold object for both models
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Top 2 models
top_models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

cv_results = []

for name, model in top_models.items():

    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)

print("\nCROSS VALIDATION RESULTS")
print("------------------------")
print(cv_df.round(3).to_string(index=False))

# STEP 8: FINAL TEST EVALUATION

# Linear Regression was the validation winner
final_model = models["linear_regression"]

# Predict test data only once
y_test_pred = final_model.predict(X_test)

# Calculate final test metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Create final test result
test_result = pd.DataFrame([{
    "model": "linear_regression",
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}])

# Add test row to existing metrics
final_metrics = pd.concat(
    [metrics_df, test_result],
    ignore_index=True
)

final_metrics.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("\nFINAL TEST RESULTS")
print("------------------")
print(final_metrics.tail(1).round(3).to_string(index=False))

import joblib

joblib.dump(final_model, "models/final_model.pkl")
