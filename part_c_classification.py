
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# ============================================================
# STEP 1: STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]

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

print("\nPART C - STRATIFIED DATA SPLIT")
print("--------------------------------")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("Training delayed:", int(y_train.sum()))
print("Validation delayed:", int(y_val.sum()))
print("Test delayed:", int(y_test.sum()))


# ============================================================
# STEP 2: BLIND BASELINE AND THREE CLASSIFIERS
# ============================================================

model_results = []

# Blind baseline: predict every parcel as not delayed
blind_predictions = np.zeros(len(y_val), dtype=int)

blind_metrics = calculate_metrics(y_val, blind_predictions)

model_results.append({
    "model": "blind_all_on_time",
    **blind_metrics
})


# Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_val)

model_results.append({
    "model": "logistic_regression",
    **calculate_metrics(y_val, logistic_pred)
})


# Decision Tree: choose depth using validation F1
tree_depth_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)
    pred = tree.predict(X_val)

    tree_f1 = f1_score(y_val, pred, zero_division=0)

    tree_depth_results.append((depth, tree_f1, tree))


best_depth, best_tree_f1, best_tree = sorted(
    tree_depth_results,
    key=lambda item: (-item[1], item[0])
)[0]

tree_pred = best_tree.predict(X_val)

model_results.append({
    "model": f"decision_tree_depth_{best_depth}",
    **calculate_metrics(y_val, tree_pred)
})


# Default Random Forest
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(X_train, y_train)
forest_pred = default_forest.predict(X_val)

model_results.append({
    "model": "random_forest_default",
    **calculate_metrics(y_val, forest_pred)
})


models_df = pd.DataFrame(model_results)

models_df.to_csv(
    "outputs/part_c_models.csv",
    index=False
)

print("\nCLASSIFIER RESULTS")
print("------------------")
print(models_df.round(4).to_string(index=False))
print("\nBest Decision Tree Depth:", best_depth)


# ============================================================
# STEP 3: FEATURE IMPORTANCE AND SELECT KBEST
# ============================================================

# Random Forest feature importance using all 17 features
importance_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

importance_forest.fit(X_train, y_train)

# F-scores using training rows only
selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": importance_forest.feature_importances_
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_scores = feature_scores.sort_values("f_rank")

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)

print("\nFEATURE SCORES")
print("--------------")
print(feature_scores.round(3).to_string(index=False))


# Compare different values of k
selectk_results = []

for k in [5, 7, 10, 12, 17]:

    selectk_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    selectk_model.fit(X_train, y_train)

    pred = selectk_model.predict(X_val)

    selectk_results.append({
        "k": k,
        "val_f1": f1_score(y_val, pred, zero_division=0)
    })


selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)

print("\nSELECT KBEST RESULTS")
print("--------------------")
print(selectk_df.round(4).to_string(index=False))


# ============================================================
# STEP 4: GRID SEARCH AND RANDOMIZED SEARCH
# ============================================================

# Fit SelectKBest on training rows only
selector_7 = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)


param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}


# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_7, y_train)

grid_pred = grid_search.best_estimator_.predict(X_val_7)

grid_val_f1 = f1_score(
    y_val, grid_pred, zero_division=0
)


# Randomized Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_7, y_train)

random_pred = random_search.best_estimator_.predict(X_val_7)

random_val_f1 = f1_score(
    y_val, random_pred, zero_division=0
)


# Untuned forest using the same 7 features
default_7_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_7_forest.fit(X_train_7, y_train)

default_7_pred = default_7_forest.predict(X_val_7)

default_7_f1 = f1_score(
    y_val, default_7_pred, zero_division=0
)


tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_7_f1
    }
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)

print("\nTUNING RESULTS")
print("--------------")
print(tuning_df.round(4).to_string(index=False))


# ============================================================
# STEP 5: CHOOSE FINAL CONFIGURATION
# ============================================================

priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

winner = sorted(
    tuning_results,
    key=lambda row: (
        -row["val_f1"],
        priority[row["search"]]
    )
)[0]

chosen_from = winner["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_

elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_

else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }


final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_model.fit(X_train, y_train)

print("\nFINAL MODEL SOURCE:", chosen_from)
print("FINAL MODEL PARAMETERS:", chosen_params)


# ============================================================
# STEP 6: ALERT THRESHOLD
# ============================================================

val_probabilities = final_model.predict_proba(X_val)[:, 1]

thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_results = []

for threshold in thresholds:

    threshold_pred = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        **calculate_metrics(y_val, threshold_pred)
    })


thresholds_df = pd.DataFrame(threshold_results)

thresholds_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)

eligible_thresholds = thresholds_df[
    thresholds_df["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1


print("\nALERT THRESHOLD RESULTS")
print("-----------------------")
print(thresholds_df.round(4).to_string(index=False))

print("\nChosen Alert Threshold:", chosen_threshold)


# ============================================================
# STEP 7: FINAL TEST EVALUATION - ONLY ONCE
# ============================================================

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = calculate_metrics(
    y_test,
    test_predictions
)

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    **test_metrics
}])

final_test_df.to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)


max_depth_value = chosen_params["max_depth"]

if max_depth_value is None:
    max_depth_value = "unlimited"


final_config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": max_depth_value,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config_df.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)


print("\nFINAL TEST RESULTS")
print("------------------")
print(final_test_df.round(4).to_string(index=False))

print("\nFINAL CONFIGURATION")
print("-------------------")
print(final_config_df.to_string(index=False))


print("\nPART C COMPLETED SUCCESSFULLY!")
