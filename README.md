# Delhivery Capstone Project

## Project Overview

This project analyzes shipment delivery performance and builds machine learning models to predict delivery time and identify shipments at risk of delay.

The project covers data cleaning, exploratory data analysis, regression, classification, hub segmentation, model packaging, and operational recommendations.

## Part A - Data Cleaning and Exploratory Data Analysis

### 1. Data Quality Checks

- Raw shipment records: 1,224
- Duplicate records identified: 24
- Cleaned shipment records: 1,200
- Delayed shipments: 278
- Overall delay rate: 23.17%
- Missing values: 0 in all 17 model features and target.

### 2. Data Cleaning

The following preprocessing steps were performed:

- Removed duplicate shipment records.
- Standardized service type values.
- Handled missing payment modes.
- Converted declared values into numeric format.
- Handled invalid dispatch dates.
- Treated extreme parcel weights as missing values.
- Applied median imputation to missing numeric values.

### 3. Feature Engineering

Created 17 features for machine learning, including:

- Promised delivery hours.
- Distance and parcel weight.
- Log-transformed declared value.
- Number of stops.
- Rain and sale-event indicators.
- Weekend and dispatch-hour features.
- Hub tier and daily capacity.
- Five noise features.

### 4. EDA Outputs

The following output files were generated:

- `outputs/part_a_checks.csv`
- `outputs/part_a_correlations.csv`
- `outputs/part_a_delay_by_group.csv`

These outputs contain data quality checks, feature correlations, and delay summaries across operational groups.

---

## Part B - Delivery Time Regression

### 1. Baseline Model

- Baseline RMSE: 18.86 hours
- Baseline R²: -0.0212

### 2. Model Comparison

| Model | Validation RMSE |
|---|---:|
| Linear Regression | 4.891 |
| Decision Tree | 6.280 |
| Random Forest | 5.468 |
| Gradient Boosting | 4.917 |

**Best validation model: Linear Regression.**

### 3. Decision Tree Analysis

- Depth 1: High bias (underfitting).
- Unlimited depth: High variance (overfitting).
- Best depth: 6.

### 4. Gradient Boosting Learning Rate

| Learning Rate | Validation R² |
|---|---:|
| 0.01 | 0.713 |
| 0.1 | 0.931 |
| 1.0 | 0.844 |

Learning rate 0.1 achieved the best validation result.

### 5. Cross Validation

| Model | Mean CV R² |
|---|---:|
| Linear Regression | 0.926 |
| Gradient Boosting | 0.935 |

### 6. Final Test Results

- Selected model: Linear Regression
- MAE: 3.567 hours
- RMSE: 4.524 hours
- R²: 0.949

The final Linear Regression model achieved an R² of 0.949 on the test set.

---

## Part C - Shipment Delay Classification

### 1. Baseline Model

- Baseline accuracy: 0.7667 (76.67%).
- The baseline predicts every shipment as on-time.
- It fails to identify delayed shipments (recall = 0).

Accuracy alone can be misleading for an imbalanced classification problem.

### 2. Classifier Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

Logistic Regression achieved the highest F1 score among the initial classifiers.

### 3. Feature Importance and Noise Features

The top 7 features by F-score are:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

- All five noise features are outside the top 7 F-score rankings.
- Combined Random Forest importance of the five noise features: 0.225 (22.5%).

### 4. SelectKBest Results

| Number of Features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

The best result was achieved with k=7 features.

### 5. Hyperparameter Tuning

| Search Method | CV Accuracy | Validation F1 |
|---|---:|---:|
| Grid Search | 0.8952 | 0.7750 |
| Randomized Search | 0.8905 | 0.7848 |
| Default Forest (7 features) | — | 0.8333 |

The default Random Forest with 7 selected features performed better on validation data than both tuned models.

Validation results are based on 180 rows, so small differences may be affected by the limited validation sample size.

### 6. Alert Threshold Selection

- Selected threshold: 0.5
- Validation precision: 0.7955
- Validation recall: 0.8333
- Validation F1: 0.8140

The threshold of 0.5 was selected because it was the highest tested threshold that maintained recall of at least 0.80.

A lower threshold catches more delayed shipments but creates more false alerts. A higher threshold reduces false alerts but may miss more delayed shipments.

### 7. Final Test Results

Selected configuration:

- Model: Random Forest with SelectKBest (k=7)
- Number of trees: 100
- Maximum depth: Unlimited
- Minimum samples per leaf: 1
- Alert threshold: 0.5

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 Score | 0.7209 |

Test recall was lower than validation recall. The test set contained only 41 delayed shipments, so each missed delayed shipment has a noticeable effect on recall.

The test set was used only for final evaluation and not for model tuning.

---

## Part D - Hub Segments, Packaging and Operations Playbook

### 1. Hub Segmentation

Hub profiles were created using shipment count, mean delivery hours, delay rate, mean distance, and mean number of stops.

KMeans clustering was performed using StandardScaler with:

- K = 3
- random_state = 42
- n_init = 10

#### Cluster Separation Comparison

| K | Separation Ratio | Cluster Sizes |
|---:|---:|---|
| 2 | 2.894 | 15, 9 |
| 3 | 5.393 | 9, 8, 7 |
| 4 | 6.305 | 8, 7, 5, 4 |
| 5 | 6.887 | 8, 7, 4, 3, 2 |

K=3 was selected because it provides three practical operational groups without creating very small clusters.

#### Segment Profiles

| Segment | Hubs | Mean Delivery Hours | Delay Rate | Mean Distance (km) | Mean Stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.24 | 0.0257 | 188.93 | 55.37 |
| Watch | 8 | 56.81 | 0.1550 | 314.18 | 70.72 |
| Escalate | 9 | 64.65 | 0.4600 | 470.48 | 84.34 |

#### Tier and Segment Mismatches

| Hub | City | Tier | Actual Segment |
|---|---|---|---|
| HUB07 | Kolkata | Tier1 | Watch |
| HUB13 | Nagpur | Tier2 | Escalate |
| HUB15 | Kochi | Tier2 | Escalate |
| HUB22 | Dehradun | Tier3 | Watch |

These mismatches show that hub tier alone does not describe actual operational delay risk.

### 2. Model Packaging and Prediction

The final classification model is saved as:

`models/delay_model.pkl`

Additional supporting files:

- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

The model was rebuilt using the same stratified train/validation/test split as Part C.

#### Reload Verification

| Metric | Result |
|---|---:|
| Test rows scored | 180 |
| Reload maximum probability difference | 0.0 |
| Alert threshold | 0.5 |

The saved model and in-memory model produced identical test probabilities.

The prediction script processes 48 new shipments and writes:

`outputs/new_shipment_actions.csv`

#### New Shipment Prediction Results

| Action | Shipments |
|---|---:|
| No action | 40 |
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| Total flagged | 8 |

The prediction output contains 48 unique shipment IDs and zero missing values.

The CSV was generated twice and both SHA256 hashes matched, confirming deterministic output.

### 3. Operations Recommendation

The nine Escalate hubs should receive priority operational attention:

- HUB13 - Nagpur
- HUB15 - Kochi
- HUB17 - Patna
- HUB18 - Ranchi
- HUB19 - Guwahati
- HUB20 - Raipur
- HUB21 - Bhubaneswar
- HUB23 - Siliguri
- HUB24 - Agartala

The segment-level delay rate for Escalate hubs is 46%.

On the Part C test set:

- Precision = 0.6889. Approximately 6.9 out of every 10 flagged shipments were actually delayed.
- Recall = 0.7561. Approximately 7.6 out of every 10 delayed shipments were identified.

Flagged shipments at Escalate hubs should be sent to the hub manager. Other flagged shipments should receive proactive customer calls.

The four tier/segment mismatches should be reviewed before making permanent hub tier changes.

### 4. Production Plan

| Production Characteristic | Repository Implementation |
|---|---|
| Accuracy | `outputs/part_c_final_test.csv` stores final classification metrics. |
| Reliability | `outputs/part_d_reload_check.csv` verifies saved model predictions. |
| Reproducibility | Fixed random_state=42 and deterministic prediction output. |
| Maintainability | Shared preprocessing functions are stored in `src/shipment_prep.py`. |
| Consistency | `feature_order.csv` and `impute_values.csv` preserve training preprocessing. |
| Monitorability | `outputs/new_shipment_actions.csv` records shipment-level predictions and actions. |

#### Monitoring Triggers

The operations team should monitor these quantities weekly:

1. Actual delay rate above 30% should trigger an operational review.
2. Model recall below 0.75 on matured shipments should trigger investigation.
3. Mean shipment distance changing by more than 20% from the training baseline should trigger a data drift review.
4. Missing or invalid feature values above 5% should trigger a data quality investigation.

These are proposed operational alert thresholds and should be reviewed after collecting real production data.

#### Retraining Schedule

- Retrain the model quarterly.
- Retrain earlier after major changes in hub operations, delivery routes, or service policies.
- Compare sale-event performance during 15–28 September with a comparable sale-event period, rather than ordinary non-sale weeks.
- Monitor changes in distance, number of stops, hub capacity, service type, and missing-value rates.

### 5. Limitations

1. The project uses a limited synthetic dataset and may not represent real delivery operations.
2. The test set contains only 41 delayed shipments, so classification metrics may change with new data.
3. The selected alert threshold was evaluated on one validation split and may need recalibration in production.
4. Hub segments are based on historical aggregate profiles and may change when operating conditions change.
5. The model does not use live traffic, weather, or real-time hub workload information.

---

## Project Structure

```text
Delhivery-capstone-project/
│
├── data/
├── models/
├── outputs/
├── src/
│   └── shipment_prep.py
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/rawatboy24/Delhivery-capstone-project.git
cd Delhivery-capstone-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Data

```bash
python generate_data.py
```

### 4. Run Part A - Data Cleaning and EDA

```bash
python part_a_clean_eda.py
```

### 5. Run Part B - Regression

```bash
python part_b_regression.py
```

### 6. Run Part C - Classification

```bash
python part_c_classification.py
```

### 7. Run Part D - Hub Segmentation

```bash
python part_d_segments.py
```

### 8. Package the Model

```bash
python package_model.py
```

### 9. Generate New Shipment Predictions

```bash
python predict_delay.py
```

The `package_model.py` script regenerates `models/delay_model.pkl` and the supporting model files.

The committed pickle is intended to be loaded with the pinned scikit-learn version. If the installed version differs, regenerate the model by running `python package_model.py`.

On Windows, the `py` launcher can be used instead of `python`.

---

## Requirements

The project uses the following pinned dependencies:

- NumPy 2.5.4
- pandas 3.1.0
- scikit-learn 1.9.1
- joblib 1.6.0
