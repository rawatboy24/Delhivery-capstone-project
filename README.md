# \## Model Results (Regression)

# 

# \### Baseline Model

# \- Baseline RMSE: 18.86 hours

# \- Baseline R²: -0.0212

# 

# \### Model Comparison

# \- Linear Regression RMSE: 4.891

# \- Decision Tree RMSE: 6.280

# \- Random Forest RMSE: 5.468

# \- Gradient Boosting RMSE: 4.917

# 

# Best validation model: Linear Regression.

# 

# \### Decision Tree Analysis

# \- Depth 1: High Bias (underfitting).

# \- Unlimited depth: High Variance (overfitting).

# \- Best depth: 6.

# 

# \### Gradient Boosting Learning Rate

# \- 0.01: R² = 0.713 (too slow).

# \- 0.1: R² = 0.931 (best).

# \- 1.0: R² = 0.844 (too aggressive).

# 

# \### Cross Validation

# \- Linear Regression Mean R²: 0.926

# \- Gradient Boosting Mean R²: 0.935

# 

# \### Final Test Results

# \- Selected Model: Linear Regression

# \- MAE: 3.567 hours

# \- RMSE: 4.524 hours

# \- R²: 0.949



\## Model Results (Classification)



\### 1. Baseline Model

\- Blind baseline accuracy: 0.7667 (76.67%).

\- The baseline predicts every shipment as on-time.

\- It fails to identify any delayed shipment (recall = 0).

\- Therefore, accuracy alone can be misleading for an imbalanced classification problem.



\### 2. Classifier Comparison



| Model | Accuracy | Precision | Recall | F1 |

|---|---:|---:|---:|---:|

| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |

| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |

| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |



Logistic Regression achieved the highest F1 score among the initial classifiers.



\### 3. Feature Importance and Noise Features

\- The top 7 features by F-score are distance\_km, hub\_tier, hub\_daily\_capacity, n\_stops, promised\_hours, rain\_flag, and sale\_event\_flag.

\- All five noise features are outside the top 7 F-score rankings.

\- Combined Random Forest importance of the five noise features: 0.225 (22.5%).

\- This indicates that the noise features still contribute some importance to the Random Forest model.



\### 4. SelectKBest Results



| k | Validation F1 |

|---:|---:|

| 5 | 0.6988 |

| 7 | 0.8333 |

| 10 | 0.7857 |

| 12 | 0.8049 |

| 17 | 0.7848 |



The best result was achieved with k=7 features. Therefore, 7 features were selected for tuning and the final model.



\### 5. Hyperparameter Tuning



| Search | CV Accuracy | Validation F1 |

|---|---:|---:|

| Grid Search | 0.8952 | 0.7750 |

| Randomized Search | 0.8905 | 0.7848 |

| Default Forest (7 features) | — | 0.8333 |



The default Random Forest with 7 selected features performed better on validation data than both tuned models. Therefore, the default configuration was selected.



Validation results are based on 180 rows, so small differences may be affected by the limited validation sample size.



\### 6. Alert Threshold Selection

\- Selected threshold: 0.5.

\- Validation precision: 0.7955.

\- Validation recall: 0.8333.

\- Validation F1: 0.8140.



The threshold of 0.5 was selected because it was the highest tested threshold that maintained recall of at least 0.80.



A lower threshold catches more delayed shipments but creates more false alerts. A higher threshold reduces false alerts but may miss more delayed shipments.



\### 7. Final Test Results

\- Selected model: Random Forest with SelectKBest (k=7).

\- Number of trees: 100.

\- Maximum depth: Unlimited.

\- Minimum samples per leaf: 1.

\- Alert threshold: 0.5.



| Metric | Test Result |

|---|---:|

| Accuracy | 0.8667 |

| Precision | 0.6889 |

| Recall | 0.7561 |

| F1 Score | 0.7209 |



Test recall (0.7561) was lower than validation recall (0.8333). The test set contained only 41 delayed shipments, so each missed delayed shipment has a noticeable effect on recall.



The test set was used only for the final evaluation and not for model tuning.

