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

