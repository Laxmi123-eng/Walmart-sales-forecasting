#  Walmart Sales Forecasting using LightGBM

##  End-to-End Time Series Forecasting with Feature Engineering, Validation & Log-Transformed Target

---

###  Overview

This project focuses on building an **accurate and interpretable sales forecasting model** for Walmart stores using historical weekly sales data.  
The main goal is to predict **future weekly sales** across multiple stores and departments by leveraging a combination of **temporal, economic, and promotional features**.

This project demonstrates real-world **time-series modeling**, **data preprocessing**, and **gradient boosting** using **LightGBM**, a high-performance algorithm optimized for speed and accuracy.

---

##  Key Objectives

1. Forecast **weekly sales** for each `(Store, Dept)` combination.  
2. Engineer meaningful **temporal and lag-based features** to capture sales patterns.  
3. Handle **missing data**, **holiday effects**, and **economic indicators**.  
4. Apply a **log-transform target** to stabilize predictions and eliminate negative outputs.  
5. Validate results using **TimeSeriesSplit** and out-of-sample testing to ensure generalization.

---

##  Dataset Description

| File | Description |
|------|--------------|
| **train.csv** | Historical weekly sales per store & department |
| **test.csv** | Weeks requiring predictions (no Weekly_Sales column) |
| **features.csv** | Economic data (Temperature, Fuel_Price, CPI, Unemployment, MarkDowns) |
| **stores.csv** | Metadata for each store (Type, Size) |

---

##  Project Pipeline

###  Data Preparation
- All datasets are merged on `Store` and `Date`.
- `MarkDown1`–`MarkDown5` columns are filled with 0 (no promotion when missing).
- Duplicate `IsHoliday` columns (from multiple merges) are unified.
- `Store` and `Dept` serve as entity identifiers.

###  Feature Engineering
To extract time-based patterns and store-level signals:
- **Temporal features:**  
  `year`, `month`, `week`, `quarter`,  
  `is_month_end`, `is_month_start`, `is_year_end`, `is_year_start`.
- **Lag features:** previous `1, 2, 4, 8, 52` weeks of sales.  
  Captures recent performance and yearly seasonality.
- **Rolling statistics:**  
  - `roll_mean_4`, `roll_mean_8`, `roll_mean_52`  
  - `roll_std_4`, `roll_std_8`  
  These capture smoothed trends and volatility.
- **Categorical encoding:**  
  Store types and holidays are label-encoded for the model.

###  Handling Missing Values
- Lags and rolling windows naturally generate NaNs at the start of each series.  
  → Filled with group medians or 0 for stability.  
- Infinite values (e.g., from division or anomalies) replaced with NaN, then filled.

---

##  Target Transformation (Log-Transform)

Sales vary widely between stores and departments, creating a **highly skewed target distribution**.  
To stabilize it and prevent negative forecasts, we transform the target variable as:

\[
y = \log(1 + \text{Weekly Sales})
\]

After prediction, results are transformed back using:

\[
\hat{y} = e^{\hat{y}} - 1
\]

 This guarantees **all forecasts are positive** and reduces the impact of outliers.

---

##  Model Training (LightGBM)

**Model:** `LGBMRegressor`  
**Framework:** Gradient Boosting Decision Trees  
**Objective:** Mean Absolute Error (MAE)

### Model Parameters
```python
n_estimators = 1000
learning_rate = 0.05
num_leaves = 31
max_depth = 8
subsample = 0.8
colsample_bytree = 0.8
objective = "mae"
random_state = 42
