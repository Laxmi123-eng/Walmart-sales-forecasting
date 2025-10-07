#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(".") 

train    = pd.read_csv(DATA_DIR/"C:/Users/kprab/OneDrive/Documents/TCD/Projects/Walmart sales forecast/train.csv",    parse_dates=["Date"])
test     = pd.read_csv(DATA_DIR/"C:/Users/kprab/OneDrive/Documents/TCD/Projects/Walmart sales forecast/test.csv",     parse_dates=["Date"])
features = pd.read_csv(DATA_DIR/"C:/Users/kprab/OneDrive/Documents/TCD/Projects/Walmart sales forecast/features.csv" , parse_dates=["Date"])
stores   = pd.read_csv(DATA_DIR/"C:/Users/kprab/OneDrive/Documents/TCD/Projects/Walmart sales forecast/stores.csv")

print(train.shape, test.shape, features.shape, stores.shape)


# In[54]:


def prep_base(df):
    out = df.merge(features, on=["Store","Date"], how="left")
    out = out.merge(stores, on="Store", how="left")
    return out

train_m = prep_base(train)
test_m  = prep_base(test)

# Clean duplicate IsHoliday columns
for df in [train_m, test_m]:
    if "IsHoliday_x" in df.columns:
        df["IsHoliday"] = df["IsHoliday_x"]
    df.drop(columns=["IsHoliday_x","IsHoliday_y"], errors="ignore", inplace=True)

# Fill NaNs in MarkDown columns (means no promotion)
for col in [f"MarkDown{i}" for i in range(1,6)]:
    if col in train_m.columns:
        train_m[col] = train_m[col].fillna(0)
        test_m[col]  = test_m[col].fillna(0)

print("✅ Merge done. NaNs (after MarkDown fix):")
print(train_m.isna().sum().sort_values(ascending=False).head(10))


# In[56]:


def add_time_features(df):
    d = df.copy()
    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.month
    d["week"] = d["Date"].dt.isocalendar().week.astype(int)
    d["quarter"] = d["Date"].dt.quarter
    d["is_month_end"] = d["Date"].dt.is_month_end.astype(int)
    d["is_month_start"] = d["Date"].dt.is_month_start.astype(int)
    d["is_year_end"] = d["Date"].dt.is_year_end.astype(int)
    d["is_year_start"] = d["Date"].dt.is_year_start.astype(int)
    return d

train_m = add_time_features(train_m)
test_m  = add_time_features(test_m)


# In[59]:


def add_lags_rolls_v2(df, group_cols=["Store", "Dept"], target="Weekly_Sales"):
    d = df.copy()
    if target in d.columns:
        d[target] = d[target].astype(float)
    else:
        # if Weekly_Sales missing (test set), create placeholder
        d[target] = np.nan

    # --- Simple lags ---
    d["lag_1"]  = d.groupby(group_cols)[target].shift(1)
    d["lag_2"]  = d.groupby(group_cols)[target].shift(2)
    d["lag_4"]  = d.groupby(group_cols)[target].shift(4)
    d["lag_8"]  = d.groupby(group_cols)[target].shift(8)
    d["lag_52"] = d.groupby(group_cols)[target].shift(52)

    # --- Rolling means ---
    d["roll_mean_4"]  = d.groupby(group_cols)[target].transform(lambda s: s.shift(1).rolling(4).mean())
    d["roll_mean_8"]  = d.groupby(group_cols)[target].transform(lambda s: s.shift(1).rolling(8).mean())
    d["roll_mean_52"] = d.groupby(group_cols)[target].transform(lambda s: s.shift(1).rolling(52).mean())

    # --- Rolling stds ---
    d["roll_std_4"] = d.groupby(group_cols)[target].transform(lambda s: s.shift(1).rolling(4).std())
    d["roll_std_8"] = d.groupby(group_cols)[target].transform(lambda s: s.shift(1).rolling(8).std())

    return d

train_feat = add_lags_rolls_v2(train_m)
test_feat  = add_lags_rolls_v2(test_m)

print(" Lag and rolling features added.")


# In[63]:


# Fill any numeric NaNs (from first few lag weeks)
for df in [train_feat, test_feat]:
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical columns (e.g., Type = A/B/C)
cat_cols = train_feat.select_dtypes(exclude=["number","datetime64[ns]"]).columns
for c in cat_cols:
    cats = pd.Categorical(train_feat[c].astype(str))
    train_feat[c] = cats.codes
    mapping = {cat: code for code, cat in enumerate(cats.categories)}
    test_feat[c] = test_feat[c].astype(str).map(mapping).fillna(-1).astype(int)

print(" Encoded categoricals and filled remaining NaNs.")


# In[65]:


X = train_feat.drop(columns=["Weekly_Sales", "Date"], errors="ignore")
y = train_feat["Weekly_Sales"].values
X_test = test_feat.drop(columns=["Date"], errors="ignore")

# Align columns
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# Weighted holidays (5x)
weights = np.where(train_feat["IsHoliday"].astype(int)==1, 5, 1)

def wmae(y_true, y_pred, w):
    return (np.abs(y_true - y_pred) * w).sum() / w.sum()


# In[71]:


get_ipython().system('pip install lightgbm')


# In[79]:


import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

tscv = TimeSeriesSplit(n_splits=3)
scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_tr, w_val = weights[tr_idx], weights[val_idx]

    print(f"\n🔹 Training Fold {fold+1}")

    model = lgb.LGBMRegressor(
        objective="mae",
        n_estimators=3000,            # we can set higher, early stopping will cut it
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    s = wmae(y_val, preds, w_val)
    print(f"✅ Fold {fold+1} WMAE: {s:,.2f}")
    scores.append(s)

print(f"\n🎯 Average WMAE: {np.mean(scores):,.2f}")


# In[88]:


final_model = lgb.LGBMRegressor(
    objective="mae",
    n_estimators=1000,       # or the number you used in Option 3
    learning_rate=0.05,
    num_leaves=31,           # smaller leaves = faster
    max_depth=8,             # from Option 3
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("⏳ Training final LightGBM model on all data...")
final_model.fit(X, y, sample_weight=weights)
print("✅ Final model trained successfully!")

# --- Predict test set (no best_iteration since no early stopping) ---
test_pred = final_model.predict(X_test)

# --- Build submission file ---
test_feat["Weekly_Sales"] = test_pred
test_feat["Id"] = (
    test_feat["Store"].astype(str) + "_" +
    test_feat["Dept"].astype(str) + "_" +
    test_feat["Date"].dt.date.astype(str)
)

submission = test_feat[["Id", "Weekly_Sales"]]
submission.to_csv("submission_lightgbm_fast.csv", index=False)
print("submission_lightgbm_fast.csv created successfully!")


# In[90]:


import matplotlib.pyplot as plt

# pick the last 500 training points as a validation preview
pred_train = final_model.predict(X)
plt.figure(figsize=(10,4))
plt.plot(y[-500:], label="Actual")
plt.plot(pred_train[-500:], label="Predicted")
plt.title("Actual vs Predicted (last 500 weeks)")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




