"""
train_model.py — Train a GradientBoosting regressor to predict energy usage.
Saves the trained pipeline as models/efficiency_model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data.simulate import simulate_session

# ── 1. Data ──────────────────────────────────────────────────────────────────
print("Generating training data …")
df = simulate_session(n_samples=5000, seed=0)

FEATURES = ["speed_kmh", "terrain", "weather", "braking_events",
            "accel_aggression", "idle_time_s", "engine_load_pct"]
TARGET   = "energy_wh_km"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 2. Pre-processing ─────────────────────────────────────────────────────────
numeric_features  = ["speed_kmh","braking_events","accel_aggression",
                     "idle_time_s","engine_load_pct"]
cat_features      = ["terrain","weather"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                          numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),    cat_features),
])

# ── 3. Model ──────────────────────────────────────────────────────────────────
model = Pipeline([
    ("prep",  preprocessor),
    ("gbr",   GradientBoostingRegressor(
                  n_estimators=200,
                  max_depth=4,
                  learning_rate=0.08,
                  subsample=0.85,
                  random_state=42)),
])

print("Training …")
model.fit(X_train, y_train)

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
cv_r2  = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

print(f"\n── Test Results ─────────────────────────")
print(f"  MAE  : {mae:.2f} Wh/km")
print(f"  R²   : {r2:.4f}")
print(f"  CV R²: {cv_r2:.4f}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
os.makedirs("driving_optimizer/models", exist_ok=True)
with open("driving_optimizer/models/efficiency_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved → driving_optimizer/models/efficiency_model.pkl")
