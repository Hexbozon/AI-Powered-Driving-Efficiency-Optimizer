# DriveIQ — AI Driving Efficiency Optimizer

An end-to-end machine learning project that simulates driving data, trains a
Gradient Boosting model to predict energy consumption, and provides a live
interactive dashboard with personalized driving tips.

---

## Project Structure

```
driving_optimizer/
├── data/
│   └── simulate.py          ← Synthetic data generator
├── models/
│   └── efficiency_model.pkl ← Trained GBR model (auto-created)
├── train_model.py           ← Model training + evaluation
├── tips_engine.py           ← Rule + ML-based tip generator
├── predictor.py             ← Inference API + CLI demo
├── dashboard.html           ← Interactive browser dashboard
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
Output:
```
MAE  : 6.79 Wh/km
R²   : 0.8926
CV R²: 0.8838
Model saved → models/efficiency_model.pkl
```

### 3. Run CLI inference
```bash
python predictor.py
```

### 4. Open the dashboard
Open `dashboard.html` in any browser — no server needed.

---

## Features

| Component         | Details |
|-------------------|---------|
| **ML Model**      | GradientBoostingRegressor (200 trees, depth=4, lr=0.08) |
| **Pipeline**      | StandardScaler + OneHotEncoder via ColumnTransformer |
| **R² Score**      | 0.89 on held-out test set |
| **Data**          | 5,000 synthetic sessions (speed, terrain, weather, braking, etc.) |
| **Tips Engine**   | 12+ rule-based tips mapped to severity (critical / warning / info) |
| **Dashboard**     | Interactive sliders, efficiency ring, timeline chart, feature importance |

---

## Input Features

| Feature              | Type        | Range / Values |
|----------------------|-------------|----------------|
| `speed_kmh`          | Numeric     | 5–140 km/h |
| `terrain`            | Categorical | flat, hilly, urban, highway, mountain |
| `weather`            | Categorical | clear, rain, snow, wind |
| `braking_events`     | Integer     | 0–10 |
| `accel_aggression`   | Float       | 0.0–1.0 |
| `idle_time_s`        | Float       | 0–60 s/min |
| `engine_load_pct`    | Float       | 10–100% |

**Target:** `energy_wh_km` — energy used per km (Wh/km)

---

## Extending the Project

- **Real data**: swap `simulate.py` with a CSV from your OBD-II reader or GPS logger.
- **Flask API**: wrap `predictor.predict()` in a `/predict` endpoint.
- **TensorFlow**: replace the GBR in `train_model.py` with a Keras Dense network.
- **EV range**: multiply `energy_wh_km` × route distance ÷ battery capacity (kWh).
