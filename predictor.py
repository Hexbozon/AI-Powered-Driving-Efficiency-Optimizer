"""
predictor.py — Load the trained model and run inference with tips.
"""

import os, pickle
import pandas as pd
from tips_engine import generate_tips, efficiency_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "efficiency_model.pkl")

_model = None

def _load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict(inputs: dict) -> dict:
    """
    inputs: {speed_kmh, terrain, weather, braking_events,
             accel_aggression, idle_time_s, engine_load_pct}
    returns: {energy_wh_km, efficiency_score, tips}
    """
    model = _load_model()
    row = pd.DataFrame([inputs])
    energy = float(model.predict(row)[0])
    score  = efficiency_score(energy)
    tips   = generate_tips(energy, inputs)

    return {
        "energy_wh_km":     round(energy, 1),
        "efficiency_score": score,
        "tips": [
            {
                "category": t.category,
                "severity": t.severity,
                "icon":     t.icon,
                "message":  t.message,
                "saving_pct": t.saving_pct,
            }
            for t in tips
        ],
    }


# ── Quick CLI demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "speed_kmh":        95,
        "terrain":          "hilly",
        "weather":          "rain",
        "braking_events":   5,
        "accel_aggression": 0.55,
        "idle_time_s":      22,
        "engine_load_pct":  68,
    }
    result = predict(sample)
    print(f"\nPredicted energy : {result['energy_wh_km']} Wh/km")
    print(f"Efficiency score : {result['efficiency_score']}/100\n")
    print("── Driving Tips ─────────────────────────────────")
    for t in result["tips"]:
        print(f"  [{t['severity'].upper():8s}] {t['icon']}  {t['message']}")
        if t["saving_pct"] > 0:
            print(f"             ↳ Potential saving: {t['saving_pct']}%")
