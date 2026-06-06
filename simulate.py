"""
simulate.py — Generate synthetic driving session data for training and demo.
"""

import numpy as np
import pandas as pd
import random

TERRAINS = ["flat", "hilly", "urban", "highway", "mountain"]
WEATHER   = ["clear", "rain", "snow", "wind"]

def simulate_session(n_samples: int = 300, seed: int = None) -> pd.DataFrame:
    """Return a DataFrame of n_samples driving records."""
    rng = np.random.default_rng(seed)

    terrain  = rng.choice(TERRAINS, size=n_samples)
    weather  = rng.choice(WEATHER,  size=n_samples)

    # Base speed depends on terrain
    speed_base = {"flat": 80, "hilly": 60, "urban": 45, "highway": 110, "mountain": 50}
    speed = np.array([speed_base[t] for t in terrain], dtype=float)
    speed += rng.normal(0, 12, n_samples)
    speed = np.clip(speed, 5, 140)

    # Braking events (0–10)
    braking = rng.poisson(lam=3, size=n_samples).clip(0, 10)

    # Acceleration aggressiveness 0–1
    accel_agg = rng.beta(2, 5, n_samples)

    # Idle time (seconds per minute of driving)
    idle_time = rng.exponential(scale=8, size=n_samples).clip(0, 60)

    # Engine load (%)
    engine_load = 30 + 0.3 * speed + 15 * accel_agg + rng.normal(0, 5, n_samples)
    engine_load = np.clip(engine_load, 10, 100)

    # Energy consumption (Wh/km) — ground truth label
    terrain_penalty = {"flat": 0, "hilly": 18, "urban": 12, "highway": -5, "mountain": 30}
    weather_penalty = {"clear": 0, "rain": 8, "snow": 20, "wind": 10}

    energy = (
        120
        + 0.8  * speed
        + 6    * braking
        + 40   * accel_agg
        + 0.4  * idle_time
        + 0.5  * engine_load
        + np.array([terrain_penalty[t] for t in terrain])
        + np.array([weather_penalty[w] for w in weather])
        + rng.normal(0, 8, n_samples)
    )
    energy = np.clip(energy, 60, 400)

    df = pd.DataFrame({
        "speed_kmh":        np.round(speed, 1),
        "terrain":          terrain,
        "weather":          weather,
        "braking_events":   braking,
        "accel_aggression": np.round(accel_agg, 3),
        "idle_time_s":      np.round(idle_time, 1),
        "engine_load_pct":  np.round(engine_load, 1),
        "energy_wh_km":     np.round(energy, 1),
    })
    return df


if __name__ == "__main__":
    df = simulate_session(n_samples=2000, seed=42)
    out = "driving_optimizer/data/training_data.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} records → {out}")
    print(df.describe())
