"""
tips_engine.py — Rule + ML-based driving tip generator.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class Tip:
    category: str      # "speed" | "braking" | "idle" | "accel" | "terrain" | "general"
    severity: str      # "info" | "warning" | "critical"
    icon: str
    message: str
    saving_pct: float  # estimated % energy saving if followed


def generate_tips(prediction: float, inputs: dict) -> List[Tip]:
    """
    Return a ranked list of actionable tips given the model inputs and
    the predicted energy_wh_km.
    """
    tips: List[Tip] = []

    speed        = inputs.get("speed_kmh", 0)
    braking      = inputs.get("braking_events", 0)
    accel_agg    = inputs.get("accel_aggression", 0)
    idle_time    = inputs.get("idle_time_s", 0)
    engine_load  = inputs.get("engine_load_pct", 0)
    terrain      = inputs.get("terrain", "flat")
    weather      = inputs.get("weather", "clear")

    # ── Speed ─────────────────────────────────────────────────────────────────
    if speed > 110:
        tips.append(Tip("speed","critical","🚀",
            f"You're travelling at {speed:.0f} km/h. Reducing to 100 km/h cuts "
            f"air-resistance drag by ~19% and can save up to 15% energy.", 15.0))
    elif speed > 90:
        tips.append(Tip("speed","warning","⚡",
            f"Speed of {speed:.0f} km/h is above the optimal efficiency band "
            f"(80–90 km/h). Easing off slightly saves ~8% energy.", 8.0))
    elif speed < 30 and terrain in ("highway","flat"):
        tips.append(Tip("speed","info","🐢",
            "Very low speed on flat/highway terrain increases energy per km. "
            "Maintain a steadier pace around 60–80 km/h.", 5.0))

    # ── Braking ───────────────────────────────────────────────────────────────
    if braking >= 7:
        tips.append(Tip("braking","critical","🛑",
            f"{braking} braking events detected — likely frequent hard braking. "
            "Anticipate stops earlier and coast to a gradual halt to save up to 12%.", 12.0))
    elif braking >= 4:
        tips.append(Tip("braking","warning","⚠️",
            f"{braking} braking events — moderate. Try to increase following distance "
            "and reduce sudden stops. Estimated saving: ~6%.", 6.0))

    # ── Acceleration aggressiveness ───────────────────────────────────────────
    if accel_agg > 0.65:
        tips.append(Tip("accel","critical","🏎️",
            f"Aggressive acceleration score: {accel_agg:.0%}. Gradual acceleration "
            "uses 30–40% less energy. Aim for smooth, progressive throttle inputs.", 14.0))
    elif accel_agg > 0.40:
        tips.append(Tip("accel","warning","📈",
            f"Moderate acceleration aggressiveness ({accel_agg:.0%}). Easing into "
            "speed transitions can shave ~7% off energy use.", 7.0))

    # ── Idle time ─────────────────────────────────────────────────────────────
    if idle_time > 30:
        tips.append(Tip("idle","critical","⏸️",
            f"Idle time: {idle_time:.0f}s/min is very high. Turn off the engine "
            "when stopped >60 s — saves up to 10% on urban routes.", 10.0))
    elif idle_time > 15:
        tips.append(Tip("idle","warning","⏱️",
            f"Idle time of {idle_time:.0f}s/min. Reduce unnecessary engine idling "
            "when parked or in traffic queues. Saves ~4%.", 4.0))

    # ── Engine load ───────────────────────────────────────────────────────────
    if engine_load > 75:
        tips.append(Tip("accel","warning","🔧",
            f"Engine load at {engine_load:.0f}%. Check tyre pressure and reduce "
            "A/C usage. Optimal load is 40–60%. Saves ~5%.", 5.0))

    # ── Terrain specific ──────────────────────────────────────────────────────
    if terrain == "mountain":
        tips.append(Tip("terrain","info","⛰️",
            "Mountain terrain detected. Use engine braking on descents and "
            "build momentum before climbs rather than accelerating mid-hill.", 8.0))
    elif terrain == "hilly":
        tips.append(Tip("terrain","info","🏔️",
            "Hilly route — anticipate crests and use kinetic energy to carry "
            "speed over the top. Saves repeated acceleration energy.", 5.0))

    # ── Weather ───────────────────────────────────────────────────────────────
    if weather == "snow":
        tips.append(Tip("terrain","warning","❄️",
            "Snow conditions add significant rolling resistance. Keep speeds "
            "conservative (40–60 km/h) and avoid wheelspin on acceleration.", 0.0))
    elif weather == "rain":
        tips.append(Tip("terrain","info","🌧️",
            "Rain increases rolling resistance by ~8%. Maintain safe following "
            "distance and brake earlier to avoid hard stops.", 0.0))
    elif weather == "wind":
        tips.append(Tip("terrain","info","💨",
            "Headwind detected. Keep speed lower and use slipstreaming where "
            "possible. Tailwind? You can ease off the throttle.", 0.0))

    # ── Overall efficiency score ───────────────────────────────────────────────
    if prediction < 150:
        tips.append(Tip("general","info","🌿",
            f"Great efficiency! Predicted {prediction:.0f} Wh/km — "
            "you're in the top tier. Keep it up!", 0.0))
    elif prediction > 250:
        tips.append(Tip("general","critical","🔥",
            f"High energy use: {prediction:.0f} Wh/km. Combining the tips above "
            "could reduce this by 20–35%.", 0.0))

    # Sort: critical first, then by saving potential
    order = {"critical": 0, "warning": 1, "info": 2}
    tips.sort(key=lambda t: (order[t.severity], -t.saving_pct))
    return tips


def efficiency_score(prediction: float) -> int:
    """Map predicted Wh/km to a 0–100 score (higher = more efficient)."""
    # 80 Wh/km → 100,  350 Wh/km → 0
    score = 100 - int((prediction - 80) / (350 - 80) * 100)
    return max(0, min(100, score))
