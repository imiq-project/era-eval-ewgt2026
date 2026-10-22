"""
ERA sensitivity evaluation — condition grid.

8 canned proactive_context dicts (C1..C8) following the EVAL_SPEC fractional
factorial design. Each dict matches the shape ERA reads in
prompts/era_prompts.py::_format_environmental_context, verified against the
producers in orchestrator.py (_quick_weather_check, _quick_traffic_check,
_quick_parking_check, _quick_air_quality_check, _add_time_of_day_context).

Field-name correctness is critical — a key mismatch silently drops the
dimension from ERA's view. Confirmed producer keys:
  weather:     temperature, humidity, wind_speed, condition, conditions,
               sensor_nearby, plus eval-only: feels_like, rain_mm,
               snowfall_cm, is_day
  traffic:     status, delay_minutes, incident_count, incidents (list)
  parking:     total_available, total_capacity, status, has_nearby_parking
  air_quality: pm25, pm10, no2 (plus optional aqi/quality)
  top-level:   current_time, current_hour, day_of_week, is_weekend, is_dark
"""

from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Numeric levels — single source of truth.
# Used both for building proactive_context AND for downstream metric scripts
# (monotonicity, Spearman ρ).
# ---------------------------------------------------------------------------
LEVELS: Dict[str, Dict[str, Any]] = {
    # weather levels — rain in mm/h, snow in cm/h
    "rain_clear": 0,
    "rain_light": 2,
    "rain_heavy": 10,
    "snow_heavy": 5,           # cm/h, paired with sub-zero temp + condition="snow"

    "wind_calm": 5,            # km/h
    "wind_breezy": 18,         # km/h, used in rain/snow conditions

    "temp_warm": 20,
    "temp_cool": 5,
    "temp_freezing": -5,

    # air quality
    "pm25_good": 10,           # WHO guideline daily mean is 15
    "pm25_poor": 60,           # ERA prompt calls >55 "unhealthy"
    "pm10_good": 20,
    "pm10_poor": 90,
    "no2_good": 20,
    "no2_poor": 80,

    # traffic
    "traffic_free_delay_min": 0,
    "traffic_free_incidents": 0,
    "traffic_cong_delay_min": 12,
    "traffic_cong_incidents": 1,

    # parking — N/100 spots
    "parking_avail_count": 80,
    "parking_scarce_count": 3,
    "parking_capacity": 100,

    # humidity (constant — not part of the factorial)
    "humidity_dry": 50,
    "humidity_wet": 90,
}


# ---------------------------------------------------------------------------
# Per-condition level assignments — encodes the EVAL_SPEC table directly.
# Used by metric scripts to compute monotonicity / Spearman ρ.
# ---------------------------------------------------------------------------
CONDITION_LEVELS: Dict[str, Dict[str, Any]] = {
    # ID:       weather       temp_C     time_label    pm25_level     traffic     parking
    "C1": {"weather": "clear",      "temp_c":  20, "time": "noon",    "pm25": "good", "traffic": "free",      "parking": "available"},
    "C2": {"weather": "clear",      "temp_c":   5, "time": "22:00",   "pm25": "good", "traffic": "congested", "parking": "scarce"},
    "C3": {"weather": "light_rain", "temp_c":   5, "time": "noon",    "pm25": "good", "traffic": "congested", "parking": "scarce"},
    "C4": {"weather": "light_rain", "temp_c":  20, "time": "22:00",   "pm25": "poor", "traffic": "free",      "parking": "available"},
    "C5": {"weather": "heavy_rain", "temp_c":   5, "time": "noon",    "pm25": "poor", "traffic": "free",      "parking": "scarce"},
    "C6": {"weather": "heavy_rain", "temp_c":  20, "time": "22:00",   "pm25": "good", "traffic": "congested", "parking": "available"},
    "C7": {"weather": "snow",       "temp_c":  -5, "time": "noon",    "pm25": "good", "traffic": "free",      "parking": "scarce"},
    "C8": {"weather": "snow",       "temp_c":  -5, "time": "22:00",   "pm25": "poor", "traffic": "congested", "parking": "available"},
}


# ---------------------------------------------------------------------------
# Helpers — build proactive_context fragments from levels.
# ---------------------------------------------------------------------------

def _build_weather(weather_level: str, temp_c: int, is_day: bool) -> Dict[str, Any]:
    """Build a weather dict matching the producer shape ERA reads.

    Producer keys (orchestrator._quick_weather_check):
        temperature, humidity, wind_speed, condition, conditions, sensor_nearby
    Eval-only keys ERA prompt also reads (era_prompts.py:308-330):
        feels_like, rain_mm, snowfall_cm, is_day
    """
    rain_mm = 0
    snowfall_cm = 0
    wind = LEVELS["wind_calm"]
    humidity = LEVELS["humidity_dry"]

    if weather_level == "clear":
        condition = "clear"
    elif weather_level == "light_rain":
        condition = "light rain"
        rain_mm = LEVELS["rain_light"]
        wind = LEVELS["wind_breezy"]
        humidity = LEVELS["humidity_wet"]
    elif weather_level == "heavy_rain":
        condition = "heavy rain"
        rain_mm = LEVELS["rain_heavy"]
        wind = LEVELS["wind_breezy"]
        humidity = LEVELS["humidity_wet"]
    elif weather_level == "snow":
        condition = "snow"
        snowfall_cm = LEVELS["snow_heavy"]
        wind = LEVELS["wind_breezy"]
        humidity = LEVELS["humidity_wet"]
    else:
        raise ValueError(f"Unknown weather level: {weather_level}")

    # Wind chill / wet chill — drops feels_like below ambient
    feels_like = temp_c
    if wind > 10:
        feels_like -= 3
    if rain_mm > 5 or snowfall_cm > 0:
        feels_like -= 2

    return {
        "condition": condition,
        "conditions": condition,        # both keys read by ERA prompt
        "temperature": temp_c,
        "feels_like": feels_like,
        "wind_speed": wind,             # NOT wind_speed_kmh — producer uses bare wind_speed
        "humidity": humidity,
        "rain_mm": rain_mm,
        "precipitation_mm": rain_mm,    # ERA prompt falls back to this if rain_mm absent
        "snowfall_cm": snowfall_cm,     # NOT snowfall_mm — ERA prompt reads cm
        "is_day": is_day,
        "sensor_nearby": True,
        "source_sensor": False,         # synthetic for eval
    }


def _build_traffic(traffic_level: str) -> Dict[str, Any]:
    """Build a traffic dict matching the producer shape.

    ERA prompt reads: status, delay_minutes, incidents (length).
    Producer also includes: incident_count, message, route data.
    """
    if traffic_level == "free":
        return {
            "status": "free flow",
            "delay_minutes": LEVELS["traffic_free_delay_min"],
            "incident_count": LEVELS["traffic_free_incidents"],
            "incidents": [],
        }
    elif traffic_level == "congested":
        return {
            "status": "congested",
            "delay_minutes": LEVELS["traffic_cong_delay_min"],
            "incident_count": LEVELS["traffic_cong_incidents"],
            "incidents": [
                {
                    "type": "congestion",
                    "severity": "moderate",
                    "delay_minutes": LEVELS["traffic_cong_delay_min"],
                    "location": "city center",
                    "road_numbers": [],
                    "description": "Heavy traffic on main approach",
                }
            ],
        }
    else:
        raise ValueError(f"Unknown traffic level: {traffic_level}")


def _build_parking(parking_level: str) -> Dict[str, Any]:
    """Build a parking dict matching the producer shape.

    ERA prompt reads: total_available.
    Producer also: total_capacity, status, has_nearby_parking, parking_lots.
    """
    capacity = LEVELS["parking_capacity"]
    if parking_level == "available":
        available = LEVELS["parking_avail_count"]
        status = "available"
    elif parking_level == "scarce":
        available = LEVELS["parking_scarce_count"]
        status = "full"
    else:
        raise ValueError(f"Unknown parking level: {parking_level}")

    return {
        "total_available": available,
        "total_capacity": capacity,
        "status": status,
        "has_nearby_parking": True,
        "is_filtered_by_location": True,
        "parking_lots": [
            {"name": "Synthetic Lot 1", "available": available, "capacity": capacity, "distance_km": 0.2}
        ],
    }


def _build_air_quality(pm25_level: str) -> Dict[str, Any]:
    """Build an air_quality dict matching the producer shape.

    ERA prompt reads: pm25, pm10. Producer also extracts no2 if present.
    """
    if pm25_level == "good":
        return {
            "pm25": LEVELS["pm25_good"],
            "pm10": LEVELS["pm10_good"],
            "no2": LEVELS["no2_good"],
            "sensor_nearby": True,
            "quality": "good",
        }
    elif pm25_level == "poor":
        return {
            "pm25": LEVELS["pm25_poor"],
            "pm10": LEVELS["pm10_poor"],
            "no2": LEVELS["no2_poor"],
            "sensor_nearby": True,
            "quality": "unhealthy",
        }
    else:
        raise ValueError(f"Unknown PM2.5 level: {pm25_level}")


def _build_time_block(time_label: str) -> Dict[str, Any]:
    """Build top-level time-of-day fields.

    ERA prompt reads: current_time, day_of_week, is_weekend, is_dark.
    Producer (_add_time_of_day_context) also sets current_hour.

    We hold day_of_week constant (Wednesday) so the COMMUTE_PEAK archetype
    (07:00-09:30 or 16:30-19:00) is never triggered — neither noon nor
    22:00 falls in those windows. This keeps the archetype variation tied
    to weather/darkness, not weekday vs weekend.
    """
    if time_label == "noon":
        return {
            "current_time": "12:00",
            "current_hour": 12,
            "day_of_week": "Wednesday",
            "is_weekend": False,
            "is_dark": False,
        }
    elif time_label == "22:00":
        return {
            "current_time": "22:00",
            "current_hour": 22,
            "day_of_week": "Wednesday",
            "is_weekend": False,
            "is_dark": True,
        }
    else:
        raise ValueError(f"Unknown time label: {time_label}")


# ---------------------------------------------------------------------------
# Build all 8 conditions.
# ---------------------------------------------------------------------------

def build_condition(condition_id: str) -> Dict[str, Any]:
    """Build a complete proactive_context dict for the given condition ID.

    The output is a dict suitable for passing directly to
    era_agent.rerank(..., proactive_context=<this>).
    """
    if condition_id not in CONDITION_LEVELS:
        raise ValueError(f"Unknown condition ID: {condition_id}. Use C1..C8.")

    levels = CONDITION_LEVELS[condition_id]
    time_block = _build_time_block(levels["time"])
    is_day = not time_block["is_dark"]

    context: Dict[str, Any] = {}
    context["weather"] = _build_weather(levels["weather"], levels["temp_c"], is_day)
    context["traffic"] = _build_traffic(levels["traffic"])
    context["parking"] = _build_parking(levels["parking"])
    context["air_quality"] = _build_air_quality(levels["pm25"])
    context.update(time_block)

    # Tag the context with its ID so downstream code can verify
    context["_condition_id"] = condition_id
    return context


CONDITIONS: Dict[str, Dict[str, Any]] = {
    cid: build_condition(cid) for cid in CONDITION_LEVELS
}


# ---------------------------------------------------------------------------
# Numeric encoding for metric scripts.
# Maps each condition to scalar numeric features per dimension, used for
# Spearman ρ and the "expected direction" monotonicity table from EVAL_SPEC.
# ---------------------------------------------------------------------------
CONDITION_NUMERIC: Dict[str, Dict[str, float]] = {}
for cid, lv in CONDITION_LEVELS.items():
    weather_to_rain_mm = {
        "clear": 0.0,
        "light_rain": float(LEVELS["rain_light"]),
        "heavy_rain": float(LEVELS["rain_heavy"]),
        "snow": 0.0,        # snow uses snowfall_cm, not rain_mm
    }
    weather_to_snowfall_cm = {
        "clear": 0.0, "light_rain": 0.0, "heavy_rain": 0.0,
        "snow": float(LEVELS["snow_heavy"]),
    }
    pm25_value = LEVELS["pm25_good"] if lv["pm25"] == "good" else LEVELS["pm25_poor"]
    delay = LEVELS["traffic_free_delay_min"] if lv["traffic"] == "free" else LEVELS["traffic_cong_delay_min"]
    parking_n = LEVELS["parking_avail_count"] if lv["parking"] == "available" else LEVELS["parking_scarce_count"]

    CONDITION_NUMERIC[cid] = {
        "rain_mm": weather_to_rain_mm[lv["weather"]],
        "snowfall_cm": weather_to_snowfall_cm[lv["weather"]],
        "temp_c": float(lv["temp_c"]),
        "pm25": float(pm25_value),
        "traffic_delay_min": float(delay),
        "parking_available": float(parking_n),
        "is_dark": 1.0 if lv["time"] == "22:00" else 0.0,
    }


# ---------------------------------------------------------------------------
# Self-check & smoke print
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Verify every condition has the right shape. Run with: python -m evaluation.condition_grid"""
    required_top = {"weather", "traffic", "parking", "air_quality",
                    "current_time", "current_hour", "is_dark"}
    required_weather = {"condition", "temperature", "feels_like", "wind_speed",
                        "rain_mm", "snowfall_cm", "is_day"}
    required_traffic = {"status", "delay_minutes", "incidents"}
    required_parking = {"total_available", "total_capacity"}
    required_aq = {"pm25", "pm10"}

    for cid, ctx in CONDITIONS.items():
        missing = required_top - set(ctx.keys())
        assert not missing, f"{cid}: missing top-level keys {missing}"

        w_missing = required_weather - set(ctx["weather"].keys())
        assert not w_missing, f"{cid}.weather: missing {w_missing}"

        t_missing = required_traffic - set(ctx["traffic"].keys())
        assert not t_missing, f"{cid}.traffic: missing {t_missing}"

        p_missing = required_parking - set(ctx["parking"].keys())
        assert not p_missing, f"{cid}.parking: missing {p_missing}"

        aq_missing = required_aq - set(ctx["air_quality"].keys())
        assert not aq_missing, f"{cid}.air_quality: missing {aq_missing}"

    print("[OK] all 8 conditions have required keys")


if __name__ == "__main__":
    import json
    _self_check()
    print("\nCondition grid:\n" + "=" * 70)
    for cid, levels in CONDITION_LEVELS.items():
        print(f"  {cid}: {levels['weather']:11s} {levels['temp_c']:>3}°C  "
              f"{levels['time']:5s}  PM2.5={levels['pm25']:4s}  "
              f"traffic={levels['traffic']:9s}  parking={levels['parking']}")
    print("=" * 70)
    print("\nC5 example proactive_context (heavy rain, cool, noon, poor AQ, free traffic, scarce parking):")
    print(json.dumps(CONDITIONS["C5"], indent=2))
