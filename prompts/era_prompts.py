"""
Prompt templates for the Environmental Reasoning Agent (ERA).
ERA evaluates all route options against real-time environmental conditions
and produces structured per-option reasoning that the Synthesizer presents to users.
"""

import json
from typing import Dict, Any, Optional, List, Tuple


ERA_SYSTEM_PROMPT = """You are an Environmental Reasoning Agent (ERA) for a Magdeburg mobility assistant.

You receive route options (transit, walking, cycling, driving) and real-time environmental data.
Your job: reason about how current conditions affect each option and produce a structured evaluation.

You are not following a formula. You are making a judgment — the way a knowledgeable local would advise a friend. The guidance below helps you reason, but the final scores and recommendation are yours to decide based on the full picture.

## INPUT

You receive:
1. Route options with distances, durations, segments
2. Environmental context: weather, traffic, parking, air quality, time of day, day of week

## SCORING FOUNDATION

Your scoring is grounded in standard transport economics: McFadden's discrete choice theory (1974) and value-of-time research (Wardman 2014). Environmental conditions ADJUST a baseline that comes from generalized cost and distance-appropriate mode choice.

### Generalized cost (in weighted minutes, not raw clock time)

People weigh different time components differently. When comparing options, mentally compute:

  generalized_cost = IVT × 1.0
                   + walking_time × 1.5
                   + waiting_time × 2.0
                   + n_transfers × 5  (minutes equivalent per transfer)
                   + mode_startup_cost

Where:
- IVT (in-vehicle time): sitting on tram, in car, etc.
- walking_time: any outdoor walking the trip requires (legs to/from stops, full walk for walking option)
- waiting_time: time at stops or waiting for transfers
- mode_startup_cost: hidden friction at trip start. Driving ≈ 3-5 min (find car, navigate parking, walk from spot to door — especially when parking is scarce). Cycling ≈ 1-2 min (unlock, lock at destination). Transit waiting is captured separately. Walking ≈ 0.

A 3-minute drive that requires 5 minutes of parking friction is generalized cost 8, not 3. An 8-minute walk is generalized cost 8 — equivalent. This is why short trips often favor walking even when "driving is faster on the map."

### Distance-band mode appropriateness

Empirical modal share (Pucher & Buehler 2010, Forsyth & Krizek 2011) gives baselines BEFORE environmental adjustment:

  <500 m:        walking dominant. Driving rarely sensible (startup ≈ trip time).
                 Walking 8-9. Cycling 6-7. Transit 4-5. Driving 3-5.
  500-1500 m:    walking + cycling competitive.
                 Walking 7-8. Cycling 7-8. Transit 5-6. Driving 5-6.
  1500-5000 m:   cycling, transit, driving viable; walking less so.
                 Cycling 7-8. Transit 6-7. Driving 7. Walking 3-5.
  >5000 m:       transit + driving dominate.
                 Transit 6-8. Driving 7-8. Cycling 5-7. Walking 1-2.

These are baselines, not caps. Heavy rain, ice, darkness, or scarce parking shift the appropriate choice within and sometimes across bands. Apply environmental adjustments on top.

## REASONING PROCESS

Reason in three steps: classify the situation, analyze relevant dimensions, then score.

### STEP 0 — SITUATION CLASSIFICATION

Before analyzing dimensions, identify which situation archetype best describes this trip. Use the timestamp, day of week, trip distance, weather, and available modes to decide.

Archetypes:
- COMMUTE_PEAK: Weekday during typical rush hours (roughly 07:00–09:30 or 16:30–19:00). Door-to-door time, transit reliability, and traffic conditions tend to matter most.
- LEISURE: Weekend or weekday off-peak with no time pressure. Comfort, walkability, air quality, and enjoyment of the journey tend to matter more than raw speed.
- LATE_NIGHT: Roughly 22:00–06:00. Safety, transit availability, and walking exposure in darkness become the central concerns.
- BAD_WEATHER: Active rain, strong wind, snow, ice, or feels-like temperature well below freezing. Minimizing outdoor exposure and choosing sheltered options becomes the priority.
- STANDARD: None of the above clearly dominates. All dimensions start with roughly equal weight.

If multiple archetypes apply (e.g., late night AND bad weather), pick the one that most strongly shapes the decision and note the overlap.

Based on the archetype, decide which dimensions are primary (most influential for scoring), secondary (relevant but less decisive), and deprioritised (unlikely to differentiate options). This split guides how much each dimension affects the holistic score — a primary dimension should strongly influence the final score, while a deprioritised one should only matter if it reveals something extreme.

### PHASE 1 — DIMENSION ANALYSIS

For each of the six dimensions below, first decide whether it is salient — does it meaningfully differentiate between the available options given current conditions?

**Salience check**: If a dimension does not differentiate options (e.g., air quality is good and all options have similar exposure, or traffic is irrelevant because no driving option exists), mark it as not salient: set "salient" to false, write a brief reason, and move on. Do not force analysis where there is nothing to analyze. Non-salient dimensions get null scores in Phase 2.

If the dimension IS salient, reason through it:
- **Think**: What specific aspect of this dimension could make one option noticeably better or worse right now?
- **Observe**: List the relevant data values from the input. Quote actual numbers.
- **Assess**: Given what you observed, which options benefit and which suffer on this dimension?

**weather_exposure**
Consider precipitation, temperature, and wind together — they combine. How long is each option exposed outdoors? As a reference:
- Heavy or sustained rain with long exposure tends to be a strong deterrent
- Brief exposure (a minute or two) in light rain is usually tolerable
- Extreme temperatures matter more the longer you are outside
- Strong wind affects cycling more than walking, but both suffer in gusts

**safety**
Is it dark? Could surfaces be icy? Consider the duration and context — a short walk to a bus stop in the dark is very different from a 15-minute walk through unlit areas. Cycling in darkness or on icy roads carries real risk.

**traffic_parking**
What does traffic look like? Any incidents? Is parking available? Consider whether rush hour applies. Heavy delays or incidents may make driving worse than alternatives. Clear roads with available parking may make driving attractive for longer trips.

**air_quality**
What are the PM2.5, PM10, NO2 values? Reference ranges for PM2.5:
- Up to ~12 µg/m³ is generally considered good
- ~13–35 µg/m³ is moderate — worth mentioning but usually not a deterrent
- ~36–55 µg/m³ is elevated — may matter for prolonged outdoor exertion (use "elevated", not "poor")
- Above ~55 µg/m³ is unhealthy — outdoor exercise becomes a real concern
These are guidelines, not cutoffs. A PM2.5 of 34 with a 5-minute walk is different from 34 with a 40-minute cycle.

**transit_availability**
What time is it? Late at night (roughly 23:00–05:00), transit in Magdeburg is unreliable or not running — this matters a lot. Early morning and late evening may have reduced frequency. During regular hours, transit is dependable.

**practicality**
Apply the generalized cost framing from the SCORING FOUNDATION above. Do not just compare raw clock times. For each option, mentally weight: IVT × 1.0, walking × 1.5, waiting × 2.0, transfers × 5 min penalty, plus mode-startup cost (driving has 3-5 min hidden friction even on a short trip; cycling has 1-2 min). A 3-minute drive that requires parking has roughly the same generalized cost as a 7-8 minute walk — that is why walking often wins on short trips. Then locate the trip in the distance bands above to set a baseline before environmental adjustment.

### PHASE 2 — HOLISTIC SCORING

Bring your Step 0 classification and Phase 1 analysis together. For each option, assign:
1. Per-dimension scores (1-10) in "dimension_scores" — only for salient dimensions (null for non-salient)
2. A holistic "score" (1-10) — your overall judgment

The holistic score should reflect the archetype's dimension priorities. Primary dimensions should weigh heavily. Deprioritised dimensions should only pull the score if they reveal something extreme. This is not a formula — use your judgment on how much each dimension matters for this specific trip.

General tendencies (not rigid rules — override when the situation calls for it):
- In neutral conditions, shorter active options like walking and cycling tend to be preferable
- In rain or snow, options that minimize outdoor exposure tend to be better
- Late at night, transit may not be available — factor this heavily

Score interpretation:
- 9-10: Excellent choice given conditions
- 7-8: Good choice, minor drawbacks
- 5-6: Acceptable but with notable trade-offs
- 3-4: Poor choice given conditions
- 1-2: Strongly inadvisable given conditions

## WORKED EXAMPLES (for calibration)

### Example A — Short campus trip, good weather
Trip: 600 m on-campus. Walking 8 min. Cycling 4 min. Transit 15 min (with 11 min walking legs). Driving 3 min.
Conditions: clear, 18°C, noon, free traffic, ample parking, good AQ.
Distance band: 500-1500 m → walking & cycling competitive baseline.
Generalized cost: walking ≈ 8 (pure outdoor). Cycling ≈ 4 + 1 (unlock) ≈ 5. Transit ≈ 4 + 11×1.5 + 5×2 ≈ 30. Driving ≈ 3 + 4 (startup, parking) ≈ 7. So walking and driving are roughly equivalent in generalized cost — the conditions are pleasant — and active modes get a small bonus on a campus trip.
Expected scores: walking 8, cycling 8, driving 5, transit 4. Recommend walking.

### Example B — Long trip, heavy rain
Trip: 6 km. Walking 75 min. Cycling 22 min. Transit 25 min (with 11 min walking legs, 1 transfer). Driving 18 min.
Conditions: heavy rain 10 mm/h, 5°C, wind 18 km/h, congested traffic (+12 min), parking scarce.
Distance band: >5000 m → walking impractical, transit/driving dominate.
Heavy rain dominates: outdoor modes are heavily penalised. Driving is sheltered but congested adds delay; scarce parking adds friction. Transit walking legs are fully exposed. Cycling is fully exposed for 22 minutes.
Expected scores: driving 7, transit 4, cycling 2, walking 1. Recommend driving (the shelter outweighs the parking friction at this distance).

### Example C — Medium trip, late at night, scarce parking
Trip: 2 km. Walking 25 min. Cycling 8 min. Transit 12 min (with 6 min walking legs, no transfer). Driving 6 min.
Conditions: clear, 5°C, 22:00, congested, scarce parking, good AQ.
Distance band: 1500-5000 m → cycling/transit/driving viable.
Late-night reduces transit reliability. Darkness penalises walking safety despite good weather. Scarce parking and congestion hurt driving. Cycling has darkness risk but the trip is short.
Expected scores: driving 6, transit 5, cycling 6, walking 4. Recommend cycling (or driving — close call). Note: in heavy rain at the same time, driving would clearly win.

These examples illustrate the calibration. Driving is appropriate for long trips, bad weather, or when active-mode penalties dominate — but it loses on short trips with good conditions, where its hidden startup cost catches up to it.

## OUTPUT FORMAT

Return ONLY a JSON object (no extra text, no markdown):
{
    "situation": {
        "archetype": "COMMUTE_PEAK" | "LEISURE" | "LATE_NIGHT" | "BAD_WEATHER" | "STANDARD",
        "archetype_reasoning": "<one sentence explaining why this archetype applies>",
        "primary_dimensions": ["<dim1>", "<dim2>"],
        "secondary_dimensions": ["<dim3>", "<dim4>"],
        "deprioritised_dimensions": ["<dim5>", "<dim6>"]
    },
    "dimension_analysis": {
        "weather_exposure": {
            "salient": <bool>,
            "conditions": "<actual values — or brief reason if not salient>",
            "impact": "<your assessment, or null if not salient>",
            "penalized_modes": [],
            "severity": "none" | "low" | "moderate" | "high"
        },
        "safety": {
            "salient": <bool>,
            "conditions": "<actual values or reason>",
            "impact": "<your assessment or null>",
            "penalized_modes": [],
            "severity": "none" | "low" | "moderate" | "high"
        },
        "traffic_parking": {
            "salient": <bool>,
            "conditions": "<actual values or reason>",
            "impact": "<your assessment or null>",
            "penalized_modes": [],
            "severity": "none" | "low" | "moderate" | "high"
        },
        "air_quality": {
            "salient": <bool>,
            "conditions": "<actual values or reason>",
            "impact": "<your assessment or null>",
            "penalized_modes": [],
            "severity": "none" | "low" | "moderate" | "high"
        },
        "transit_availability": {
            "salient": <bool>,
            "conditions": "<actual values or reason>",
            "impact": "<your assessment or null>",
            "penalized_modes": [],
            "severity": "none" | "low" | "moderate" | "high"
        },
        "practicality": {
            "salient": <bool>,
            "conditions": "<actual values or reason>",
            "impact": "<your assessment or null>",
            "penalized_modes": [],
            "severity": "none" | "low" | "moderate" | "high"
        }
    },
    "options": [
        {
            "mode": "transit" | "walking" | "driving" | "cycling",
            "route_index": <int>,
            "dimension_scores": {
                "weather_exposure": <int 1-10 or null if not salient>,
                "safety": <int 1-10 or null>,
                "traffic_parking": <int 1-10 or null>,
                "air_quality": <int 1-10 or null>,
                "transit_availability": <int 1-10 or null>,
                "practicality": <int 1-10 or null>
            },
            "score": <int 1-10>,
            "outdoor_minutes": <float>,
            "weather_ok": <bool>,
            "safety_ok": <bool>,
            "reasoning": "<1-2 sentences: your synthesis explaining the holistic score>",
            "trade_off": "<1 sentence: what this option sacrifices vs the recommended one, or null if this IS recommended>"
        }
    ],
    "recommended_index": <int>,
    "recommendation_reason": "<1 sentence: why this is the best choice right now>",
    "condition_summary": "<1 sentence: the key environmental factors shaping your decision>",
    "warnings": ["<optional: safety or weather warnings worth flagging>"]
}

## FORMATTING CONVENTIONS

- route_index: 0, 1, 2... for transit candidates, -1 for walking, -2 for driving, -3 for cycling
- recommended_index = route_index of your top pick
- dimension_scores: 1-10 for salient dimensions, null for non-salient
- weather_ok: true unless weather meaningfully degrades this option
- safety_ok: true unless there are real safety concerns (darkness with long exposure, icy surfaces)
- outdoor_minutes: approximate time spent outdoors (walking legs + wait for transit; full duration for walk/cycle)
- Reasoning should be factual and grounded — mention actual temperatures, precipitation amounts, air quality values
- trade_off should compare against your recommended option (null if this IS the recommended option)
"""


def _format_candidate(idx: int, candidate: Dict[str, Any]) -> str:
    """Format one transit candidate for the ERA prompt."""
    lines = [f"### Transit Option {idx}"]

    segments = candidate.get("segments", [])
    total_walk_meters = 0
    total_walk_minutes = 0

    for seg in segments:
        if seg.get("type") == "walking":
            dist = seg.get("distance_meters", 0)
            time_min = seg.get("walk_time_minutes", 0)
            total_walk_meters += dist
            total_walk_minutes += time_min
            lines.append(f"- Walk {dist}m ({time_min} min) from {seg.get('from', '?')} to {seg.get('to', '?')}")
        elif seg.get("type") == "transit":
            line_name = seg.get("line", "?")
            board = seg.get("from") or seg.get("board_at", "?")
            alight = seg.get("to") or seg.get("alight_at", "?")
            stops = seg.get("stop_count", 0)
            dur = seg.get("duration_minutes", "?")
            lines.append(f"- Ride {line_name}: {board} -> {alight} ({stops} stops, {dur} min)")

    duration = candidate.get("estimated_duration_minutes", 0)
    transfers = candidate.get("transfers", 0)

    lines.append(f"- Total: ~{duration} min, {total_walk_meters}m walking ({total_walk_minutes} min outdoor), {transfers} transfer(s)")

    return "\n".join(lines)


def _format_walking_alternative(walking_alt: Dict[str, Any]) -> str:
    """Format the walking alternative for the ERA prompt."""
    if not walking_alt:
        return ""

    distance = walking_alt.get("distance", walking_alt.get("distance_meters", "?"))
    if isinstance(distance, (int, float)) and distance > 100:
        distance_str = f"{distance}m"
    else:
        distance_str = str(distance)

    duration_sec = walking_alt.get("duration_seconds", 0)
    if duration_sec:
        duration_min = max(1, round(duration_sec / 60))
        duration_str = f"~{duration_min} min"
    else:
        duration_str = walking_alt.get("duration", "?")

    return f"### Walking Option\n- Distance: {distance_str}, Duration: {duration_str}\n- Entire journey is outdoors"


def _format_driving_alternative(driving_alt: Dict[str, Any]) -> str:
    """Format the driving alternative for the ERA prompt."""
    if not driving_alt:
        return ""

    distance = driving_alt.get("distance", driving_alt.get("distance_meters", "?"))
    if isinstance(distance, (int, float)) and distance > 100:
        distance_str = f"{distance}m"
    else:
        distance_str = str(distance)

    duration_sec = driving_alt.get("duration_seconds", 0)
    if duration_sec:
        duration_min = max(1, round(duration_sec / 60))
        duration_str = f"~{duration_min} min"
    else:
        duration_str = driving_alt.get("duration", "?")

    traffic_status = driving_alt.get("traffic_status", "unknown")
    traffic_delay = driving_alt.get("traffic_delay_minutes", 0)

    parts = [
        f"### Driving Option",
        f"- Distance: {distance_str}, Duration: {duration_str}",
        f"- Traffic: {traffic_status}" + (f" (+{traffic_delay} min delay)" if traffic_delay else ""),
        f"- Minimal outdoor exposure (walk to/from car)",
    ]
    return "\n".join(parts)


def _format_cycling_alternative(cycling_alt: Dict[str, Any]) -> str:
    """Format the cycling alternative for the ERA prompt."""
    if not cycling_alt:
        return ""

    distance = cycling_alt.get("distance", cycling_alt.get("distance_meters", "?"))
    if isinstance(distance, (int, float)) and distance > 100:
        distance_str = f"{distance}m"
    else:
        distance_str = str(distance)

    duration_sec = cycling_alt.get("duration_seconds", 0)
    if duration_sec:
        duration_min = max(1, round(duration_sec / 60))
        duration_str = f"~{duration_min} min"
    else:
        duration_str = cycling_alt.get("duration", "?")

    return f"### Cycling Option\n- Distance: {distance_str}, Duration: {duration_str}\n- Entire journey is outdoors"


def _format_environmental_context(proactive_context: Optional[Dict[str, Any]]) -> str:
    """Format environmental context for the ERA prompt."""
    if not proactive_context:
        return "No environmental data available."

    parts = []

    # Weather
    weather = proactive_context.get("weather")
    if weather:
        condition = weather.get("condition", weather.get("conditions", "unknown"))
        temp = weather.get("temperature")
        feels_like = weather.get("feels_like")
        wind = weather.get("wind_speed", weather.get("wind_kmh"))
        rain_mm = weather.get("rain_mm", weather.get("precipitation_mm", 0))
        snowfall = weather.get("snowfall_cm", 0)
        is_day = weather.get("is_day")

        w_parts = [f"Condition: {condition}"]
        if temp is not None:
            w_parts.append(f"Temp: {temp}°C")
        if feels_like is not None:
            w_parts.append(f"Feels like: {feels_like}°C")
        if wind is not None:
            w_parts.append(f"Wind: {wind} km/h")
        if rain_mm and float(rain_mm) > 0:
            w_parts.append(f"Rain: {rain_mm} mm")
        else:
            w_parts.append("Rain: 0 mm (dry)")
        if snowfall and float(snowfall) > 0:
            w_parts.append(f"Snowfall: {snowfall} cm")
        if is_day is not None:
            w_parts.append(f"Daylight: {'yes' if is_day else 'no'}")

        parts.append("Weather: " + ", ".join(w_parts))

    # Time of day
    current_time = proactive_context.get("current_time")
    is_dark = proactive_context.get("is_dark")

    day_of_week = proactive_context.get("day_of_week")
    is_weekend = proactive_context.get("is_weekend")

    if current_time or is_dark is not None:
        time_parts = []
        if current_time:
            time_parts.append(f"Time: {current_time}")
        if day_of_week:
            time_parts.append(f"Day: {day_of_week}")
        if is_weekend is not None:
            time_parts.append(f"Weekend: {'yes' if is_weekend else 'no'}")
        if is_dark is not None:
            time_parts.append(f"Dark: {'yes' if is_dark else 'no'}")
        parts.append("Time: " + ", ".join(time_parts))

    # Traffic
    traffic = proactive_context.get("traffic")
    if traffic:
        status = traffic.get("status", "unknown")
        delay = traffic.get("delay_minutes", 0)
        incidents = traffic.get("incidents", [])

        t_parts = [f"Status: {status}"]
        if delay:
            t_parts.append(f"Delay: {delay} min")
        if incidents:
            t_parts.append(f"Incidents: {len(incidents)}")

        parts.append("Traffic: " + ", ".join(t_parts))

    # Parking
    parking = proactive_context.get("parking")
    if parking:
        available = parking.get("total_available", parking.get("available"))
        if available is not None:
            parts.append(f"Parking: {available} spots available")

    # Air quality
    air_quality = proactive_context.get("air_quality")
    if air_quality:
        pm25 = air_quality.get("pm25")
        pm10 = air_quality.get("pm10")
        aq_parts = []
        if pm25 is not None:
            aq_parts.append(f"PM2.5: {pm25}")
        if pm10 is not None:
            aq_parts.append(f"PM10: {pm10}")
        if aq_parts:
            parts.append("Air quality: " + ", ".join(aq_parts))

    return "\n".join(parts) if parts else "No environmental data available."


def build_era_prompt(
    candidates: List[Dict[str, Any]],
    walking_alternative: Optional[Dict[str, Any]] = None,
    driving_alternative: Optional[Dict[str, Any]] = None,
    cycling_alternative: Optional[Dict[str, Any]] = None,
    proactive_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Build the message list for the ERA LLM call."""

    # Build route options section
    route_parts = ["## ROUTE OPTIONS\n"]

    for idx, candidate in enumerate(candidates):
        route_parts.append(_format_candidate(idx, candidate))
        route_parts.append("")

    walk_text = _format_walking_alternative(walking_alternative)
    if walk_text:
        route_parts.append(walk_text)
        route_parts.append("")

    drive_text = _format_driving_alternative(driving_alternative)
    if drive_text:
        route_parts.append(drive_text)
        route_parts.append("")

    cycle_text = _format_cycling_alternative(cycling_alternative)
    if cycle_text:
        route_parts.append(cycle_text)
        route_parts.append("")

    # Build environmental context section
    env_text = _format_environmental_context(proactive_context)
    route_parts.append("## CURRENT CONDITIONS\n")
    route_parts.append(env_text)

    user_message = "\n".join(route_parts)

    return [
        {"role": "system", "content": ERA_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


VALID_DIMENSIONS = frozenset([
    "weather_exposure", "safety", "traffic_parking",
    "air_quality", "transit_availability", "practicality",
])

VALID_SEVERITIES = frozenset(["none", "low", "moderate", "high"])

VALID_ARCHETYPES = frozenset([
    "COMMUTE_PEAK", "LEISURE", "LATE_NIGHT", "BAD_WEATHER", "STANDARD",
])


def validate_era_output(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate the parsed ERA JSON output has required fields."""
    if not isinstance(data, dict):
        return False, "Output must be a JSON object"

    # --- Step 0: situation ---
    situation = data.get("situation")
    if not isinstance(situation, dict):
        return False, "Missing or invalid 'situation'"
    archetype = situation.get("archetype")
    if archetype not in VALID_ARCHETYPES:
        return False, f"Invalid archetype: {archetype}"
    if "archetype_reasoning" not in situation:
        return False, "situation missing 'archetype_reasoning'"
    for key in ("primary_dimensions", "secondary_dimensions", "deprioritised_dimensions"):
        if not isinstance(situation.get(key), list):
            return False, f"situation missing or invalid '{key}'"
    # All dimensions must appear in exactly one priority bucket
    all_assigned = set(situation["primary_dimensions"] + situation["secondary_dimensions"] + situation["deprioritised_dimensions"])
    missing_dims = VALID_DIMENSIONS - all_assigned
    if missing_dims:
        # Soft: don't fail, just warn — LLM might omit a deprioritised one
        pass

    # --- Phase 1: dimension_analysis ---
    dim_analysis = data.get("dimension_analysis")
    if not isinstance(dim_analysis, dict):
        return False, "Missing or invalid 'dimension_analysis'"

    for dim in VALID_DIMENSIONS:
        if dim not in dim_analysis:
            return False, f"dimension_analysis missing '{dim}'"
        entry = dim_analysis[dim]
        if not isinstance(entry, dict):
            return False, f"dimension_analysis['{dim}'] must be an object"
        if "salient" not in entry:
            return False, f"dimension_analysis['{dim}'] missing 'salient'"
        if "conditions" not in entry:
            return False, f"dimension_analysis['{dim}'] missing 'conditions'"
        # impact can be null for non-salient dimensions
        severity = entry.get("severity", "none")
        if severity not in VALID_SEVERITIES:
            return False, f"dimension_analysis['{dim}'] invalid severity: {severity}"

    # --- Phase 2: options ---
    if "options" not in data:
        return False, "Missing 'options'"

    if not isinstance(data["options"], list):
        return False, "'options' must be a list"

    if len(data["options"]) == 0:
        return False, "'options' cannot be empty"

    for i, opt in enumerate(data["options"]):
        if "mode" not in opt:
            return False, f"options[{i}] missing 'mode'"
        if opt["mode"] not in ("transit", "walking", "driving", "cycling"):
            return False, f"options[{i}] invalid mode: {opt['mode']}"
        if "route_index" not in opt:
            return False, f"options[{i}] missing 'route_index'"
        if "score" not in opt:
            return False, f"options[{i}] missing 'score'"
        score = opt["score"]
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            return False, f"options[{i}] score must be 1-10, got {score}"
        if "reasoning" not in opt:
            return False, f"options[{i}] missing 'reasoning'"

        # Validate dimension_scores (nullable for non-salient dimensions)
        dim_scores = opt.get("dimension_scores")
        if not isinstance(dim_scores, dict):
            return False, f"options[{i}] missing or invalid 'dimension_scores'"
        for dim in VALID_DIMENSIONS:
            if dim not in dim_scores:
                return False, f"options[{i}].dimension_scores missing '{dim}'"
            ds = dim_scores[dim]
            if ds is not None:
                if not isinstance(ds, (int, float)) or ds < 1 or ds > 10:
                    return False, f"options[{i}].dimension_scores['{dim}'] must be 1-10 or null, got {ds}"

    if "recommended_index" not in data:
        return False, "Missing 'recommended_index'"

    if "recommendation_reason" not in data:
        return False, "Missing 'recommendation_reason'"

    if "condition_summary" not in data:
        return False, "Missing 'condition_summary'"

    return True, ""
