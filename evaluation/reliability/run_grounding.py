"""
Section 5.4 Experiment 2 — Grounding & Faithfulness.

For every successful trial in evaluation/era_sensitivity/trials.jsonl:
  1. Pick the recommended option (route_index == era_output.recommended_index;
     fallback: highest-scoring option).
  2. Build a plain-text condition_description from condition_grid.CONDITION_LEVELS
     + LEVELS for the trial's condition_id.
  3. Send a 2-question judge prompt to gpt-5.4 (OpenAI, key from GPT_API env).
  4. Parse Q1 (grounding) and Q2 (faithfulness violation). Q2=YES means the
     justification invented a condition not present.

A separate judge model is used (NOT Qwen) per EVAL_SPEC §5.4 to avoid
self-evaluation bias.

Output:
    evaluation/reliability/grounding/grounding_trials.jsonl
    evaluation/reliability/grounding/grounding_summary.json

Usage:
    python evaluation/reliability/run_grounding.py
    python evaluation/reliability/run_grounding.py --smoke   # 1 call only
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Load .env so GPT_API + others are available (used as fallback if the
# constant below is left as the placeholder).
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_DIR / ".env")

# ===========================================================================
# >>>  EDIT THIS LINE  <<<
#   Either paste your OpenAI key directly here (replacing the placeholder)
#   OR leave it as-is and put GPT_API=<your_key> in the .env file.
JUDGE_API_KEY = "REPLACE_WITH_YOUR_OPENAI_KEY"
# ===========================================================================

OUT_DIR = PROJECT_DIR / "evaluation" / "reliability" / "grounding"
TRIALS_OUT = OUT_DIR / "grounding_trials.jsonl"
SUMMARY_OUT = OUT_DIR / "grounding_summary.json"

SENS_TRIALS = PROJECT_DIR / "evaluation" / "era_sensitivity" / "trials.jsonl"

JUDGE_MODEL = "gpt-5.4"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 50
PACE_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Condition description builder — uses condition_grid as single source of truth
# ---------------------------------------------------------------------------

def build_condition_description(cid: str) -> str:
    """Plain-text summary of an injected condition for the judge prompt.

    Reads the actual proactive_context dict from CONDITIONS[cid] so every
    field ERA sees (wind, feels_like, snowfall, etc.) is surfaced to the
    judge. Listing only a subset causes false-positive hallucinations
    when ERA correctly references a field the judge wasn't told about.
    """
    from evaluation.condition_grid import CONDITIONS

    ctx = CONDITIONS[cid]
    w = ctx["weather"]
    tr = ctx.get("traffic") or {}
    pk = ctx.get("parking") or {}
    aq = ctx.get("air_quality") or {}

    # Weather line — combine condition + precipitation amount
    cond = w.get("condition", "?")
    rain = w.get("rain_mm", 0) or 0
    snow = w.get("snowfall_cm", 0) or 0
    if rain > 0:
        weather_str = f"{cond} (~{rain} mm/h)"
    elif snow > 0:
        weather_str = f"{cond} (~{snow} cm/h)"
    else:
        weather_str = f"{cond}, no precipitation"

    temp = w.get("temperature")
    feels = w.get("feels_like")
    wind = w.get("wind_speed")
    humidity = w.get("humidity")

    time_str = ctx.get("current_time", "?")
    is_dark = ctx.get("is_dark", False)
    daylight_str = "dark outside" if is_dark else "daylight"
    day_of_week = ctx.get("day_of_week", "")

    pm25 = aq.get("pm25")
    pm10 = aq.get("pm10")
    no2 = aq.get("no2")
    aq_label = aq.get("quality", "?")

    delay = tr.get("delay_minutes", 0) or 0
    incident_count = len(tr.get("incidents") or [])
    if delay > 0 or incident_count > 0:
        traffic_str = f"{tr.get('status','?')} (~{delay} min delay, {incident_count} incident(s))"
    else:
        traffic_str = f"{tr.get('status','?')}, no incidents"

    avail = pk.get("total_available")
    capacity = pk.get("total_capacity")
    parking_str = f"{pk.get('status','?')} ({avail}/{capacity} spaces free)"

    return (
        f"Weather: {weather_str}\n"
        f"Temperature: {temp}°C (feels like {feels}°C)\n"
        f"Wind: {wind} km/h\n"
        f"Humidity: {humidity}%\n"
        f"Time of day: {time_str} on {day_of_week} ({daylight_str})\n"
        f"Air quality: {aq_label} (PM2.5 = {pm25} μg/m³, PM10 = {pm10}, NO2 = {no2})\n"
        f"Traffic: {traffic_str}\n"
        f"Parking: {parking_str}"
    )


# ---------------------------------------------------------------------------
# Trial loading — same dedupe logic as evaluation/ablation/run_ablation.py
# ---------------------------------------------------------------------------

def load_unique_trials() -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(SENS_TRIALS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("error"):
                continue
            pair = (t["query_id"], t["condition_id"])
            existing = by_pair.get(pair)
            if existing is None or t.get("timestamp", "") > existing.get("timestamp", ""):
                by_pair[pair] = t
    return by_pair


def pick_recommended_option(era_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find the option whose route_index matches recommended_index. Fall back
    to highest-scoring option."""
    rec_idx = era_output.get("recommended_index")
    options = era_output.get("options") or []
    if rec_idx is not None:
        for opt in options:
            if opt.get("route_index") == rec_idx:
                return opt
    if options:
        return max(options, key=lambda o: o.get("score", -1))
    return None


# ---------------------------------------------------------------------------
# Judge prompt + parsing
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are evaluating an AI mobility assistant's justification text.

INJECTED CONDITIONS:
{condition_description}

JUSTIFICATION TEXT:
{reasoning_text}

Answer exactly two questions with only "YES" or "NO" on separate lines:

Q1 GROUNDING: Does the justification text explicitly mention or clearly reference at least one of the injected conditions listed above?
Answer YES if the text references weather, temperature, time of day, air quality, traffic, or parking conditions that match what was injected.
Answer NO if the justification contains only generic statements with no reference to the specific conditions.

Q2 FAITHFULNESS: Does the justification text mention any environmental conditions that were NOT present in the injected conditions above?
Answer YES if the text invents or hallucinates conditions (e.g. mentions rain when the condition was clear, mentions congestion when traffic was free).
Answer NO if the text only references conditions that were actually injected.

Respond with exactly two lines:
Q1: YES or NO
Q2: YES or NO"""


def build_prompt(condition_description: str, reasoning_text: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        condition_description=condition_description,
        reasoning_text=reasoning_text,
    )


def parse_judge_response(text: str) -> Tuple[Optional[bool], Optional[bool]]:
    """Return (q1_yes, q2_yes). Either may be None on parse failure."""
    q1 = None
    q2 = None
    for raw_line in text.splitlines():
        line = raw_line.strip().upper()
        if line.startswith("Q1"):
            if "YES" in line:
                q1 = True
            elif "NO" in line:
                q1 = False
        elif line.startswith("Q2"):
            if "YES" in line:
                q2 = True
            elif "NO" in line:
                q2 = False
    return q1, q2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="§5.4 grounding + faithfulness judge")
    parser.add_argument("--smoke", action="store_true",
                        help="Send 1 judge call only (verifies model + key) and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap total trials this run")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pick API key: placeholder constant if the user pasted one, else env var
    if JUDGE_API_KEY and JUDGE_API_KEY != "REPLACE_WITH_YOUR_OPENAI_KEY":
        api_key = JUDGE_API_KEY
        key_source = "JUDGE_API_KEY constant"
    else:
        api_key = os.getenv("GPT_API")
        key_source = "GPT_API env var"
    if not api_key:
        sys.exit(
            "FATAL: no API key found. Either:\n"
            "  (a) edit JUDGE_API_KEY at the top of run_grounding.py, or\n"
            "  (b) set GPT_API=<your_key> in the .env file."
        )
    print(f"  Using key from: {key_source}")

    print("=" * 70)
    print(f"§5.4 GROUNDING & FAITHFULNESS — judge: {JUDGE_MODEL}")
    print("=" * 70)

    from openai import OpenAI
    # Explicitly pin to api.openai.com — .env has OPENAI_BASE_URL pointing at
    # Academic Cloud (for Qwen) and the SDK would otherwise pick that up.
    judge_client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    print("[Load] reading sensitivity trials...")
    trials_by_pair = load_unique_trials()
    print(f"  {len(trials_by_pair)} unique successful trials")

    # Wipe prior grounding outputs for clean run
    if TRIALS_OUT.exists():
        backup = TRIALS_OUT.with_suffix(f".jsonl.bak.{int(time.time())}")
        TRIALS_OUT.rename(backup)
        print(f"  Existing trials backed up -> {backup.name}")

    # --- SMOKE TEST: 1 call to verify model + key ---
    smoke_pair = next(iter(sorted(trials_by_pair.keys())))
    smoke_trial = trials_by_pair[smoke_pair]
    smoke_opt = pick_recommended_option(smoke_trial.get("era_output") or {})
    if smoke_opt is None:
        sys.exit("FATAL: smoke trial has no options to extract reasoning from.")
    smoke_cond = build_condition_description(smoke_pair[1])
    smoke_prompt = build_prompt(smoke_cond, smoke_opt.get("reasoning", ""))

    print(f"\n[Smoke] testing {JUDGE_MODEL} with 1 call ({smoke_pair[0]} x {smoke_pair[1]})...")
    smoke_start = time.time()
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            max_completion_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": smoke_prompt}],
        )
        smoke_text = resp.choices[0].message.content.strip()
        smoke_q1, smoke_q2 = parse_judge_response(smoke_text)
        print(f"  smoke ok ({time.time() - smoke_start:.1f}s)")
        print(f"  raw response: {smoke_text!r}")
        print(f"  parsed: Q1={smoke_q1}  Q2={smoke_q2}")
    except Exception as e:
        sys.exit(f"FATAL: smoke test failed: {type(e).__name__}: {e}")

    if args.smoke:
        print("\n--smoke flag set, exiting after smoke test.")
        return

    # --- Full run ---
    pairs = sorted(trials_by_pair.keys())
    if args.limit:
        pairs = pairs[: args.limit]
    total = len(pairs)
    print(f"\n[Run] judging {total} trials, pace={PACE_SECONDS}s")

    n_q1_yes = 0
    n_q2_yes = 0
    n_judge_errors = 0
    by_condition: Dict[str, Dict[str, int]] = defaultdict(lambda: {"q1_yes": 0, "q2_no": 0, "n": 0, "errors": 0})
    by_mode: Dict[str, Dict[str, int]] = defaultdict(lambda: {"q1_yes": 0, "n": 0})
    faithfulness_violations: List[Dict[str, Any]] = []
    judge_errors: List[Dict[str, Any]] = []

    run_start = time.time()
    for idx, (qid, cid) in enumerate(pairs, 1):
        trial = trials_by_pair[(qid, cid)]
        era = trial.get("era_output") or {}
        opt = pick_recommended_option(era)
        if opt is None:
            n_judge_errors += 1
            judge_errors.append({
                "query_id": qid, "condition_id": cid,
                "error": "no options to extract reasoning from",
            })
            continue

        recommended_mode = opt.get("mode") or "?"
        reasoning_text = opt.get("reasoning", "")
        condition_description = build_condition_description(cid)
        prompt = build_prompt(condition_description, reasoning_text)

        record_base = {
            "query_id": qid,
            "condition_id": cid,
            "recommended_mode": recommended_mode,
            "reasoning_text": reasoning_text,
            "condition_description": condition_description,
            "judge_model": JUDGE_MODEL,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        try:
            resp = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                max_completion_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            q1, q2 = parse_judge_response(raw)

            if q1 is None or q2 is None:
                n_judge_errors += 1
                judge_errors.append({
                    "query_id": qid, "condition_id": cid,
                    "raw_response": raw, "parsed": {"q1": q1, "q2": q2},
                    "error": "parse failure",
                })
                rec = {**record_base,
                       "judge_error": True,
                       "judge_raw_response": raw,
                       "q1_grounding": None,
                       "q2_faithfulness_violation": None}
                with open(TRIALS_OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                by_condition[cid]["n"] += 1
                by_condition[cid]["errors"] += 1
                if idx % 10 == 0:
                    print(f"  Processed {idx}/{total} trials... (errors so far: {n_judge_errors})")
                if idx < total and PACE_SECONDS > 0:
                    time.sleep(PACE_SECONDS)
                continue

            # Successful judge call
            if q1:
                n_q1_yes += 1
                by_condition[cid]["q1_yes"] += 1
                by_mode[recommended_mode]["q1_yes"] += 1
            if q2:
                n_q2_yes += 1
                faithfulness_violations.append({
                    "query_id": qid,
                    "condition_id": cid,
                    "recommended_mode": recommended_mode,
                    "reasoning_text": reasoning_text,
                    "condition_description": condition_description,
                })
            else:
                by_condition[cid]["q2_no"] += 1
            by_condition[cid]["n"] += 1
            by_mode[recommended_mode]["n"] += 1

            rec = {**record_base,
                   "judge_error": False,
                   "judge_raw_response": raw,
                   "q1_grounding": q1,
                   "q2_faithfulness_violation": q2}
            with open(TRIALS_OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        except Exception as e:
            n_judge_errors += 1
            judge_errors.append({
                "query_id": qid, "condition_id": cid,
                "error": f"{type(e).__name__}: {e}",
            })
            rec = {**record_base,
                   "judge_error": True,
                   "judge_raw_response": None,
                   "q1_grounding": None,
                   "q2_faithfulness_violation": None,
                   "exception": f"{type(e).__name__}: {e}"}
            with open(TRIALS_OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            by_condition[cid]["n"] += 1
            by_condition[cid]["errors"] += 1

        if idx % 10 == 0:
            elapsed = time.time() - run_start
            print(f"  Processed {idx}/{total} trials... ({elapsed/60:.1f} min, errors: {n_judge_errors})")

        if idx < total and PACE_SECONDS > 0:
            time.sleep(PACE_SECONDS)

    wall = time.time() - run_start

    # --- Summary ---
    n_evaluated = total - n_judge_errors
    grounding_rate = round(n_q1_yes / n_evaluated, 4) if n_evaluated else 0.0
    faithfulness_rate = round(1.0 - (n_q2_yes / n_evaluated), 4) if n_evaluated else 0.0

    grounding_by_condition = {}
    for cid, c in by_condition.items():
        n_judged = c["n"] - c["errors"]
        grounding_by_condition[cid] = {
            "grounding_rate": round(c["q1_yes"] / n_judged, 4) if n_judged else None,
            "faithfulness_rate": round(c["q2_no"] / n_judged, 4) if n_judged else None,
            "n": c["n"],
            "n_judged": n_judged,
            "errors": c["errors"],
        }

    grounding_by_mode = {
        m: {
            "grounding_rate": round(c["q1_yes"] / c["n"], 4) if c["n"] else None,
            "n": c["n"],
        }
        for m, c in by_mode.items()
    }

    summary = {
        "n_trials_total": total,
        "n_judge_errors": n_judge_errors,
        "n_evaluated": n_evaluated,
        "grounding_rate": grounding_rate,
        "faithfulness_rate": faithfulness_rate,
        "grounding_by_condition": dict(sorted(grounding_by_condition.items())),
        "grounding_by_mode": dict(grounding_by_mode),
        "faithfulness_violations": faithfulness_violations,
        "judge_errors": judge_errors,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": JUDGE_TEMPERATURE,
        "wall_clock_seconds": round(wall, 1),
        "trials_path": str(TRIALS_OUT.relative_to(PROJECT_DIR)),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Done] {n_evaluated} judged, {n_judge_errors} errors, {wall/60:.1f} min wall clock")
    print(f"  Grounding rate:   {grounding_rate*100:.1f}% ({n_q1_yes}/{n_evaluated})")
    print(f"  Faithfulness:     {faithfulness_rate*100:.1f}% ({n_evaluated - n_q2_yes}/{n_evaluated} clean)")
    print(f"  Faithfulness violations: {len(faithfulness_violations)}")
    print(f"\n  trials  -> {TRIALS_OUT.relative_to(PROJECT_DIR)}")
    print(f"  summary -> {SUMMARY_OUT.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
