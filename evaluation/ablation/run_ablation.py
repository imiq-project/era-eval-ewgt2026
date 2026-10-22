"""
Section 5.3 ablation harness — runs three arms over the existing 200 ERA
sensitivity trials and writes arm_b_scores.jsonl, ablation_results.csv, and
ablation_summary.json.

Arm A — No ERA: rank by travel time.
Arm B — Rule-based ERA: deterministic linear penalties per (condition_flag, mode).
Arm C — LLM ERA: extracted from trials.jsonl.

Usage:
    python -m evaluation.ablation.run_ablation
    python evaluation/ablation/run_ablation.py
"""

import sys
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple

# Make the project root importable so condition_grid resolves regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluation.condition_grid import CONDITION_LEVELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = PROJECT_ROOT / "evaluation"
SENS_DIR = EVAL_DIR / "era_sensitivity"
OUT_DIR = EVAL_DIR / "ablation"
OUT_DIR.mkdir(exist_ok=True)

TRIALS_PATH = SENS_DIR / "trials.jsonl"
CACHE_PATH = SENS_DIR / "candidates_cache.json"
QUERIES_PATH = EVAL_DIR / "era_sensitivity_queries.json"

ARM_B_OUT = OUT_DIR / "arm_b_scores.jsonl"
RESULTS_CSV = OUT_DIR / "ablation_results.csv"
SUMMARY_JSON = OUT_DIR / "ablation_summary.json"

# ---------------------------------------------------------------------------
# Configuration knobs (locked for reproducibility — change here if needed)
# ---------------------------------------------------------------------------
BASE_SCORE = 8
SCORE_MIN, SCORE_MAX = 1, 10
TIEBREAKER = "travel_time_ascending"  # alternative: "alphabetical"
TEMPERATURE_POLICY = "most_specific"  # "most_specific" | "stacked"
SIMPLE_CONDITIONS = ["C1", "C2", "C4"]
COMPOUND_CONDITIONS = ["C3", "C5", "C6", "C7", "C8"]

# Rule-based ERA penalty table (per the EVAL_SPEC §5.3 / paper appendix)
PENALTIES: Dict[Tuple[str, str], float] = {
    # weather × mode
    ("light_rain", "walking"): -1.0,
    ("light_rain", "cycling"): -1.0,
    ("heavy_rain", "walking"): -3.0,
    ("heavy_rain", "cycling"): -3.0,
    ("snow", "walking"): -4.0,
    ("snow", "cycling"): -4.0,
    ("heavy_rain", "driving"): -0.5,
    ("snow", "driving"): -0.5,
    # temperature × mode
    ("temp_below_0", "walking"): -1.0,
    ("temp_below_0", "cycling"): -1.0,
    ("temp_below_5", "walking"): -0.5,
    ("temp_below_5", "cycling"): -0.5,
    # traffic × mode
    ("congested", "driving"): -2.0,
    ("congested", "transit"): -0.5,
    # parking × mode
    ("scarce_parking", "driving"): -2.0,
    # time × mode
    ("late_night", "walking"): -1.0,
    ("late_night", "cycling"): -1.0,
    # air quality × mode
    ("poor_aq", "walking"): -1.0,
    ("poor_aq", "cycling"): -1.0,
}

ALL_MODES = ("walking", "cycling", "driving", "transit")

# route_index ↔ mode mapping per ERAOptionEval convention (models.py)
ROUTE_INDEX_TO_MODE = {-1: "walking", -2: "driving", -3: "cycling"}
# transit uses non-negative indices 0..K-1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_trials() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load trials.jsonl, deduplicate by (qid, cid). Prefer error=None records;
    among ties keep the latest by timestamp.
    """
    trials_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    raw_count = 0
    error_records = 0
    with open(TRIALS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            raw_count += 1
            if t.get("error"):
                error_records += 1
                continue
            pair = (t["query_id"], t["condition_id"])
            existing = trials_by_pair.get(pair)
            if existing is None or t.get("timestamp", "") > existing.get("timestamp", ""):
                trials_by_pair[pair] = t
    print(f"  trials: raw={raw_count}, errors_dropped={error_records}, "
          f"unique_pairs={len(trials_by_pair)}")
    return trials_by_pair


def load_candidates() -> Dict[str, Dict[str, Any]]:
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_query_metadata() -> Dict[str, Dict[str, Any]]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["queries"]}


# ---------------------------------------------------------------------------
# Mode duration extraction (Arm A + tiebreaker for Arm B)
# ---------------------------------------------------------------------------

def get_mode_durations_seconds(qid: str, cache: Dict[str, Any]) -> Dict[str, float]:
    """Return {mode: duration_seconds} for the modes available in cache for qid.

    Walking/cycling/driving alternatives carry duration_seconds directly.
    Transit candidates carry estimated_duration_minutes — convert to seconds.
    If multiple transit candidates exist, take the one with the lowest duration
    (mirrors how Arm A would rank within transit, and how the LLM's lowest-cost
    candidate is typically the route_index=0 winner).
    """
    entry = cache.get(qid, {})
    durations: Dict[str, float] = {}

    for mode_key, alt_key in (
        ("walking", "walking_alternative"),
        ("cycling", "cycling_alternative"),
        ("driving", "driving_alternative"),
    ):
        alt = entry.get(alt_key)
        if alt and alt.get("duration_seconds") is not None:
            durations[mode_key] = float(alt["duration_seconds"])

    cands = entry.get("candidates", [])
    transit_durations = [
        c["estimated_duration_minutes"] * 60.0
        for c in cands
        if c.get("estimated_duration_minutes") is not None
    ]
    if transit_durations:
        durations["transit"] = min(transit_durations)

    return durations


# ---------------------------------------------------------------------------
# Arm A — No ERA (travel-time ranking)
# ---------------------------------------------------------------------------

def arm_a_top1(qid: str, cache: Dict[str, Any]) -> Optional[str]:
    """Lowest duration_seconds wins. Condition-independent."""
    durations = get_mode_durations_seconds(qid, cache)
    if not durations:
        return None
    return min(durations.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Arm B — Rule-based ERA
# ---------------------------------------------------------------------------

def condition_to_flags(cid: str) -> List[str]:
    """Translate CONDITION_LEVELS[cid] into the active penalty-key flags.

    Temperature policy: 'most_specific' applies only the colder threshold
    (temp_below_0 supersedes temp_below_5). 'stacked' applies both.
    """
    levels = CONDITION_LEVELS[cid]
    flags: List[str] = []

    weather = levels["weather"]
    if weather in ("light_rain", "heavy_rain", "snow"):
        flags.append(weather)

    temp = levels["temp_c"]
    if TEMPERATURE_POLICY == "most_specific":
        if temp <= 0:
            flags.append("temp_below_0")
        elif temp <= 5:
            flags.append("temp_below_5")
    elif TEMPERATURE_POLICY == "stacked":
        if temp <= 0:
            flags.append("temp_below_0")
        if temp <= 5:
            flags.append("temp_below_5")
    else:
        raise ValueError(f"Unknown TEMPERATURE_POLICY: {TEMPERATURE_POLICY}")

    if levels["traffic"] == "congested":
        flags.append("congested")
    if levels["parking"] == "scarce":
        flags.append("scarce_parking")
    if levels["time"] == "22:00":
        flags.append("late_night")
    if levels["pm25"] == "poor":
        flags.append("poor_aq")

    return flags


def score_mode(mode: str, flags: List[str]) -> Tuple[float, List[str]]:
    """Apply the penalty table to a single mode for a given flag set.

    Returns (final_score_clamped, list_of_applied_keys).
    The applied list mirrors the prompt schema: '<flag>_<mode>'.
    """
    score = float(BASE_SCORE)
    applied: List[str] = []
    for flag in flags:
        penalty = PENALTIES.get((flag, mode))
        if penalty is not None:
            score += penalty
            applied.append(f"{flag}_{mode}")
    final = max(SCORE_MIN, min(SCORE_MAX, score))
    return final, applied


def arm_b_score_all_modes(
    qid: str, cid: str, durations: Dict[str, float]
) -> Dict[str, Tuple[float, List[str]]]:
    """Score every available mode for this (qid, cid). Returns {mode: (final, applied)}."""
    flags = condition_to_flags(cid)
    out: Dict[str, Tuple[float, List[str]]] = {}
    for mode in ALL_MODES:
        if mode in durations:
            out[mode] = score_mode(mode, flags)
    return out


def arm_b_top1(
    scored: Dict[str, Tuple[float, List[str]]], durations: Dict[str, float]
) -> Optional[str]:
    """Pick the top-scoring mode. Tiebreak by travel time ascending."""
    if not scored:
        return None
    max_score = max(v[0] for v in scored.values())
    candidates = [m for m, v in scored.items() if v[0] == max_score]
    if len(candidates) == 1:
        return candidates[0]
    if TIEBREAKER == "travel_time_ascending":
        return min(candidates, key=lambda m: durations.get(m, float("inf")))
    if TIEBREAKER == "alphabetical":
        return sorted(candidates)[0]
    raise ValueError(f"Unknown TIEBREAKER: {TIEBREAKER}")


# ---------------------------------------------------------------------------
# Arm C — LLM ERA (extracted from trials.jsonl)
# ---------------------------------------------------------------------------

def arm_c_top1(trial: Dict[str, Any]) -> Optional[str]:
    """Resolve era_output.recommended_index → mode by matching against options."""
    era = trial.get("era_output") or {}
    rec_idx = era.get("recommended_index")
    if rec_idx is None:
        return None
    # Direct mapping for negative indices (works even if option list is empty)
    if rec_idx in ROUTE_INDEX_TO_MODE:
        # Verify option exists for this index — otherwise fall back to lookup
        for opt in era.get("options", []):
            if opt.get("route_index") == rec_idx:
                return opt.get("mode") or ROUTE_INDEX_TO_MODE[rec_idx]
        return ROUTE_INDEX_TO_MODE[rec_idx]
    # Positive index: must be a transit option
    for opt in era.get("options", []):
        if opt.get("route_index") == rec_idx:
            return opt.get("mode")
    return None


def llm_score_for_mode(trial: Dict[str, Any], mode: str) -> Optional[float]:
    """Return the LLM's score for the option matching `mode`.

    For transit (multiple route_indices possible) returns the highest score.
    """
    era = trial.get("era_output") or {}
    matches = [opt.get("score") for opt in era.get("options", [])
               if opt.get("mode") == mode and opt.get("score") is not None]
    if not matches:
        return None
    return float(max(matches))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_arm_b_scores(rows: List[Dict[str, Any]]) -> None:
    with open(ARM_B_OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_results_csv(rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "query_id", "condition_id",
        "no_era_top1", "rule_top1", "llm_top1",
        "no_era_rule_agree", "rule_llm_agree", "no_era_llm_agree",
        "llm_rule_disagree_reason",
    ]
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_summary(summary: Dict[str, Any]) -> None:
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agree_rate(rows: List[Dict[str, Any]], col: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[col]) / len(rows)


def build_summary(
    rows: List[Dict[str, Any]],
    trials_by_pair: Dict[Tuple[str, str], Dict[str, Any]],
    queries_meta: Dict[str, Dict[str, Any]],
    raw_trial_count: int,
    error_records_dropped: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_trials": len(rows),
        "n_queries": len({r["query_id"] for r in rows}),
        "n_conditions": len({r["condition_id"] for r in rows}),
        "provenance": {
            "trials_source": str(TRIALS_PATH.relative_to(PROJECT_ROOT)),
            "candidates_source": str(CACHE_PATH.relative_to(PROJECT_ROOT)),
            "queries_source": str(QUERIES_PATH.relative_to(PROJECT_ROOT)),
            "raw_trial_records": raw_trial_count,
            "error_records_dropped": error_records_dropped,
            "tiebreaker": TIEBREAKER,
            "temperature_policy": TEMPERATURE_POLICY,
            "base_score": BASE_SCORE,
            "score_clamp": [SCORE_MIN, SCORE_MAX],
        },
    }

    # Overall agreement
    summary["overall"] = {
        "rule_llm_agreement_rate": round(_agree_rate(rows, "rule_llm_agree"), 4),
        "no_era_llm_agreement_rate": round(_agree_rate(rows, "no_era_llm_agree"), 4),
        "no_era_rule_agreement_rate": round(_agree_rate(rows, "no_era_rule_agree"), 4),
    }

    # By condition
    by_cond: Dict[str, Dict[str, Any]] = {}
    for cid in sorted({r["condition_id"] for r in rows}):
        sub = [r for r in rows if r["condition_id"] == cid]
        by_cond[cid] = {
            "rule_llm_agreement_rate": round(_agree_rate(sub, "rule_llm_agree"), 4),
            "no_era_llm_agreement_rate": round(_agree_rate(sub, "no_era_llm_agree"), 4),
            "no_era_rule_agreement_rate": round(_agree_rate(sub, "no_era_rule_agree"), 4),
            "n_trials": len(sub),
        }
    summary["by_condition"] = by_cond

    # Simple vs compound
    def _bucket_stats(cids: List[str]) -> Dict[str, Any]:
        sub = [r for r in rows if r["condition_id"] in cids]
        return {
            "conditions": cids,
            "rule_llm_agreement_rate": round(_agree_rate(sub, "rule_llm_agree"), 4),
            "no_era_llm_agreement_rate": round(_agree_rate(sub, "no_era_llm_agree"), 4),
            "no_era_rule_agreement_rate": round(_agree_rate(sub, "no_era_rule_agree"), 4),
            "n_trials": len(sub),
        }
    summary["by_condition_type"] = {
        "simple": _bucket_stats(SIMPLE_CONDITIONS),
        "compound": _bucket_stats(COMPOUND_CONDITIONS),
    }

    # Disagreement breakdown (rule vs LLM)
    disagree: Counter = Counter()
    for r in rows:
        if not r["rule_llm_agree"]:
            disagree[f"rule={r['rule_top1']}_llm={r['llm_top1']}"] += 1
    summary["disagreement_breakdown"] = dict(disagree.most_common())

    # Bonus: by dominant_exposed_mode (from query metadata)
    by_dom: Dict[str, Dict[str, Any]] = {}
    for dom in sorted({queries_meta[r["query_id"]].get("dominant_exposed_mode", "?")
                       for r in rows}):
        sub = [r for r in rows
               if queries_meta[r["query_id"]].get("dominant_exposed_mode") == dom]
        if sub:
            by_dom[dom] = {
                "rule_llm_agreement_rate": round(_agree_rate(sub, "rule_llm_agree"), 4),
                "no_era_llm_agreement_rate": round(_agree_rate(sub, "no_era_llm_agree"), 4),
                "n_trials": len(sub),
            }
    summary["by_dominant_exposed_mode"] = by_dom

    # Bonus: by distance category
    by_dist: Dict[str, Dict[str, Any]] = {}
    for dist in sorted({queries_meta[r["query_id"]].get("distance_category", "?")
                        for r in rows}):
        sub = [r for r in rows
               if queries_meta[r["query_id"]].get("distance_category") == dist]
        if sub:
            by_dist[dist] = {
                "rule_llm_agreement_rate": round(_agree_rate(sub, "rule_llm_agree"), 4),
                "no_era_llm_agreement_rate": round(_agree_rate(sub, "no_era_llm_agree"), 4),
                "n_trials": len(sub),
            }
    summary["by_distance_category"] = by_dist

    # Bonus: trials_where_llm_reranks_rule + mean LLM score gap on disagreement
    rerank_count = 0
    score_gaps: List[float] = []
    for r in rows:
        if r["rule_llm_agree"]:
            continue
        rerank_count += 1
        trial = trials_by_pair.get((r["query_id"], r["condition_id"]))
        if not trial:
            continue
        s_llm = llm_score_for_mode(trial, r["llm_top1"]) if r["llm_top1"] else None
        s_rule = llm_score_for_mode(trial, r["rule_top1"]) if r["rule_top1"] else None
        if s_llm is not None and s_rule is not None:
            score_gaps.append(s_llm - s_rule)
    summary["overall"]["trials_where_llm_reranks_rule"] = rerank_count
    if score_gaps:
        summary["overall"]["mean_llm_score_gap_when_disagree"] = round(
            sum(score_gaps) / len(score_gaps), 3
        )
        summary["overall"]["n_score_gap_observations"] = len(score_gaps)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Section 5.3 ablation harness")
    print("=" * 70)
    print(f"  output dir: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  tiebreaker: {TIEBREAKER}")
    print(f"  temperature_policy: {TEMPERATURE_POLICY}")
    print()

    # --- Load ---
    print("[1] Loading inputs...")
    # Recount errors here for provenance (load_trials drops them silently)
    raw_count = 0
    error_records = 0
    with open(TRIALS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_count += 1
            try:
                if json.loads(line).get("error"):
                    error_records += 1
            except json.JSONDecodeError:
                pass
    trials_by_pair = load_trials()
    cache = load_candidates()
    queries_meta = load_query_metadata()
    print(f"  candidates_cache: {len(cache)} queries")
    print(f"  query_metadata:   {len(queries_meta)} queries")

    # --- Sanity checks ---
    print("\n[2] Sanity checks...")
    missing_cache = []
    missing_meta = []
    missing_modes_log: List[str] = []
    for (qid, cid) in trials_by_pair:
        if qid not in cache:
            missing_cache.append((qid, cid))
        if qid not in queries_meta:
            missing_meta.append((qid, cid))
        durations = get_mode_durations_seconds(qid, cache) if qid in cache else {}
        missing_modes = [m for m in ALL_MODES if m not in durations]
        if missing_modes:
            missing_modes_log.append(f"  {qid}/{cid}: missing modes {missing_modes}")
    if missing_cache:
        print(f"  WARNING: {len(missing_cache)} trials missing candidates_cache entries")
    if missing_meta:
        print(f"  WARNING: {len(missing_meta)} trials missing query metadata")
    if missing_modes_log:
        print(f"  NOTE: {len(missing_modes_log)} trials with <4 modes available")
        for m in missing_modes_log[:3]:
            print(m)
    else:
        print("  All trials have all 4 modes available.")

    # --- Compute arms ---
    print("\n[3] Computing arms...")
    arm_b_records: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    skipped = 0

    for (qid, cid), trial in sorted(trials_by_pair.items()):
        if qid not in cache:
            skipped += 1
            continue

        durations = get_mode_durations_seconds(qid, cache)

        # Arm A
        a_top = arm_a_top1(qid, cache)

        # Arm B
        b_scored = arm_b_score_all_modes(qid, cid, durations)
        b_top = arm_b_top1(b_scored, durations)
        for mode, (final_score, applied) in b_scored.items():
            arm_b_records.append({
                "query_id": qid,
                "condition_id": cid,
                "mode": mode,
                "base_score": BASE_SCORE,
                "penalties_applied": applied,
                "final_score": final_score,
            })

        # Arm C
        c_top = arm_c_top1(trial)

        # Agreements
        rule_llm_agree = (b_top is not None and b_top == c_top)
        no_era_llm_agree = (a_top is not None and a_top == c_top)
        no_era_rule_agree = (a_top is not None and a_top == b_top)

        if rule_llm_agree:
            disagree_reason = "agree"
        else:
            disagree_reason = f"rule={b_top} llm={c_top}"

        result_rows.append({
            "query_id": qid,
            "condition_id": cid,
            "no_era_top1": a_top,
            "rule_top1": b_top,
            "llm_top1": c_top,
            "no_era_rule_agree": no_era_rule_agree,
            "rule_llm_agree": rule_llm_agree,
            "no_era_llm_agree": no_era_llm_agree,
            "llm_rule_disagree_reason": disagree_reason,
        })

    if skipped:
        print(f"  WARNING: skipped {skipped} trials with no cache entry")

    print(f"  arm_b_records: {len(arm_b_records)}")
    print(f"  result_rows:   {len(result_rows)}")

    # --- Write files ---
    print("\n[4] Writing outputs...")
    write_arm_b_scores(arm_b_records)
    print(f"  wrote {ARM_B_OUT.relative_to(PROJECT_ROOT)}  ({len(arm_b_records)} records)")
    write_results_csv(result_rows)
    print(f"  wrote {RESULTS_CSV.relative_to(PROJECT_ROOT)}  ({len(result_rows)} rows)")

    summary = build_summary(
        result_rows, trials_by_pair, queries_meta,
        raw_trial_count=raw_count, error_records_dropped=error_records,
    )
    write_summary(summary)
    print(f"  wrote {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")

    # --- Stdout summary ---
    print("\n[5] Headline numbers")
    print("  " + "-" * 64)
    o = summary["overall"]
    print(f"  Rule  ↔  LLM  agreement: {o['rule_llm_agreement_rate']:.1%} "
          f"  ({sum(1 for r in result_rows if r['rule_llm_agree'])}/{len(result_rows)})")
    print(f"  No-ERA ↔ LLM agreement: {o['no_era_llm_agreement_rate']:.1%} "
          f"  ({sum(1 for r in result_rows if r['no_era_llm_agree'])}/{len(result_rows)})")
    print(f"  No-ERA ↔ Rule agreement: {o['no_era_rule_agreement_rate']:.1%} "
          f"  ({sum(1 for r in result_rows if r['no_era_rule_agree'])}/{len(result_rows)})")

    print(f"\n  Trials where LLM reranks Rule: {o.get('trials_where_llm_reranks_rule', 0)}")
    if "mean_llm_score_gap_when_disagree" in o:
        print(f"  Mean LLM score gap on disagreement (LLM_pick - Rule_pick): "
              f"{o['mean_llm_score_gap_when_disagree']:+.2f}")

    print("\n  Simple vs Compound (Rule↔LLM):")
    s = summary["by_condition_type"]["simple"]["rule_llm_agreement_rate"]
    c = summary["by_condition_type"]["compound"]["rule_llm_agreement_rate"]
    print(f"    simple   {SIMPLE_CONDITIONS}:  {s:.1%}")
    print(f"    compound {COMPOUND_CONDITIONS}: {c:.1%}")
    print(f"    delta:    {(s - c):+.1%}  (positive = simple agrees more, expected)")

    print("\n  Per-condition Rule↔LLM agreement:")
    for cid, stats in summary["by_condition"].items():
        bar = "#" * int(stats["rule_llm_agreement_rate"] * 25)
        print(f"    {cid}: {stats['rule_llm_agreement_rate']:.0%}  {bar}")

    print("\n  Top disagreement patterns (rule vs LLM):")
    for pattern, count in list(summary["disagreement_breakdown"].items())[:8]:
        print(f"    {count:3d}  {pattern}")

    print("\n[OK] Run analyze_ablation.py for the per-trial qualitative report.")
    print("=" * 70)


if __name__ == "__main__":
    main()
