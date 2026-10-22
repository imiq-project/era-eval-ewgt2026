"""
Section 5.4 Experiment 1 — Stability.

Re-runs ERA 5 times on each of 10 selected queries under condition C1 at
temperature 0.1, using the candidate cache from the sensitivity evaluation.
The full pipeline is NOT re-executed; only ERA is re-evaluated.

Output:
    evaluation/reliability/stability/stability_trials.jsonl  (one record per qid x run x mode)
    evaluation/reliability/stability/stability_summary.json

Usage:
    python evaluation/reliability/run_stability.py
"""

import sys
import os
import json
import time
import contextlib
import io
import statistics
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Mix: 3 short, 4 medium, 3 long per the brief
QUERY_IDS = ["Q01", "Q03", "Q06", "Q08", "Q10", "Q11", "Q15", "Q18", "Q22", "Q25"]
N_RUNS = 5
CONDITION_ID = "C1"
TEMPERATURE = 0.1
PACE_SECONDS = 12.0
MODES = ("walking", "cycling", "transit", "driving")

OUT_DIR = PROJECT_DIR / "evaluation" / "reliability" / "stability"
TRIALS_PATH = OUT_DIR / "stability_trials.jsonl"
SUMMARY_PATH = OUT_DIR / "stability_summary.json"
CACHE_PATH = PROJECT_DIR / "evaluation" / "era_sensitivity" / "candidates_cache.json"


def append_trial(record: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def extract_mode_scores(era_output) -> dict:
    """Return {mode: max_score_or_None} from an ERAOutput.

    For transit (multiple route_indices possible) takes the highest score —
    same convention as evaluation/ablation/run_ablation.py:llm_score_for_mode.
    """
    by_mode = {m: [] for m in MODES}
    for opt in era_output.options:
        if opt.mode in by_mode:
            by_mode[opt.mode].append(opt.score)
    return {m: (max(v) if v else None) for m, v in by_mode.items()}


def main() -> None:
    print("=" * 70)
    print(f"§5.4 STABILITY — {len(QUERY_IDS)} queries x {N_RUNS} runs at tau={TEMPERATURE}")
    print(f"Condition: {CONDITION_ID}  |  Pace: {PACE_SECONDS}s between calls")
    print("=" * 70)

    # --- Wipe prior trials so we get a clean run (no resume for stability) ---
    if TRIALS_PATH.exists():
        backup = TRIALS_PATH.with_suffix(f".jsonl.bak.{int(time.time())}")
        TRIALS_PATH.rename(backup)
        print(f"  Existing trials backed up -> {backup.name}")

    # --- Load cache ---
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    missing = [q for q in QUERY_IDS if q not in cache or not cache[q].get("success")]
    if missing:
        sys.exit(f"FATAL: missing or failed cache entries for: {missing}")
    print(f"  Cache: all {len(QUERY_IDS)} queries available")

    # --- Init pipeline (we only need era_agent) ---
    print("\n[Init] Loading APP context...")
    init_start = time.time()
    from APP import create_app
    ctx = create_app(verbose=False)
    print(f"  Done in {time.time() - init_start:.1f}s")

    # Force-create ERA agent if not already initialized (defensive)
    if not ctx.orchestrator.era_agent:
        from agents.era_agent import create_era_agent
        from config import (ERA_AGENT_MODEL, ERA_AGENT_TIMEOUT,
                            ERA_AGENT_MAX_TOKENS)
        ctx.orchestrator.era_agent = create_era_agent(
            client=ctx.client,
            config={
                "model": ERA_AGENT_MODEL,
                "timeout": ERA_AGENT_TIMEOUT,
                "temperature": TEMPERATURE,
                "max_tokens": ERA_AGENT_MAX_TOKENS,
            },
            verbose=False,
        )
    era_agent = ctx.orchestrator.era_agent
    era_agent.temperature = TEMPERATURE  # force tau=0.1 for stability
    print(f"  ERA agent ready (model={era_agent.model}, tau={era_agent.temperature})")

    # --- Build C1 context ---
    from evaluation.condition_grid import CONDITIONS
    c1_context = CONDITIONS[CONDITION_ID]

    # --- Run loop ---
    total_calls = len(QUERY_IDS) * N_RUNS
    n_done = 0
    n_fail = 0
    run_start = time.time()
    # Track scores for summary: scores_by_qid_mode[qid][mode] = [score per run]
    scores_by_qid_mode: dict = {q: {m: [] for m in MODES} for q in QUERY_IDS}

    for qid in QUERY_IDS:
        cached = cache[qid]
        for run_num in range(1, N_RUNS + 1):
            n_done += 1
            label = f"[{n_done}/{total_calls}]"
            print(f"  {label} {qid} run {run_num}/5  ", end="", flush=True)

            call_start = time.time()
            try:
                # Suppress orchestrator's verbose ERA stdout — we only need the dataclass
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    era_out = era_agent.rerank(
                        candidates=cached.get("candidates", []),
                        walking_alt=cached.get("walking_alternative"),
                        driving_alt=cached.get("driving_alternative"),
                        cycling_alt=cached.get("cycling_alternative"),
                        proactive_context=c1_context,
                    )
                elapsed = time.time() - call_start
                mode_scores = extract_mode_scores(era_out)
                ts = datetime.utcnow().isoformat() + "Z"

                for mode, score in mode_scores.items():
                    record = {
                        "query_id": qid,
                        "run_number": run_num,
                        "condition_id": CONDITION_ID,
                        "mode": mode,
                        "score": score,
                        "era_temperature": TEMPERATURE,
                        "era_model": era_agent.model,
                        "timestamp": ts,
                    }
                    append_trial(record)
                    if score is not None:
                        scores_by_qid_mode[qid][mode].append(score)

                # Compact log
                summary = " ".join(f"{m[0]}={mode_scores[m]}" for m in MODES)
                print(f"{summary}  ({elapsed:.1f}s)")

            except Exception as e:
                n_fail += 1
                elapsed = time.time() - call_start
                print(f"FAIL ({elapsed:.1f}s): {type(e).__name__}: {str(e)[:80]}")
                # Record the failure so we can see it in the trials file
                append_trial({
                    "query_id": qid,
                    "run_number": run_num,
                    "condition_id": CONDITION_ID,
                    "mode": None,
                    "score": None,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "era_temperature": TEMPERATURE,
                    "era_model": era_agent.model,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })

            # Pace BETWEEN calls (not after last)
            if n_done < total_calls and PACE_SECONDS > 0:
                time.sleep(PACE_SECONDS)

    wall = time.time() - run_start

    # --- Compute summary ---
    print(f"\n[Summary] computing sigmas...")
    per_query_per_mode_sigma: dict = {}
    all_scores = []
    sigma_by_mode = {m: [] for m in MODES}
    max_sigma = 0.0
    max_sigma_qid = None
    max_sigma_mode = None

    for qid in QUERY_IDS:
        per_query_per_mode_sigma[qid] = {}
        for mode in MODES:
            scores = scores_by_qid_mode[qid][mode]
            if len(scores) >= 2:
                sigma = round(statistics.stdev(scores), 4)
            elif len(scores) == 1:
                sigma = 0.0
            else:
                sigma = None
            per_query_per_mode_sigma[qid][mode] = sigma
            if sigma is not None:
                sigma_by_mode[mode].append(sigma)
                if sigma > max_sigma:
                    max_sigma = sigma
                    max_sigma_qid = qid
                    max_sigma_mode = mode
            all_scores.extend(scores)

    mean_sigma_overall = (
        round(statistics.mean(
            sig for q in per_query_per_mode_sigma.values()
            for sig in q.values()
            if sig is not None
        ), 4)
        if any(sig is not None for q in per_query_per_mode_sigma.values() for sig in q.values())
        else None
    )

    mean_sigma_by_mode = {
        m: (round(statistics.mean(v), 4) if v else None)
        for m, v in sigma_by_mode.items()
    }

    score_range = (
        {"min": min(all_scores), "max": max(all_scores)}
        if all_scores else {"min": None, "max": None}
    )

    summary = {
        "n_queries": len(QUERY_IDS),
        "n_runs_per_query": N_RUNS,
        "condition": CONDITION_ID,
        "temperature": TEMPERATURE,
        "era_model": era_agent.model,
        "n_calls_attempted": total_calls,
        "n_calls_failed": n_fail,
        "wall_clock_seconds": round(wall, 1),
        "per_query_per_mode_sigma": per_query_per_mode_sigma,
        "mean_sigma_overall": mean_sigma_overall,
        "mean_sigma_by_mode": mean_sigma_by_mode,
        "max_sigma_observed": round(max_sigma, 4),
        "query_with_max_sigma": max_sigma_qid,
        "mode_with_max_sigma": max_sigma_mode,
        "score_range_observed": score_range,
        "queries": QUERY_IDS,
        "trials_path": str(TRIALS_PATH.relative_to(PROJECT_DIR)),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] {n_done - n_fail} ok, {n_fail} failed")
    print(f"  Wall clock: {wall/60:.1f} min")
    print(f"  Mean sigma overall: {mean_sigma_overall}")
    print(f"  Mean sigma by mode: {mean_sigma_by_mode}")
    print(f"  Max sigma: {max_sigma:.4f} ({max_sigma_qid}/{max_sigma_mode})")
    print(f"  Score range: {score_range}")
    print(f"\n  trials  -> {TRIALS_PATH.relative_to(PROJECT_DIR)}")
    print(f"  summary -> {SUMMARY_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
