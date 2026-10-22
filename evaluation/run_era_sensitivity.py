"""
ERA sensitivity evaluation harness.

Runs each query through the full pipeline ONCE to build a candidate cache,
then replays ERA across 8 conditions per query with synthetic proactive_context.

Output:
    evaluation/era_sensitivity/candidates_cache.json   # query_id -> enriched candidates
    evaluation/era_sensitivity/trials.jsonl            # one line per (query, condition) trial
    evaluation/era_sensitivity/run_meta.json           # run metadata

Auto-resumes: trials.jsonl is append-only and the harness skips (query_id, condition_id)
pairs already present.

Usage:
    python evaluation/run_era_sensitivity.py --phase smoke
    python evaluation/run_era_sensitivity.py --phase pilot
    python evaluation/run_era_sensitivity.py --phase full
    python evaluation/run_era_sensitivity.py --status
    python evaluation/run_era_sensitivity.py --rebuild-cache --ids Q01,Q02

Phases:
    smoke  — 1 query x 4 conditions (8 ERA calls, ~1 min)
    pilot  — 25 queries x 4 conditions (100 ERA calls, ~15 min)
    full   — 25 queries x 8 conditions (200 ERA calls, ~30 min)
"""

import sys
import os
import argparse
import json
import time
import hashlib
import contextlib
import io
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR = PROJECT_DIR / "evaluation" / "era_sensitivity"
CANDIDATES_CACHE = OUTPUT_DIR / "candidates_cache.json"
TRIALS_FILE = OUTPUT_DIR / "trials.jsonl"
META_FILE = OUTPUT_DIR / "run_meta.json"
QUERIES_FILE = PROJECT_DIR / "evaluation" / "era_sensitivity_queries.json"

PHASE_CONFIG = {
    "smoke": {"query_ids": ["Q01"],            "condition_ids": ["C1", "C5", "C2", "C8"]},
    "pilot": {"query_ids": None,               "condition_ids": ["C1", "C2", "C5", "C8"]},
    "full":  {"query_ids": None,               "condition_ids": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]},
}


# ---------------------------------------------------------------------------
# Output serialization helpers
# ---------------------------------------------------------------------------

def era_output_to_dict(era_out) -> dict:
    """Convert ERAOutput dataclass to JSON-serialisable dict."""
    return {
        "situation": asdict(era_out.situation),
        "dimension_analysis": {
            k: asdict(v) for k, v in era_out.dimension_analysis.items()
        },
        "options": [asdict(opt) for opt in era_out.options],
        "recommended_index": era_out.recommended_index,
        "recommendation_reason": era_out.recommendation_reason,
        "condition_summary": era_out.condition_summary,
        "warnings": era_out.warnings,
    }


def prompt_version_hash() -> str:
    """SHA1 of the ERA system prompt — for reproducibility tracking."""
    from prompts.era_prompts import ERA_SYSTEM_PROMPT
    return hashlib.sha1(ERA_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Resumability — load done trials & cached candidates
# ---------------------------------------------------------------------------

def load_done_trials() -> set:
    """Read trials.jsonl and return set of (query_id, condition_id) already completed."""
    if not TRIALS_FILE.exists():
        return set()
    done = set()
    with open(TRIALS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("error") is None:
                    done.add((rec["query_id"], rec["condition_id"]))
            except json.JSONDecodeError:
                continue
    return done


def load_candidate_cache() -> dict:
    """Load the per-query candidate cache."""
    if not CANDIDATES_CACHE.exists():
        return {}
    with open(CANDIDATES_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_candidate_cache(cache: dict) -> None:
    """Atomic write of candidate cache."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CANDIDATES_CACHE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)
    tmp.replace(CANDIDATES_CACHE)


def append_trial(record: dict) -> None:
    """Append one trial result to trials.jsonl."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRIALS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Phase A — Build candidate cache
# ---------------------------------------------------------------------------

def build_candidates_for_query(orch, query_dict: dict, suppress_logs: bool = True, force_all_modes: bool = True) -> dict:
    """Run pipeline for one query and extract route candidates + alternatives.

    When force_all_modes=True (default for the eval), appends "Compare routes." to the
    query — this matches one of the trigger phrases in orchestrator._wants_all_options()
    and forces enrichment of walking/cycling/driving alternatives, regardless of how the
    router LLM happens to classify a particular query. Without this, ~30-50% of queries
    end up with only 1-2 modes enriched depending on whether the router flags
    transport_mode as a missing entity.

    Returns a dict with keys:
        candidates, walking_alternative, driving_alternative, cycling_alternative,
        pipeline_seconds, success, error
    """
    qid = query_dict["id"]
    qtext_original = query_dict["query"]
    qtext = f"{qtext_original} Compare routes." if force_all_modes else qtext_original
    session_id = f"era_eval_{qid}"

    start = time.time()
    out_buffer = io.StringIO() if suppress_logs else None

    try:
        if out_buffer is not None:
            with contextlib.redirect_stdout(out_buffer):
                result = orch.process_query_full(qtext, session_id=session_id)
        else:
            result = orch.process_query_full(qtext, session_id=session_id)
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "pipeline_seconds": round(time.time() - start, 2),
            "candidates": [],
            "walking_alternative": None,
            "driving_alternative": None,
            "cycling_alternative": None,
        }

    pipeline_sec = round(time.time() - start, 2)
    specialist = result.get("specialist_results", {}) or {}
    neo4j_data = specialist.get("neo4j", {}) or {}

    candidates = neo4j_data.get("candidates") or []
    if not candidates and neo4j_data.get("route"):
        candidates = [neo4j_data["route"]]

    return {
        "success": True,
        "error": None,
        "pipeline_seconds": pipeline_sec,
        "candidates": candidates,
        "walking_alternative": specialist.get("walking_alternative"),
        "driving_alternative": specialist.get("driving_alternative"),
        "cycling_alternative": specialist.get("cycling_alternative"),
        "_neo4j_success": neo4j_data.get("success", False),
        "_query_original": qtext_original,
        "_query_sent": qtext,
        "_force_all_modes": force_all_modes,
        "_n_modes_enriched": sum(1 for k in ("walking_alternative", "driving_alternative", "cycling_alternative")
                                  if specialist.get(k)),
    }


# ---------------------------------------------------------------------------
# Phase B — ERA trials
# ---------------------------------------------------------------------------

def run_era_trial(era_agent, cached: dict, condition_ctx: dict, query_id: str, condition_id: str, prompt_hash: str) -> dict:
    """Call era_agent.rerank with cached candidates and synthetic context. Returns trial record."""
    record = {
        "query_id": query_id,
        "condition_id": condition_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt_hash": prompt_hash,
        "era_temperature": era_agent.temperature,
        "era_model": era_agent.model,
        "error": None,
    }

    has_options = (
        bool(cached.get("candidates"))
        or cached.get("walking_alternative")
        or cached.get("driving_alternative")
        or cached.get("cycling_alternative")
    )
    if not has_options:
        record["error"] = "No options to evaluate (candidates and alternatives both empty)"
        record["era_seconds"] = 0
        return record

    start = time.time()
    try:
        era_out = era_agent.rerank(
            candidates=cached.get("candidates", []),
            walking_alt=cached.get("walking_alternative"),
            driving_alt=cached.get("driving_alternative"),
            cycling_alt=cached.get("cycling_alternative"),
            proactive_context=condition_ctx,
        )
        record["era_seconds"] = round(time.time() - start, 2)
        record["era_output"] = era_output_to_dict(era_out)
    except Exception as e:
        record["era_seconds"] = round(time.time() - start, 2)
        record["error"] = f"{type(e).__name__}: {str(e)}"
    return record


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def report_status(queries: list, condition_ids: list) -> None:
    cache = load_candidate_cache()
    done = load_done_trials()

    print(f"Output dir: {OUTPUT_DIR}")
    print(f"  candidates cached: {len(cache)} / {len(queries)} queries")
    n_planned = len(queries) * len(condition_ids)
    print(f"  trials done:       {len(done)} / {n_planned} planned")
    failed = []
    if TRIALS_FILE.exists():
        with open(TRIALS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error"):
                        failed.append((rec["query_id"], rec["condition_id"], rec["error"][:60]))
                except json.JSONDecodeError:
                    continue
    if failed:
        print(f"  failed trials:     {len(failed)}")
        for qid, cid, err in failed[:5]:
            print(f"     - {qid} {cid}: {err}")
        if len(failed) > 5:
            print(f"     ... and {len(failed) - 5} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ERA sensitivity evaluation harness")
    parser.add_argument("--phase", choices=list(PHASE_CONFIG.keys()), default="smoke",
                        help="smoke (1xQ x 4xC), pilot (25xQ x 4xC), full (25xQ x 8xC)")
    parser.add_argument("--ids", default=None,
                        help="Override query IDs: 'Q01,Q05' or 'Q01-Q10'")
    parser.add_argument("--conditions", default=None,
                        help="Override condition IDs: 'C1,C5'")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="ERA LLM temperature for the eval (default 0.0)")
    parser.add_argument("--era-delay", type=float, default=2.0,
                        help="Seconds to sleep between ERA calls (default 2)")
    parser.add_argument("--pipeline-delay", type=float, default=5.0,
                        help="Seconds to sleep between full pipeline runs (default 5)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of candidate cache for the selected queries")
    parser.add_argument("--status", action="store_true",
                        help="Show progress and exit (does not run anything)")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Cap total trials this run (for safe iteration)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        all_queries = json.load(f)["queries"]
    qmap = {q["id"]: q for q in all_queries}

    phase_cfg = PHASE_CONFIG[args.phase]
    selected_ids = phase_cfg["query_ids"] or [q["id"] for q in all_queries]

    if args.ids:
        ids_arg = args.ids.strip()
        if "-" in ids_arg and "," not in ids_arg:
            a, b = ids_arg.split("-")
            ai = int(a.strip().lstrip("Q"))
            bi = int(b.strip().lstrip("Q"))
            selected_ids = [f"Q{n:02d}" for n in range(ai, bi + 1)]
        else:
            selected_ids = [s.strip() for s in ids_arg.split(",")]

    selected_queries = [qmap[qid] for qid in selected_ids if qid in qmap]
    selected_conditions = phase_cfg["condition_ids"]
    if args.conditions:
        selected_conditions = [c.strip() for c in args.conditions.split(",")]

    if args.status:
        report_status(selected_queries, selected_conditions)
        return

    print("=" * 70)
    print(f"ERA Sensitivity Evaluation — phase={args.phase}")
    print("=" * 70)
    print(f"Queries:    {len(selected_queries)} ({', '.join(selected_ids[:5])}{', ...' if len(selected_ids) > 5 else ''})")
    print(f"Conditions: {len(selected_conditions)} ({', '.join(selected_conditions)})")
    print(f"Total trials: {len(selected_queries) * len(selected_conditions)}")
    print(f"τ (ERA temperature): {args.temperature}")
    print()

    # ---- Build candidate cache ----
    from evaluation.condition_grid import CONDITIONS, build_condition

    print("[Phase A] Initializing pipeline...")
    init_start = time.time()
    from APP import create_app
    ctx = create_app(verbose=False)
    print(f"  Pipeline ready in {time.time() - init_start:.1f}s")

    cache = load_candidate_cache()

    queries_to_pipeline = []
    for q in selected_queries:
        if args.rebuild_cache or q["id"] not in cache or not cache[q["id"]].get("success"):
            queries_to_pipeline.append(q)

    if queries_to_pipeline:
        print(f"\n[Phase A] Building candidate cache for {len(queries_to_pipeline)} queries...")
        # Disable ERA during pipeline to avoid wasted calls
        prev_era_enabled = ctx.orchestrator.era_enabled
        ctx.orchestrator.era_enabled = False
        try:
            for i, q in enumerate(queries_to_pipeline, 1):
                print(f"  [{i}/{len(queries_to_pipeline)}] {q['id']}: {q['query'][:70]}")
                result = build_candidates_for_query(ctx.orchestrator, q, suppress_logs=True)
                cache[q["id"]] = result
                save_candidate_cache(cache)
                if result["success"]:
                    n_cands = len(result["candidates"])
                    n_alts = sum(1 for k in ("walking_alternative", "driving_alternative", "cycling_alternative")
                                 if result.get(k))
                    print(f"     ok ({result['pipeline_seconds']}s): {n_cands} transit candidates + {n_alts} alts")
                else:
                    print(f"     FAILED: {result['error']}")
                if i < len(queries_to_pipeline) and args.pipeline_delay > 0:
                    time.sleep(args.pipeline_delay)
        finally:
            ctx.orchestrator.era_enabled = prev_era_enabled
    else:
        print("[Phase A] All candidates cached — skipping pipeline pass")

    # ---- Run ERA trials ----
    print(f"\n[Phase B] Running ERA trials...")

    if not ctx.orchestrator.era_agent:
        # Force-create ERA agent if it wasn't initialized (era_enabled=False at startup)
        from agents.era_agent import create_era_agent
        from config import (ERA_AGENT_MODEL, ERA_AGENT_TIMEOUT, ERA_AGENT_TEMPERATURE,
                            ERA_AGENT_MAX_TOKENS)
        ctx.orchestrator.era_agent = create_era_agent(
            client=ctx.client,
            config={
                "model": ERA_AGENT_MODEL,
                "timeout": ERA_AGENT_TIMEOUT,
                "temperature": args.temperature,
                "max_tokens": ERA_AGENT_MAX_TOKENS,
            },
            verbose=False,
        )

    era_agent = ctx.orchestrator.era_agent
    era_agent.temperature = args.temperature

    prompt_hash = prompt_version_hash()
    done = load_done_trials()

    # Build the trial list (skipping already-done)
    trials_planned = []
    for q in selected_queries:
        cached = cache.get(q["id"])
        if not cached or not cached.get("success"):
            print(f"  Skipping {q['id']}: no cached candidates")
            continue
        for cid in selected_conditions:
            if (q["id"], cid) in done:
                continue
            trials_planned.append((q, cid))

    if not trials_planned:
        print("  No trials remaining (all done) — use --rebuild-cache or --rerun to redo")
        report_status(selected_queries, selected_conditions)
        return

    if args.max_trials:
        trials_planned = trials_planned[: args.max_trials]

    print(f"  {len(trials_planned)} trials to run "
          f"({len(selected_queries) * len(selected_conditions) - len(trials_planned)} already done)")

    # Save run meta
    meta = {
        "phase": args.phase,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "selected_query_ids": selected_ids,
        "selected_condition_ids": selected_conditions,
        "n_trials_planned": len(trials_planned),
        "era_temperature": args.temperature,
        "era_model": era_agent.model,
        "prompt_hash": prompt_hash,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Execute
    n_ok = 0
    n_fail = 0
    for i, (q, cid) in enumerate(trials_planned, 1):
        ctx_dict = CONDITIONS[cid]
        cached = cache[q["id"]]
        elapsed_label = f"[{i}/{len(trials_planned)}]"
        print(f"  {elapsed_label} {q['id']} x {cid}: ", end="", flush=True)

        record = run_era_trial(era_agent, cached, ctx_dict, q["id"], cid, prompt_hash)
        append_trial(record)

        if record["error"]:
            n_fail += 1
            print(f"FAIL ({record['era_seconds']}s) — {record['error'][:70]}")
        else:
            n_ok += 1
            opts = record["era_output"]["options"]
            scores_str = " ".join(f"{o['mode'][:1]}={o['score']}" for o in opts[:4])
            rec_idx = record["era_output"]["recommended_index"]
            print(f"ok ({record['era_seconds']}s) — {scores_str} rec={rec_idx}")

        if i < len(trials_planned) and args.era_delay > 0:
            time.sleep(args.era_delay)

    print(f"\n[Done] {n_ok} ok, {n_fail} failed")
    report_status(selected_queries, selected_conditions)


if __name__ == "__main__":
    main()
