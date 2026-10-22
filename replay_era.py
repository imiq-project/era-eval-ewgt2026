"""
Standalone replay of the ERA sensitivity / stability trials from cached route candidates.

This script needs ONLY an OpenAI-compatible chat endpoint. It does not touch
Neo4j, FIWARE, OpenRouteService or TomTom: the route candidates that ERA
scores were computed once by the full pipeline and are shipped in
evaluation/era_sensitivity/candidates_cache.json.

Environment (real runs):
    OPENAI_API_KEY   API key for the endpoint
    OPENAI_BASE_URL  e.g. https://chat-ai.academiccloud.de/v1  (optional; default = api.openai.com)
    ERA_MODEL        model id, default qwen3-30b-a3b-instruct-2507 (the model used in the paper)

Usage:
    python replay_era.py --dry-run                         # plumbing check, no network: replays the recorded outputs
    python replay_era.py --queries Q01,Q02 --conditions C1,C5 --out replay.jsonl
    python replay_era.py --out replay_full.jsonl           # all 25 queries x 8 conditions (200 calls)
    python replay_era.py --stability --runs 5 --temperature 0.1 --out stability_replay.jsonl
                                                            # paper Sec. 5.4: 10 queries x 5 runs under C1

Output records have the same shape as evaluation/era_sensitivity/trials.jsonl, so
evaluation/analyze_full_pilot.py can be pointed at a replay file by copying it over
that path (or by editing the path at the top of that script).
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.condition_grid import CONDITIONS          # noqa: E402
from agents.era_agent import create_era_agent             # noqa: E402
from prompts.era_prompts import ERA_SYSTEM_PROMPT         # noqa: E402

CACHE_PATH = ROOT / "evaluation" / "era_sensitivity" / "candidates_cache.json"
TRIALS_PATH = ROOT / "evaluation" / "era_sensitivity" / "trials.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "era_sensitivity_queries.json"

STABILITY_QUERY_IDS = ["Q01", "Q03", "Q06", "Q08", "Q10", "Q11", "Q15", "Q18", "Q22", "Q25"]  # paper Sec. 5.4
DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"


def prompt_hash() -> str:
    """SHA1[:12] of the ERA system prompt. The paper's trials were run with 2da280da4306."""
    return hashlib.sha1(ERA_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


class RecordedClient:
    """Dry-run stand-in for the OpenAI client: returns the ERA output recorded for the
    (query, condition) pair currently being replayed. Exercises the full prompt build,
    JSON parsing and validation path without any network call."""

    def __init__(self, trials_path: Path):
        self.recorded = {}
        with open(trials_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("error"):
                    self.recorded[(r["query_id"], r["condition_id"])] = r["era_output"]
        self.current = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        assert params["messages"][0]["content"] == ERA_SYSTEM_PROMPT, "system prompt drifted"
        content = json.dumps(self.recorded[self.current])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_client(dry_run: bool):
    if dry_run:
        return RecordedClient(TRIALS_PATH)
    from openai import OpenAI  # imported lazily so --dry-run works without the package
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set (see module docstring)")
    return OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL") or None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=None, help="comma-separated query ids, default: all 25")
    ap.add_argument("--conditions", default=None, help="comma-separated condition ids, default: C1..C8")
    ap.add_argument("--runs", type=int, default=1, help="repeat each (query, condition) this many times")
    ap.add_argument("--temperature", type=float, default=0.0, help="ERA sampling temperature (paper: 0.0 sensitivity, 0.1 stability)")
    ap.add_argument("--stability", action="store_true", help="paper Sec. 5.4 setup: 10 fixed queries, C1 only; combine with --runs 5 --temperature 0.1")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between calls")
    ap.add_argument("--out", default="replay.jsonl", help="output JSONL path")
    ap.add_argument("--dry-run", action="store_true", help="no network: replay recorded outputs to check the plumbing")
    args = ap.parse_args()

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    query_ids = [q.strip() for q in args.queries.split(",")] if args.queries else sorted(cache)
    condition_ids = [c.strip() for c in args.conditions.split(",")] if args.conditions else list(CONDITIONS)
    if args.stability:
        query_ids = STABILITY_QUERY_IDS if not args.queries else query_ids
        condition_ids = ["C1"]

    model = os.environ.get("ERA_MODEL", DEFAULT_MODEL)
    client = make_client(args.dry_run)
    era = create_era_agent(client, config={"model": model, "timeout": 60, "max_retries": 1,
                                           "temperature": args.temperature, "max_tokens": 2500})

    print(f"prompt_hash={prompt_hash()} (paper: 2da280da4306)  model={model}  tau={args.temperature}  "
          f"{len(query_ids)} queries x {len(condition_ids)} conditions x {args.runs} runs  dry_run={args.dry_run}")

    n_ok = n_err = 0
    mismatches = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for qid in query_ids:
            cached = cache.get(qid)
            if not cached or not cached.get("success"):
                print(f"  skip {qid}: no cached candidates")
                continue
            for cid in condition_ids:
                for run in range(1, args.runs + 1):
                    if args.dry_run:
                        client.current = (qid, cid)
                    t0 = time.time()
                    rec = {"query_id": qid, "condition_id": cid, "run": run,
                           "timestamp": datetime.now(timezone.utc).isoformat(),
                           "prompt_hash": prompt_hash(), "era_temperature": args.temperature,
                           "era_model": model, "error": None, "era_seconds": None, "era_output": None}
                    try:
                        output = era.rerank(
                            candidates=cached["candidates"],
                            walking_alt=cached.get("walking_alternative"),
                            driving_alt=cached.get("driving_alternative"),
                            cycling_alt=cached.get("cycling_alternative"),
                            proactive_context=CONDITIONS[cid],
                        )
                        rec["era_output"] = to_jsonable(output)
                        n_ok += 1
                        if args.dry_run:
                            got = {o["mode"]: o["score"] for o in rec["era_output"]["options"]}
                            exp = {o["mode"]: o["score"] for o in client.recorded[(qid, cid)]["options"]}
                            if got != exp:
                                mismatches += 1
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {e}"
                        n_err += 1
                    rec["era_seconds"] = round(time.time() - t0, 2)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"  {qid} {cid} run{run}: {'ok' if rec['error'] is None else rec['error']}")
                    if args.delay and not args.dry_run:
                        time.sleep(args.delay)

    print(f"\ndone: {n_ok} ok, {n_err} errors -> {args.out}")
    if args.dry_run:
        print(f"dry-run score round-trip mismatches: {mismatches} (expect 0)")
        return 0 if mismatches == 0 and n_err == 0 else 1
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
