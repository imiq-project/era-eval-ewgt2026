"""
Read the ablation outputs and print the four reports requested in the
EVAL_SPEC §5.3 prompt:

  1. Overall agreement table (No-ERA vs LLM, Rule vs LLM)
  2. Agreement rate by condition type (simple vs compound)
  3. Top-5 most common rule↔LLM disagreement patterns
  4. C3 deep-dive: every trial where Rule ≠ LLM under C3, with the LLM's
     full justification text and per-option reasoning extracted from
     trials.jsonl. These are the candidates for paper qualitative examples.

Usage:
    python -m evaluation.ablation.analyze_ablation
    python evaluation/ablation/analyze_ablation.py
"""

import sys
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = PROJECT_ROOT / "evaluation"
SENS_DIR = EVAL_DIR / "era_sensitivity"
OUT_DIR = EVAL_DIR / "ablation"

TRIALS_PATH = SENS_DIR / "trials.jsonl"
QUERIES_PATH = EVAL_DIR / "era_sensitivity_queries.json"
RESULTS_CSV = OUT_DIR / "ablation_results.csv"
SUMMARY_JSON = OUT_DIR / "ablation_summary.json"


def load_results() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["no_era_rule_agree"] = (r["no_era_rule_agree"] == "True")
            r["rule_llm_agree"] = (r["rule_llm_agree"] == "True")
            r["no_era_llm_agree"] = (r["no_era_llm_agree"] == "True")
            rows.append(r)
    return rows


def load_summary() -> Dict[str, Any]:
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_trials_indexed() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Same dedupe logic as run_ablation.load_trials — kept self-contained."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(TRIALS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("error"):
                continue
            pair = (t["query_id"], t["condition_id"])
            existing = out.get(pair)
            if existing is None or t.get("timestamp", "") > existing.get("timestamp", ""):
                out[pair] = t
    return out


def load_queries() -> Dict[str, Dict[str, Any]]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["queries"]}


def get_option_for_mode(trial: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    """Return the option dict whose mode matches; for transit, returns the
    highest-scoring one (mirroring run_ablation's transit handling)."""
    options = (trial.get("era_output") or {}).get("options", []) or []
    matches = [o for o in options if o.get("mode") == mode]
    if not matches:
        return None
    return max(matches, key=lambda o: o.get("score", -1))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_overall(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print("=" * 74)
    print("REPORT 1 — Overall agreement")
    print("=" * 74)
    n = len(rows)
    o = summary["overall"]

    def _frac(rate: float) -> str:
        return f"{int(round(rate * n))}/{n}"

    print()
    print(f"  Pair                       Agreement     Count")
    print(f"  " + "-" * 50)
    print(f"  Rule  ↔  LLM        :        {o['rule_llm_agreement_rate']:6.1%}     {_frac(o['rule_llm_agreement_rate'])}")
    print(f"  No-ERA ↔  LLM       :        {o['no_era_llm_agreement_rate']:6.1%}     {_frac(o['no_era_llm_agreement_rate'])}")
    print(f"  No-ERA ↔  Rule      :        {o['no_era_rule_agreement_rate']:6.1%}     {_frac(o['no_era_rule_agreement_rate'])}")
    print()
    print(f"  Trials where LLM reranks Rule:  {o.get('trials_where_llm_reranks_rule', 0)}")
    if "mean_llm_score_gap_when_disagree" in o:
        print(f"  Mean LLM score gap on disagreement (LLM_pick − Rule_pick):  "
              f"{o['mean_llm_score_gap_when_disagree']:+.2f}")
        print(f"    (positive = LLM thinks its own pick is meaningfully better)")
    print()


def report_by_condition_type(summary: Dict[str, Any]) -> None:
    print("=" * 74)
    print("REPORT 2 — Agreement by condition type (simple vs compound)")
    print("=" * 74)
    bct = summary["by_condition_type"]
    s = bct["simple"]
    c = bct["compound"]
    print()
    print(f"  Simple    {s['conditions']}: Rule↔LLM = {s['rule_llm_agreement_rate']:.1%}  "
          f"(n={s['n_trials']})")
    print(f"  Compound  {c['conditions']}: Rule↔LLM = {c['rule_llm_agreement_rate']:.1%}  "
          f"(n={c['n_trials']})")
    delta = s["rule_llm_agreement_rate"] - c["rule_llm_agreement_rate"]
    sign = "simple agrees more" if delta > 0 else "compound agrees more" if delta < 0 else "equal"
    print(f"  Delta:    {delta:+.1%}  ({sign})")
    print()
    print("  Per-condition breakdown:")
    print(f"    {'CID':4s}  {'Rule↔LLM':>8s}  {'NoERA↔LLM':>10s}  n")
    for cid, stats in summary["by_condition"].items():
        bucket = "simple" if cid in s["conditions"] else "compound"
        print(f"    {cid:4s}  {stats['rule_llm_agreement_rate']:>8.1%}  "
              f"{stats['no_era_llm_agreement_rate']:>10.1%}  {stats['n_trials']:>2d}  ({bucket})")
    print()


def report_top_disagreements(summary: Dict[str, Any]) -> None:
    print("=" * 74)
    print("REPORT 3 — Top 5 most common rule ↔ LLM disagreement patterns")
    print("=" * 74)
    print()
    items = list(summary["disagreement_breakdown"].items())
    if not items:
        print("  (no disagreements)")
        print()
        return
    items.sort(key=lambda kv: -kv[1])
    print(f"    {'Count':>5s}  Pattern")
    print(f"    {'-' * 5}  {'-' * 50}")
    for pattern, count in items[:5]:
        print(f"    {count:>5d}  {pattern}")
    if len(items) > 5:
        print(f"    ...   {len(items) - 5} more pattern(s)")
    print()


def report_c3_deep_dive(
    rows: List[Dict[str, Any]],
    trials: Dict[Tuple[str, str], Dict[str, Any]],
    queries: Dict[str, Dict[str, Any]],
) -> None:
    """Every C3 trial where Rule ≠ LLM. Print query text, both picks, the LLM's
    top-level recommendation_reason, plus the LLM's per-option reasoning for
    BOTH the rule's pick and the LLM's pick. These are the qualitative-paper
    candidates."""
    print("=" * 74)
    print("REPORT 4 — C3 disagreements (light_rain, 5°C, noon, congested, scarce)")
    print("=" * 74)

    c3_rows = [r for r in rows if r["condition_id"] == "C3" and not r["rule_llm_agree"]]
    if not c3_rows:
        print("\n  No C3 disagreements between Rule and LLM. Surprising — "
              "look at C5 or C8 for richer cases.\n")
        return

    print(f"\n  Found {len(c3_rows)} C3 disagreement(s). Each is a paper-example candidate.\n")
    for r in c3_rows:
        qid = r["query_id"]
        rule_pick = r["rule_top1"]
        llm_pick = r["llm_top1"]
        meta = queries.get(qid, {})
        trial = trials.get((qid, "C3"))
        era = (trial or {}).get("era_output") or {}

        print("-" * 74)
        print(f"  {qid}: {meta.get('query', '(query text unavailable)')}")
        cat = meta.get("distance_category")
        dom = meta.get("dominant_exposed_mode")
        if cat or dom:
            print(f"    [distance={cat}, dominant_mode={dom}]")
        print(f"    Rule top-1:  {rule_pick}")
        print(f"    LLM  top-1:  {llm_pick}")
        if era.get("recommendation_reason"):
            print(f"    LLM rationale: {era['recommendation_reason']}")

        rule_opt = get_option_for_mode(trial, rule_pick) if trial else None
        llm_opt = get_option_for_mode(trial, llm_pick) if trial else None
        if llm_opt:
            print(f"    LLM scored {llm_pick}={llm_opt.get('score')} — "
                  f"{llm_opt.get('reasoning', '')}")
            if llm_opt.get("trade_off"):
                print(f"      trade_off: {llm_opt['trade_off']}")
        if rule_opt and rule_pick != llm_pick:
            print(f"    LLM scored {rule_pick}={rule_opt.get('score')} — "
                  f"{rule_opt.get('reasoning', '')}")
            if rule_opt.get("trade_off"):
                print(f"      trade_off: {rule_opt['trade_off']}")

        # Show ERA situation archetype for richer paper context
        sit = era.get("situation") or {}
        if sit.get("archetype"):
            print(f"    archetype:   {sit['archetype']} — {sit.get('archetype_reasoning','')}")
        print()
    print("=" * 74)


def report_strongest_examples(
    rows: List[Dict[str, Any]],
    trials: Dict[Tuple[str, str], Dict[str, Any]],
    queries: Dict[str, Dict[str, Any]],
    top_n: int = 5,
) -> None:
    """Bonus: across ALL conditions, surface the disagreements where the LLM's
    own scoring most strongly favours its pick over the rule's pick. These are
    the ones where the LLM has the clearest narrative to support."""
    print("=" * 74)
    print(f"REPORT 5 (bonus) — Top-{top_n} disagreements by LLM score gap")
    print("(where LLM most strongly disagrees with the rule, by its own scoring)")
    print("=" * 74)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        if r["rule_llm_agree"]:
            continue
        trial = trials.get((r["query_id"], r["condition_id"]))
        if not trial:
            continue
        rule_opt = get_option_for_mode(trial, r["rule_top1"])
        llm_opt = get_option_for_mode(trial, r["llm_top1"])
        if not (rule_opt and llm_opt):
            continue
        gap = float(llm_opt.get("score", 0)) - float(rule_opt.get("score", 0))
        scored.append((gap, r))

    if not scored:
        print("\n  (no disagreements with both options scored)\n")
        return

    scored.sort(key=lambda x: -x[0])
    print()
    for gap, r in scored[:top_n]:
        qid = r["query_id"]
        cid = r["condition_id"]
        meta = queries.get(qid, {})
        trial = trials[(qid, cid)]
        era = trial["era_output"]
        llm_opt = get_option_for_mode(trial, r["llm_top1"])
        rule_opt = get_option_for_mode(trial, r["rule_top1"])

        print("-" * 74)
        print(f"  [{cid}] {qid}: {meta.get('query', '(query text)')}")
        print(f"    Rule={r['rule_top1']} (LLM scored {rule_opt.get('score')})  →  "
              f"LLM={r['llm_top1']} (LLM scored {llm_opt.get('score')})  "
              f"gap={gap:+.1f}")
        if era.get("recommendation_reason"):
            print(f"    LLM rationale: {era['recommendation_reason']}")
        if llm_opt and llm_opt.get("reasoning"):
            print(f"    on {r['llm_top1']}: {llm_opt['reasoning']}")
        if rule_opt and rule_opt.get("reasoning"):
            print(f"    on {r['rule_top1']}: {rule_opt['reasoning']}")
        print()
    print("=" * 74)


def main() -> None:
    rows = load_results()
    summary = load_summary()
    trials = load_trials_indexed()
    queries = load_queries()

    print()
    report_overall(rows, summary)
    report_by_condition_type(summary)
    report_top_disagreements(summary)
    report_c3_deep_dive(rows, trials, queries)
    report_strongest_examples(rows, trials, queries, top_n=5)

    print()
    print("Files: ablation_results.csv, ablation_summary.json, arm_b_scores.jsonl")
    print("Source: era_sensitivity/trials.jsonl, era_sensitivity/candidates_cache.json")


if __name__ == "__main__":
    main()
