"""
Reads stability_summary.json and grounding_summary.json and prints the
final reliability table for §5.4 of the paper.

Usage:
    python evaluation/reliability/analyze_reliability.py
"""

import sys
import json
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

STABILITY_PATH = PROJECT_DIR / "evaluation" / "reliability" / "stability" / "stability_summary.json"
GROUNDING_PATH = PROJECT_DIR / "evaluation" / "reliability" / "grounding" / "grounding_summary.json"

STABILITY_THRESHOLD = 0.75   # mean sigma must be below this
GROUNDING_THRESHOLD = 0.85   # both rates must be above this


def fmt(x, digits=2):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def report_stability() -> None:
    print("=" * 74)
    print("STABILITY RESULTS")
    print("=" * 74)
    if not STABILITY_PATH.exists():
        print(f"  (missing: {STABILITY_PATH.relative_to(PROJECT_DIR)})")
        return
    s = json.loads(STABILITY_PATH.read_text(encoding="utf-8"))

    n_q = s.get("n_queries", "?")
    n_r = s.get("n_runs_per_query", "?")
    mean_overall = s.get("mean_sigma_overall")
    by_mode = s.get("mean_sigma_by_mode", {}) or {}
    max_sigma = s.get("max_sigma_observed")
    max_qid = s.get("query_with_max_sigma")
    max_mode = s.get("mode_with_max_sigma")
    score_range = s.get("score_range_observed", {}) or {}

    print(f"  Mean sigma across {n_q} queries x {n_r} runs: {fmt(mean_overall)} (scale 1-10)")

    parts = []
    for m in ("walking", "cycling", "transit", "driving"):
        v = by_mode.get(m)
        parts.append(f"{m}={fmt(v)}")
    print(f"  Mean sigma by mode: " + "  ".join(parts))

    print(f"  Max sigma observed: {fmt(max_sigma)} "
          f"(query {max_qid or '?'}, mode {max_mode or '?'})")
    sr_min = score_range.get("min")
    sr_max = score_range.get("max")
    print(f"  Score range across all stability trials: {sr_min} - {sr_max}")

    if mean_overall is None:
        verdict = "INSUFFICIENT DATA"
    elif mean_overall < STABILITY_THRESHOLD:
        verdict = f"STABLE (mean sigma {mean_overall:.2f} < {STABILITY_THRESHOLD})"
    else:
        verdict = f"UNSTABLE (mean sigma {mean_overall:.2f} >= {STABILITY_THRESHOLD})"
    print(f"  Verdict: {verdict}")
    print()


def report_grounding() -> None:
    print("=" * 74)
    print("GROUNDING + FAITHFULNESS RESULTS")
    print("=" * 74)
    if not GROUNDING_PATH.exists():
        print(f"  (missing: {GROUNDING_PATH.relative_to(PROJECT_DIR)})")
        return
    g = json.loads(GROUNDING_PATH.read_text(encoding="utf-8"))

    n_total = g.get("n_trials_total", "?")
    n_eval = g.get("n_evaluated", "?")
    n_err = g.get("n_judge_errors", 0)
    grounding = g.get("grounding_rate")
    faithfulness = g.get("faithfulness_rate")

    print(f"  Total trials:     {n_total}  ({n_err} judge errors, {n_eval} evaluated)")
    print(f"  Grounding rate:   {grounding*100 if grounding is not None else 0:.1f}%")
    print(f"  Faithfulness rate:{faithfulness*100 if faithfulness is not None else 0:.1f}%")

    by_cond = g.get("grounding_by_condition", {})
    if by_cond:
        print()
        print("  Grounding by condition:")
        print(f"    {'CID':<5s}  {'Grounding':>10s}  {'Faithfulness':>13s}  {'n':>4s}")
        for cid, c in sorted(by_cond.items()):
            gr = c.get("grounding_rate")
            fa = c.get("faithfulness_rate")
            n = c.get("n_judged", c.get("n", 0))
            print(f"    {cid:<5s}  "
                  f"{(gr*100 if gr is not None else 0):>9.1f}%  "
                  f"{(fa*100 if fa is not None else 0):>12.1f}%  "
                  f"{n:>4d}")

    by_mode = g.get("grounding_by_mode", {})
    if by_mode:
        print()
        print("  Grounding by recommended mode:")
        print(f"    {'mode':<10s}  {'Grounding':>10s}  {'n':>4s}")
        for mode, c in sorted(by_mode.items(), key=lambda kv: -kv[1].get("n", 0)):
            gr = c.get("grounding_rate")
            n = c.get("n", 0)
            print(f"    {mode:<10s}  "
                  f"{(gr*100 if gr is not None else 0):>9.1f}%  "
                  f"{n:>4d}")

    violations = g.get("faithfulness_violations") or []
    print()
    print(f"  Faithfulness violations: {len(violations)} cases")
    for v in violations[:10]:
        snippet = (v.get("reasoning_text") or "").replace("\n", " ").strip()
        if len(snippet) > 130:
            snippet = snippet[:127] + "..."
        print(f"    - {v.get('query_id')} x {v.get('condition_id')} "
              f"[{v.get('recommended_mode', '?')}]")
        print(f"      \"{snippet}\"")
    if len(violations) > 10:
        print(f"    ... and {len(violations) - 10} more (see grounding_summary.json)")

    if grounding is None or faithfulness is None:
        verdict = "INSUFFICIENT DATA"
    elif grounding > GROUNDING_THRESHOLD and faithfulness > GROUNDING_THRESHOLD:
        verdict = f"PASS (both rates > {GROUNDING_THRESHOLD*100:.0f}%)"
    else:
        verdict = f"REVIEW NEEDED (threshold: both rates > {GROUNDING_THRESHOLD*100:.0f}%)"
    print()
    print(f"  Verdict: {verdict}")
    print()


def main() -> None:
    report_stability()
    report_grounding()
    print("Files:")
    print(f"  stability  -> {STABILITY_PATH.relative_to(PROJECT_DIR)}")
    print(f"  grounding  -> {GROUNDING_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
