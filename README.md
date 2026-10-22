# ERA evaluation bundle

Data, prompts and analysis code behind the evaluation in

> Bilgin, D., Winkler, L., Wendemuth, A., Kauschke, L. (2026). *Environment-Aware Mobility Recommendations Through Multi-Agent LLM Reasoning Over Urban Digital Twin Data.* Euro Working Group on Transportation Annual Meeting 2026 (EWGT2026), Transportation Research Procedia.

The bundle contains everything needed to recompute every number in Sections 5.2–5.4 of the paper from the recorded trial logs, and to re-run the ERA trials against any OpenAI-compatible model endpoint. It does not require access to the Neo4j knowledge graph, the FIWARE context broker or the routing services. The route candidates ERA scores were computed once by the full pipeline and are shipped here as a cache.

## What is in here

| Path | Content |
|---|---|
| `prompts/era_prompts.py` | The ERA system prompt (SHA1 prefix `2da280da4306`, matching `run_meta.json` and every trial record) and the prompt builder |
| `agents/era_agent.py`, `agents/base_agent.py`, `models.py` | The ERA agent as used in the paper (JSON validation, output dataclasses) |
| `evaluation/condition_grid.py` | The eight environmental conditions C1–C8 (Table 1) as the exact `proactive_context` dicts ERA received |
| `evaluation/era_sensitivity_queries.json` | The 25 condition-relevant routing queries (Sec. 5.2) |
| `evaluation/era_sensitivity/candidates_cache.json` | Cached route candidates per query (transit candidates + walking / cycling / driving alternatives), computed once by the pipeline |
| `evaluation/era_sensitivity/trials.jsonl` | All 204 raw ERA trial records (200 valid; 4 errored records are dropped by the analysis), model `qwen3-30b-a3b-instruct-2507`, temperature 0.0 |
| `evaluation/analyze_full_pilot.py` | Recomputes Fig. 2 means, the walk–drive asymmetry test (Eq. 2), bootstrap CI, paired t, Spearman ρ and monotonicity |
| `evaluation/build_era_heatmap.py`, `evaluation/figures/` | Fig. 2 and the script that draws it |
| `evaluation/ablation/` | Rule-based ERA (`run_ablation.py`, penalty table `PENALTIES`), the three-arm comparison (`analyze_ablation.py`), and its outputs (Table 3 = `ablation_summary.json`) |
| `evaluation/reliability/stability/` | 10 queries × 5 reruns at τ = 0.1 under C1 (Sec. 5.4 stability), raw + summary |
| `evaluation/reliability/grounding/` | GPT-5.4 judge verdicts for all 200 justifications (Sec. 5.4 grounding / faithfulness), raw + summary, plus the judge script `run_grounding.py` |
| `nlu/` | The 100-query NLU benchmark (`NLU_data.json`) and the per-query results of the single-agent and multi-agent runs (Table 2). The NLU harness itself needs the live backends and is not included. |
| `evaluation/EVAL_SPEC.md` | The evaluation specification the runs were locked to |
| `replay_era.py` | Standalone replay of the ERA trials against any OpenAI-compatible endpoint (see below) |
| `evaluation/run_era_sensitivity.py`, `evaluation/reliability/run_stability.py` | The original harness scripts, kept for provenance. They import the full application (`APP.create_app`) and therefore need the backends; use `replay_era.py` instead. |

## Recompute the paper's numbers (no model calls)

```bash
python evaluation/analyze_full_pilot.py          # Sec. 5.2: heatmap means, asymmetry -2.32 [-3.20, -1.48], t(24) = -5.20, rho, monotonicity
python evaluation/ablation/run_ablation.py       # Sec. 5.3: deterministic rule arm + agreement table (rewrites evaluation/ablation/*)
python evaluation/ablation/analyze_ablation.py   # Sec. 5.3: breakdowns
python evaluation/reliability/analyze_reliability.py   # Sec. 5.4: sigma-bar = 0.37, grounding 100 %, faithfulness 98.5 %
```

All scripts are stdlib-only except `build_era_heatmap.py` (numpy, matplotlib).

## Re-run the ERA trials against a model

```bash
pip install openai
export OPENAI_API_KEY=...           
export OPENAI_BASE_URL=...          
export ERA_MODEL=qwen3-30b-a3b-instruct-2507  

python replay_era.py --dry-run                                  # plumbing check, no network
python replay_era.py --queries Q01 --conditions C1,C5 --out t.jsonl
python replay_era.py --out replay_full.jsonl                    # full 25 x 8 grid, 200 calls
python replay_era.py --stability --runs 5 --temperature 0.1 --out stab.jsonl   # Sec. 5.4 stability setup
```

Replay records have the same schema as `trials.jsonl`. Expect scores to differ slightly from the recorded ones even at temperature 0 (provider-side nondeterminism). The paper's stability run quantifies this (σ̄ = 0.37 on a 1–10 scale at τ = 0.1).

## Provenance

- ERA model: `qwen3-30b-a3b-instruct-2507`; sensitivity/ablation trials at τ = 0.0 (2026-05-07), stability at τ = 0.1 (2026-05-08), judge `gpt-5.4` at τ = 0.0 (2026-05-09).
- The three worked examples in the ERA prompt overlap in condition space with C1, C2 and C5–C6; see the paper's Sec. 6.
- Conditions are synthetic (injected), origins/destinations are real Magdeburg places. No personal data.
