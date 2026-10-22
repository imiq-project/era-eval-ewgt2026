# Evaluation Specification (v1)

Locked scope for the EWGT2026 full paper (8 pages).
Four result elements, ~850 total trials, no human raters required.

---

## 5.1 Pipeline NLU Benchmark

**Goal.** Measure how reliably the system parses queries. Extends Table 1 from the extended abstract.

**Dataset.** ~100 queries (79 existing + ~21 added, balanced across the 7 categories).

| Metric | Definition |
|---|---|
| Intent accuracy | exact match vs. ground-truth intent label |
| Function selection | exact match of chosen Neo4j function / FIWARE parameter set |
| Entity resolution | F1 over predicted vs. expected entities |
| Latency p50, p95 | end-to-end wall-clock |

**Comparison.** Mono-agent vs. multi-agent, same model (Qwen3-30B).

---

## 5.2 ERA Sensitivity

**Goal.** Show ERA scores move correctly when conditions change — the title-claim experiment.

**Dataset.** 25 condition-relevant queries × 8 conditions = 200 trials.

**Condition grid** (8-way fractional factorial, balanced across all six dimensions):

| ID | Weather | Temp °C | Time | AQ (PM2.5) | Traffic | Parking |
|---|---|---|---|---|---|---|
| C1 | clear | 20 | noon | good | free | available |
| C2 | clear | 5 | 22:00 | good | congested | scarce |
| C3 | light rain 2 mm/h | 5 | noon | good | congested | scarce |
| C4 | light rain 2 mm/h | 20 | 22:00 | poor | free | available |
| C5 | heavy rain 10 mm/h | 5 | noon | poor | free | scarce |
| C6 | heavy rain 10 mm/h | 20 | 22:00 | good | congested | available |
| C7 | snow | −5 | noon | good | free | scarce |
| C8 | snow | −5 | 22:00 | poor | congested | available |

Each level of each dimension appears in 4 of 8 conditions, so effects are not confounded.

**Metrics.**
- **Monotonicity rate:** % of (query, dimension) pairs where ERA score moves in the expected direction as the condition worsens.
- **Sensitivity Δ:** mean |score(C1) − score(C5)| for exposed modes (walk, bike) on outdoor-relevant queries; mean |score(parking-available) − score(parking-scarce)| for car on car-relevant queries.
- **Spearman ρ** between rainfall intensity and walk/bike score; between parking availability and car score.

**Expected directions** (used to compute monotonicity):

| Condition shift | Affected mode | Expected ERA change |
|---|---|---|
| rainfall ↑ | walk, bike | ↓ |
| rainfall ↑ | car, covered transit | ≈ or slight ↓ |
| time → night | walk | ↓ |
| PM2.5 ↑ | walk, bike | ↓ |
| temp ↓ below 0 | walk, bike | ↓ |
| traffic ↑ | car | ↓ |
| parking → scarce | car | ↓ |
| parking → scarce | walk, bike, transit | ≈ or slight ↑ (relative) |

**Output.** Heatmap (queries × conditions, faceted by mode) + monotonicity summary table.

---

## 5.3 Ablation: Rule-Based vs. LLM ERA

**Goal.** Show LLM scoring captures trade-offs hand-coded rules miss. Justifies the LLM in the title.

**Arms** (same 200 trials):
- **A — no ERA:** rank by duration only.
- **B — rule-based ERA:** transparent linear penalties, e.g.

      walk_score = 8 − 0.3·max(0, rain_mm_h − 2) − 2·is_night − 1·(PM2.5 > 50) − 0.1·max(0, 5 − temp)
      car_score  = 8 − 1·is_congested − 2·is_parking_scarce
      bike_score = walk_score − 0.5·is_congested
      transit_score = 7 − 1·is_late_night

  symmetric formulas per mode.
- **C — LLM ERA:** current system.

**Metrics.**
- Top-1 ranking agreement: B vs. C, A vs. C.
- Number of trials where C reranks B.
- Qualitative: 2–3 hand-picked examples where C's reasoning is genuinely hard to express as a rule (e.g., "short walk + light rain vs. extra transfer to avoid scarce parking").

**Output.** Comparison table + 2–3 example boxes.

---

## 5.4 Reliability: Stability, Grounding, Faithfulness

**Goal.** Convince reviewers the LLM reasoning is trustworthy.

| Sub-metric | Method | Definition |
|---|---|---|
| Stability σ̄ | 10 queries × 5 reruns at τ = 0.1 | mean std-dev of ERA score across reruns |
| Grounding rate | single LLM judge call per justification (Claude or GPT, *not* Qwen — avoids self-preference bias) | % of justifications that correctly cite ≥ 1 dominant present condition (e.g., heavy rain, scarce parking) |
| Faithfulness rate | NLI (DeBERTa-v3-MNLI) on each justification claim vs. condition state | % of justifications with zero hallucinated conditions |

Conditions that count as "dominant present" for grounding: heavy rain, snow, late-night, poor AQ, congestion, scarce parking, freezing temperature.

**Output.** One compact 1-row table:

| | Stability σ̄ | Grounding rate | Faithfulness rate |
|---|---|---|---|
| LLM ERA | … | … | … |

---

## Implementation notes

- One harness runs all four sections; sensitivity, ablation, and stability share the same trial loop.
- FIWARE Agent needs a **mock mode** that returns canned condition values from the grid instead of hitting live sensors. Lives in `clients/fiware_client.py` behind an env flag.
- All raw LLM responses (router, ERA, synthesizer) are persisted per-trial so judge/NLI grading can be re-run without regenerating.
- Judge models for grounding: Claude Opus + GPT-5 (or 4o). Never Qwen.
- All judge prompts and the rule-based ERA formulas go in the appendix.

## Out of scope for this paper (deferred to journal version)

- Multi-judge LLM panel for subjective quality scoring.
- ERA discrimination metric as a standalone result (mentioned in passing).
- Historical Magdeburg backtest on real FIWARE archives.
- User study with human raters.
- Real-deployment latency analysis beyond p50/p95.
