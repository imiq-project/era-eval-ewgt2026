# Grounding & Faithfulness Results — §5.4 Experiment 2

**Run completed:** 2026-05-09
**Wall clock:** 9.6 min
**Judge model:** `gpt-5.4` (OpenAI; never Qwen, to avoid self-evaluation bias)
**Judge temperature:** 0.0
**Trials:** 200/200 succeeded, 0 judge errors
**Trials file:** [grounding_trials.jsonl](grounding_trials.jsonl)
**Summary file:** [grounding_summary.json](grounding_summary.json)

---

## Headline numbers (paper Table 5)

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| **Grounding rate** | **100.0%** (200/200) | > 85% | **PASS** |
| **Faithfulness rate** | **98.5%** (197/200 clean) | > 85% | **PASS** |

- **Grounding** — % of justifications where the judge agreed ERA explicitly referenced at least one injected condition (weather, temp, time, AQ, traffic, parking).
- **Faithfulness** — % of justifications where the judge found *no* invented conditions absent from the injected set. Q2=YES means hallucination; faithfulness rate = 1 − Q2_yes_rate.

---

## What this means

**Grounding 100%** — ERA always engages with the digital-twin context it receives. There is no trial where the LLM produced a generic justification ignoring the conditions. The "environment-aware" claim of the paper is supported empirically: every reasoning text mentions at least one current condition.

**Faithfulness 98.5%** — ERA almost never invents conditions that weren't in the injected `proactive_context`. The 3 violations (1.5%) are all under extreme winter conditions and are described in detail below.

---

## Per-condition breakdown

| Condition | Weather | Time | Grounding | Faithfulness |
|---|---|---|---|---|
| C1 | clear | noon | 100% | **100%** |
| C2 | clear | 22:00 | 100% | **100%** |
| C3 | light rain | noon | 100% | **100%** |
| C4 | light rain | 22:00 | 100% | **100%** |
| C5 | heavy rain | noon | 100% | **100%** |
| C6 | heavy rain | 22:00 | 100% | **100%** |
| C7 | snow | noon | 100% | 96% (1 violation) |
| C8 | snow | 22:00 | 100% | 92% (2 violations) |

All faithfulness violations occur under **snow conditions (C7/C8)**. C1–C6 are perfect.

---

## Per-mode breakdown (recommended mode)

| Mode | Grounding | n |
|---|---|---|
| driving | 100% | 129 |
| transit | 100% | 66 |
| cycling | 100% | 4 |
| walking | 100% | 1 |

Driving was the recommended mode in 65% of trials — consistent with bad-weather conditions favoring sheltered modes.

---

## Detailed inspection of the 3 violations

All three are **driving recommendations under snow conditions** (C7 noon snow, C8 night snow). All come from queries where ERA correctly identified driving as the safer choice.

### Violation 1 — Q02 × C7 (snow, noon)

**Injected:** snow ~5 cm/h, −5°C feels-like −10°C, wind 18 km/h, daylight, good AQ, free traffic, scarce parking (3/100).

**ERA reasoning:**
> "Driving offers full shelter from snow, wind, and extreme cold. Minimal outdoor exposure (only 2 minutes walking to/from car) and **no risk of slipping on icy paths**. Despite the short trip, the weather makes active modes dangerous."

**What the judge flagged:** "icy paths" — ice was not explicitly in the injected condition list.

**Assessment:** **Borderline / defensible inference**. At −5°C with active snowfall, ice is a near-certain consequence. ERA is doing what a knowledgeable local would do: connecting injected facts to safety implications. This isn't fabrication, it's reasoning. The judge applied a strict reading of "mentions any condition not present."

### Violation 2 — Q15 × C8 (snow, night, congested, parking *available*)

**Injected:** snow ~5 cm/h, −5°C, wind 18 km/h, dark (22:00), unhealthy AQ, congested traffic, **parking AVAILABLE (80/100 spaces free)**.

**ERA reasoning:**
> "Driving offers shelter from extreme cold, snow, wind, and darkness. Despite congestion and **moderate parking uncertainty**, the car provides safety and minimal outdoor exposure. Generalized cost is lowest."

**What the judge flagged:** "moderate parking uncertainty" — parking was 80/100 free, not uncertain.

**Assessment:** **Real but minor misread.** ERA either skimmed the parking value or applied a bad-night-driving heuristic that overrides the data. The recommendation itself (drive) is still correct given the dominant conditions.

### Violation 3 — Q17 × C8 (snow, night, congested, parking *available*)

**Injected:** identical to Q15 × C8 above.

**ERA reasoning:**
> "The car provides full protection from snow, wind, and extreme cold. Darkness is less of a concern due to vehicle lighting and visibility. Despite congestion and **moderate parking risk**, the shelter and safety advantages are decisive. The 9-minute drive is fast and efficient."

**What the judge flagged:** "moderate parking risk" — same misread as Q15.

**Assessment:** **Real but minor misread**, same pattern as Q15. The same C8 condition (snow + night + congestion) appears to push ERA into a generic "winter driving has parking caveats" narrative regardless of the actual parking data.

### Pattern summary

| Type | Count |
|---|---|
| Defensible inference (ice from snow + −5°C) | 1 |
| Real misread (parking caveat under abundant parking) | 2 |

The 2 parking misreads are the **same** mistake under the **same** condition (C8). Suggests a single LLM tendency: when multiple bad-weather factors stack (snow + dark + congestion), it appends a generic parking concern that's decoupled from the actual parking signal. Worth flagging in the paper as a known limitation, but not a credibility issue at this scale.

---

## Methodological note — earlier judge false positives

A first run produced 33 violations (faithfulness 83.5%). All 33 mentioned "wind". The cause was a bug in `build_condition_description()`: it omitted wind, even though `condition_grid.py:_build_weather` injects `wind_speed` into the proactive_context that ERA receives. ERA correctly mentioned wind under bad-weather conditions; the judge flagged it as hallucination because it wasn't in the listed conditions.

**Fix:** `build_condition_description()` now reads `CONDITIONS[cid]` directly and surfaces every field ERA actually sees (wind, feels-like temp, humidity, NO2, day-of-week). The fix is in `evaluation/reliability/run_grounding.py`.

After the fix: 33 violations → 3 violations. The 30 wind-related false positives disappeared, confirming they were artifacts of incomplete judge briefing, not real ERA hallucinations.

The earlier run's outputs are preserved at `grounding_trials.jsonl.bak.<timestamp>` for reproducibility audit.

---

## Caveats / honest limitations

- **Single judge model.** EVAL_SPEC §5.4 originally suggested two judges (Claude Opus + GPT-5/4o) to cross-check. This experiment used `gpt-5.4` only. A second-judge run could be added if reviewers ask.
- **Synthetic conditions.** All 200 trials use the 8-condition grid from `condition_grid.py`, not real Magdeburg telemetry. Real-world FIWARE noise (missing fields, sensor faults) may produce different grounding behavior.
- **Top-recommended option only.** We judged the LLM's recommended option's reasoning, not all 4 mode justifications per trial. A wider sweep (≈800 reasonings) could reveal mode-specific patterns we're missing.
- **Strict judge.** The judge counts inferred-but-not-listed conditions as hallucinations (e.g. "icy paths" from snow + −5°C). This is conservative — a more lenient judge would push faithfulness >99%.

---

## What this gives the paper

Three trustworthy numbers for the §5.4 reliability table, plus an anecdote about the C8 parking misread that shows the kind of edge case ERA can still get wrong. All four reliability numbers — stability σ̄, score range, grounding %, faithfulness % — are now produced and reproducible from `evaluation/reliability/`.
