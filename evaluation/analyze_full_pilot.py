"""
Full §5.2 sensitivity analysis on the 200-trial dataset.
Computes: heatmap, paired Δ, Spearman ρ per dimension, monotonicity, bootstrap CIs.
Run: python -X utf8 evaluation/analyze_full_pilot.py
"""

import json
import math
import random
from collections import defaultdict
from statistics import mean, stdev, median


with open('evaluation/era_sensitivity/trials.jsonl', encoding='utf-8') as f:
    trials = [json.loads(l) for l in f if l.strip()]

ok_by_pair = {}
for t in trials:
    if not t.get('error'):
        ok_by_pair[(t['query_id'], t['condition_id'])] = t

data = defaultdict(dict)
for (qid, cid), t in ok_by_pair.items():
    for opt in t['era_output']['options']:
        data[(qid, cid)][opt['mode']] = opt['score']

queries = sorted({qid for qid, _ in ok_by_pair})
conditions = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

COND_NUM = {
    'C1': {'rain_mm':  0, 'snow_cm': 0, 'temp_c':  20, 'is_dark': 0, 'pm25': 10, 'traffic': 0,  'parking': 80},
    'C2': {'rain_mm':  0, 'snow_cm': 0, 'temp_c':   5, 'is_dark': 1, 'pm25': 10, 'traffic': 12, 'parking':  3},
    'C3': {'rain_mm':  2, 'snow_cm': 0, 'temp_c':   5, 'is_dark': 0, 'pm25': 10, 'traffic': 12, 'parking':  3},
    'C4': {'rain_mm':  2, 'snow_cm': 0, 'temp_c':  20, 'is_dark': 1, 'pm25': 60, 'traffic': 0,  'parking': 80},
    'C5': {'rain_mm': 10, 'snow_cm': 0, 'temp_c':   5, 'is_dark': 0, 'pm25': 60, 'traffic': 0,  'parking':  3},
    'C6': {'rain_mm': 10, 'snow_cm': 0, 'temp_c':  20, 'is_dark': 1, 'pm25': 10, 'traffic': 12, 'parking': 80},
    'C7': {'rain_mm':  0, 'snow_cm': 5, 'temp_c':  -5, 'is_dark': 0, 'pm25': 10, 'traffic': 0,  'parking':  3},
    'C8': {'rain_mm':  0, 'snow_cm': 5, 'temp_c':  -5, 'is_dark': 1, 'pm25': 60, 'traffic': 12, 'parking': 80},
}


def spearman(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2:
        return 0.0
    n = len(xs)
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0]*n
    rank_y = [0]*n
    for r, i in enumerate(rx):
        rank_x[i] = r
    for r, i in enumerate(ry):
        rank_y[i] = r
    d2 = sum((rank_x[i] - rank_y[i])**2 for i in range(n))
    return 1 - (6*d2)/(n*(n*n - 1))


def paired_t(values):
    if len(values) < 2:
        return 0.0
    sd = stdev(values)
    if sd == 0:
        return float('inf')
    return mean(values) / (sd / math.sqrt(len(values)))


def bootstrap_ci(values, n_iter=10000, alpha=0.05, seed=42):
    random.seed(seed)
    n = len(values)
    boots = sorted(mean(random.choice(values) for _ in range(n)) for _ in range(n_iter))
    return boots[int(alpha/2 * n_iter)], boots[int((1 - alpha/2) * n_iter)]


total_obs = sum(1 for q in queries for c in conditions if (q, c) in data)
print("=" * 78)
print(f"FULL SECTION 5.2 ANALYSIS — n={len(queries)} queries x {len(conditions)} conditions = {total_obs} obs")
print("=" * 78)

# 1. HEATMAP
print("\n=== 1. SCORE HEATMAP (mean +/- SD across 25 queries) ===")
print(f"{'Mode':9s}  " + "  ".join(f"{c:>10s}" for c in conditions))
for mode in ['walking', 'cycling', 'transit', 'driving']:
    row = [f"{mode:9s} "]
    for cid in conditions:
        scores = [data[(q, cid)][mode] for q in queries if mode in data.get((q, cid), {})]
        if len(scores) > 1:
            row.append(f"{mean(scores):4.2f}+-{stdev(scores):.2f}")
        elif scores:
            row.append(f"{scores[0]:.2f}      ")
        else:
            row.append("    -      ")
    print("  ".join(row))

# 2. PAIRED DELTA from C1
print("\n=== 2. PAIRED DELTA from C1 baseline (mean change) ===")
print(f"{'Mode':9s}  " + "  ".join(f"{'D '+c:>10s}" for c in conditions[1:]))
for mode in ['walking', 'cycling', 'transit', 'driving']:
    row = [f"{mode:9s} "]
    for cid in conditions[1:]:
        deltas = [data[(q, cid)][mode] - data[(q, 'C1')][mode] for q in queries
                  if mode in data.get((q, 'C1'), {}) and mode in data.get((q, cid), {})]
        if deltas:
            row.append(f"{mean(deltas):+5.2f}    ")
    print("  ".join(row))

# 3. SPEARMAN per dimension per mode
print("\n=== 3. SPEARMAN rho score vs each environmental dimension ===")
print("  Higher rho|dim| value should reduce score for outdoor modes (negative rho).")
print("  Driving should be near-zero. Parking exception: +rho means more spots = better drive score.")
print()
print(f"{'Mode':9s}  {'rain_mm':>9s}  {'snow_cm':>9s}  {'PM2.5':>9s}  {'is_dark':>9s}  {'traffic':>9s}  {'parking':>9s}")
for mode in ['walking', 'cycling', 'transit', 'driving']:
    row = [f"{mode:9s} "]
    for dim in ['rain_mm', 'snow_cm', 'pm25', 'is_dark', 'traffic', 'parking']:
        xs, ys = [], []
        for q in queries:
            for c in conditions:
                if (q, c) in data and mode in data[(q, c)]:
                    xs.append(COND_NUM[c][dim])
                    ys.append(data[(q, c)][mode])
        rho = spearman(xs, ys)
        row.append(f"{rho:+.3f}    ")
    print("  ".join(row))

# 4. MONOTONICITY by mode
print("\n=== 4. MONOTONICITY — % of (query, mode) pairs moving in expected direction ===")
print("  Expected directions per condition (vs C1 baseline):")
expected = {
    'C2': {'walking':'down','cycling':'down','transit':'flat','driving':'flat'},
    'C3': {'walking':'down','cycling':'down','transit':'down','driving':'flat'},
    'C4': {'walking':'down','cycling':'down','transit':'down','driving':'flat'},
    'C5': {'walking':'down','cycling':'down','transit':'down','driving':'flat'},
    'C6': {'walking':'down','cycling':'down','transit':'down','driving':'down'},
    'C7': {'walking':'down','cycling':'down','transit':'down','driving':'down'},
    'C8': {'walking':'down','cycling':'down','transit':'down','driving':'flat'},
}
total_pass = total_n = 0
per_mode = defaultdict(lambda: [0, 0])
for cid_adv, mode_dirs in expected.items():
    for q in queries:
        if (q, 'C1') not in data or (q, cid_adv) not in data:
            continue
        for mode, expected_dir in mode_dirs.items():
            if mode not in data[(q, 'C1')] or mode not in data[(q, cid_adv)]:
                continue
            d = data[(q, cid_adv)][mode] - data[(q, 'C1')][mode]
            total_n += 1
            per_mode[mode][1] += 1
            ok = (expected_dir == 'down' and d <= 0) or (expected_dir == 'flat' and abs(d) <= 1)
            if ok:
                total_pass += 1
                per_mode[mode][0] += 1

print(f"  Overall: {total_pass}/{total_n} = {100*total_pass/total_n:.1f}%")
for mode in ['walking', 'cycling', 'transit', 'driving']:
    p, t = per_mode[mode]
    print(f"  {mode:10s} {p}/{t} = {100*p/t:.1f}%")

# 5. HEADLINE — walk vs drive under heavy rain
print("\n=== 5. HEADLINE: walk_delta - drive_delta under heavy rain (C1 -> C5) ===")
walk_d = [data[(q, 'C5')]['walking'] - data[(q, 'C1')]['walking'] for q in queries
          if 'walking' in data.get((q, 'C5'), {}) and 'walking' in data.get((q, 'C1'), {})]
drive_d = [data[(q, 'C5')]['driving'] - data[(q, 'C1')]['driving'] for q in queries
           if 'driving' in data.get((q, 'C5'), {}) and 'driving' in data.get((q, 'C1'), {})]
n_pairs = min(len(walk_d), len(drive_d))
asym = [w - d for w, d in zip(walk_d[:n_pairs], drive_d[:n_pairs])]
sd = stdev(asym) if len(asym) > 1 else 0
t_stat = paired_t(asym)
ci_low, ci_high = bootstrap_ci(asym)
print(f"  walking delta:   mean {mean(walk_d):+.2f}  range [{min(walk_d)},{max(walk_d)}]  n={len(walk_d)}")
print(f"  driving delta:   mean {mean(drive_d):+.2f}  range [{min(drive_d)},{max(drive_d)}]  n={len(drive_d)}")
print(f"  asymmetry:       {mean(asym):+.2f} +- {sd:.2f}")
print(f"  95% bootstrap CI: [{ci_low:+.2f}, {ci_high:+.2f}]")
print(f"  paired t:        t({len(asym)-1}) = {t_stat:.2f}")
print(f"  CI excludes 0:   {'YES (significant)' if ci_high < 0 or ci_low > 0 else 'NO'}")

# 6. RECOMMENDATION DISTRIBUTION
print("\n=== 6. RECOMMENDED-MODE DISTRIBUTION by condition ===")


def rec_mode(t):
    out = t.get('era_output', {})
    rec_idx = out.get('recommended_index')
    for opt in out.get('options', []):
        if opt['route_index'] == rec_idx:
            return opt['mode']
    return '?'


print(f"  {'condition':9s}  {'walking':>8s} {'cycling':>8s} {'transit':>8s} {'driving':>8s}")
for cid in conditions:
    counts = defaultdict(int)
    for q in queries:
        if (q, cid) in ok_by_pair:
            counts[rec_mode(ok_by_pair[(q, cid)])] += 1
    print(f"  {cid:9s}  {counts['walking']:>8d} {counts['cycling']:>8d} {counts['transit']:>8d} {counts['driving']:>8d}")
