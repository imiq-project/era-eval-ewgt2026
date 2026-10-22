"""
Build the ERA sensitivity heatmap (Option A: compact 4-mode x 8-condition).

Output:
    evaluation/figures/era_heatmap_main.png   (300 DPI, for slides/MS Word)
    evaluation/figures/era_heatmap_main.pdf   (vector, for LaTeX)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Load data ----
ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "evaluation" / "era_sensitivity" / "trials.jsonl"
FIGDIR = ROOT / "evaluation" / "figures"
FIGDIR.mkdir(exist_ok=True)

with open(TRIALS, encoding="utf-8") as f:
    trials = [json.loads(l) for l in f if l.strip()]

# Latest successful trial per (qid, cid)
ok = {}
for t in trials:
    if not t.get("error"):
        ok[(t["query_id"], t["condition_id"])] = t

# Build mode -> condition -> [scores across queries]
modes = ["walking", "cycling", "transit", "driving"]
conditions = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# Short descriptors for x-axis — keep concise, full descriptions go in caption
COND_LABEL = {
    "C1": ("C1", "clear\nnoon"),
    "C2": ("C2", "clear\n22:00"),
    "C3": ("C3", "light rain\n+ park/traffic"),
    "C4": ("C4", "light rain\n+ 22:00/AQ"),
    "C5": ("C5", "heavy rain\n+ parking"),
    "C6": ("C6", "heavy rain\n+ 22:00/traffic"),
    "C7": ("C7", "snow\n+ parking"),
    "C8": ("C8", "snow\n+ 22:00/traffic"),
}

# ---- Compute mean scores ----
matrix = np.zeros((len(modes), len(conditions)))
sd_matrix = np.zeros((len(modes), len(conditions)))
for i, mode in enumerate(modes):
    for j, cid in enumerate(conditions):
        scores = []
        for (qid, c), t in ok.items():
            if c != cid:
                continue
            for opt in t["era_output"]["options"]:
                if opt["mode"] == mode:
                    scores.append(opt["score"])
                    break
        matrix[i, j] = np.mean(scores) if scores else np.nan
        sd_matrix[i, j] = np.std(scores) if len(scores) > 1 else 0.0

print("Score matrix (rows = modes, cols = conditions):")
print(f"  {' '*9} " + "  ".join(f"{c:>6s}" for c in conditions))
for i, mode in enumerate(modes):
    print(f"  {mode:9s} " + "  ".join(f"{matrix[i,j]:6.2f}" for j in range(len(conditions))))

# ---- Build heatmap ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 100,
})

fig, ax = plt.subplots(figsize=(12.5, 4.5))

# Colormap: RdYlGn (red = bad score, green = good score) — intuitive for "quality"
cmap = plt.get_cmap("RdYlGn")
im = ax.imshow(matrix, cmap=cmap, vmin=1, vmax=10, aspect="auto")

# Annotate each cell with its value
for i in range(len(modes)):
    for j in range(len(conditions)):
        val = matrix[i, j]
        # Pick text color based on cell brightness for readability
        cell_color = cmap((val - 1) / 9.0)
        luminance = 0.299*cell_color[0] + 0.587*cell_color[1] + 0.114*cell_color[2]
        text_color = "black" if luminance > 0.55 else "white"
        ax.text(j, i, f"{val:.2f}",
                ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold")

# X tick labels: two-line (ID + brief descriptor)
ax.set_xticks(np.arange(len(conditions)))
xtick_labels = [f"{COND_LABEL[c][0]}\n{COND_LABEL[c][1]}" for c in conditions]
ax.set_xticklabels(xtick_labels, fontsize=9, linespacing=1.4)
ax.tick_params(axis="x", which="both", length=0, pad=6)

# Y tick labels
ax.set_yticks(np.arange(len(modes)))
ax.set_yticklabels([m.capitalize() for m in modes], fontsize=11, fontweight="bold")
ax.tick_params(axis="y", which="both", length=0)

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(modes), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[1, 2.5, 5, 7.5, 10])
cbar.ax.set_yticklabels(["1\nstrongly\ninadvisable", "2.5", "5\nacceptable", "7.5", "10\nexcellent"], fontsize=8)
cbar.outline.set_visible(False)
cbar.ax.tick_params(length=0)

# Title and axis labels
fig.suptitle(
    "ERA Score Heatmap — Mean across 25 queries × 4 transport modes × 8 conditions",
    fontsize=12, fontweight="bold", y=0.99, x=0.45,
)
ax.set_xlabel("Environmental Condition", fontsize=10, labelpad=12)
ax.set_ylabel("Transport Mode", fontsize=10, labelpad=4)

# Caption with full condition definitions, below the figure
caption = (
    "Conditions: C1 clear+noon | C2 clear+22:00+congested+scarce parking | "
    "C3 light rain (2mm)+5°C+noon+congested+scarce parking | "
    "C4 light rain+20°C+22:00+poor AQ | C5 heavy rain (10mm)+5°C+noon+poor AQ+scarce parking | "
    "C6 heavy rain+20°C+22:00+congested | C7 snow (5cm)+−5°C+noon+scarce parking | "
    "C8 snow+22:00+poor AQ+congested. Each cell: mean ERA score (1–10) across n=25 queries."
)
fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8,
         color="#444444", wrap=True, style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Save in both PNG (300 DPI) and PDF (vector)
out_png = FIGDIR / "era_heatmap_main.png"
out_pdf = FIGDIR / "era_heatmap_main.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")

print(f"\nWrote: {out_png}")
print(f"Wrote: {out_pdf}")
plt.close(fig)
