"""
Ablation Study — Model Comparison Visualizer
=============================================
Reads the 5 per-model CSVs from results/ablation/ and generates:
  1. Val Dice curves (all 5 models on one chart)
  2. Val IoU curves  (all 5 models on one chart)
  3. Train vs Val Dice overlay (all 5 models — overfitting check)
  4. Best-epoch bar chart (final ranking with annotations)

Output: results/ablation/ablation_comparison.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
#  1. Config
# ──────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results" / "ablation"
OUTPUT_FILE = RESULTS_DIR / "ablation_comparison.png"

# Map: display name → CSV filename stem
MODELS = {
    "Model 1\n2-Block CNN":       "ablation_Model_1_2-Block",
    "Model 2\n3-Block CNN":       "ablation_Model_2_3-Block",
    "Model 3A\nFeature Extract":  "ablation_Model_3A_Feature_Extract",
    "Model 3B\nPartial Tune":     "ablation_Model_3B_Partial_Tune",
    "Model 3C\nFull Fine-Tune":   "ablation_Model_3C_Full_Fine-Tune",
}

# Curated color palette (ordered by expected performance)
COLORS = ["#94A3B8", "#64748B", "#38BDF8", "#818CF8", "#F472B6"]
MARKERS = ["o", "s", "^", "D", "*"]

# ──────────────────────────────────────────────
#  2. Load Data
# ──────────────────────────────────────────────
dfs = {}
for label, stem in MODELS.items():
    path = RESULTS_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path).dropna()
    dfs[label] = df
    print(f"Loaded '{stem}' — {len(df)} epochs")

# ──────────────────────────────────────────────
#  3. Figure Setup
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0F172A",
    "axes.facecolor":    "#1E293B",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#CBD5E1",
    "axes.titlecolor":   "#F1F5F9",
    "axes.grid":         True,
    "grid.color":        "#334155",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8",
    "text.color":        "#F1F5F9",
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  "#1E293B",
    "legend.edgecolor":  "#334155",
    "legend.labelcolor": "#CBD5E1",
})

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    "Ablation Study — Encoder Architecture Comparison\nFedMedSeg · RSNA Pneumonia Segmentation",
    fontsize=18, fontweight="bold", color="#F1F5F9", y=0.98
)

gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                       left=0.07, right=0.97, top=0.92, bottom=0.08)

ax_dice  = fig.add_subplot(gs[0, 0])   # Val Dice curves
ax_iou   = fig.add_subplot(gs[0, 1])   # Val IoU curves
ax_train = fig.add_subplot(gs[1, 0])   # Train vs Val Dice
ax_bar   = fig.add_subplot(gs[1, 1])   # Best-epoch ranking bar chart

# ──────────────────────────────────────────────
#  4. Plot 1 — Validation Dice Curves
# ──────────────────────────────────────────────
ax_dice.set_title("Validation Dice Score per Epoch", fontsize=13, pad=10)
ax_dice.set_xlabel("Epoch")
ax_dice.set_ylabel("Dice Score")

for (label, df), color, marker in zip(dfs.items(), COLORS, MARKERS):
    epochs = df["epoch"].values
    ax_dice.plot(epochs, df["val_dice"].values,
                 color=color, marker=marker, markersize=5,
                 linewidth=2, label=label.replace("\n", " "),
                 alpha=0.9)
    # Mark peak
    best_idx = df["val_dice"].idxmax()
    ax_dice.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, "val_dice"],
                    color=color, s=120, zorder=5, edgecolors="white", linewidths=1.2)

ax_dice.axhline(0.65, color="#FBBF24", linewidth=1.2, linestyle=":", alpha=0.8,
                label="Target (0.65)")
ax_dice.legend(fontsize=8, loc="lower right")
ax_dice.set_ylim(0.15, 0.75)

# ──────────────────────────────────────────────
#  5. Plot 2 — Validation IoU Curves
# ──────────────────────────────────────────────
ax_iou.set_title("Validation IoU Score per Epoch", fontsize=13, pad=10)
ax_iou.set_xlabel("Epoch")
ax_iou.set_ylabel("IoU Score")

for (label, df), color, marker in zip(dfs.items(), COLORS, MARKERS):
    epochs = df["epoch"].values
    ax_iou.plot(epochs, df["val_iou"].values,
                color=color, marker=marker, markersize=5,
                linewidth=2, label=label.replace("\n", " "),
                alpha=0.9)
    best_idx = df["val_iou"].idxmax()
    ax_iou.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, "val_iou"],
                   color=color, s=120, zorder=5, edgecolors="white", linewidths=1.2)

ax_iou.legend(fontsize=8, loc="lower right")
ax_iou.set_ylim(0.10, 0.65)

# ──────────────────────────────────────────────
#  6. Plot 3 — Train vs Val Dice (Overfitting Check)
# ──────────────────────────────────────────────
ax_train.set_title("Train vs Val Dice — Overfitting Check", fontsize=13, pad=10)
ax_train.set_xlabel("Epoch")
ax_train.set_ylabel("Dice Score")

for (label, df), color, marker in zip(dfs.items(), COLORS, MARKERS):
    epochs = df["epoch"].values
    short = label.replace("\n", " ")
    ax_train.plot(epochs, df["val_dice"].values,
                  color=color, linewidth=2, label=f"{short} (val)",
                  alpha=0.9)
    ax_train.plot(epochs, df["train_dice"].values,
                  color=color, linewidth=1.2, linestyle="--",
                  alpha=0.5)

# Legend entries for solid/dashed distinction
solid_patch = mpatches.Patch(color="#94A3B8", label="── Val Dice")
dash_patch  = mpatches.Patch(color="#94A3B8", label="╌╌ Train Dice (dashed)")
handles, labels_leg = ax_train.get_legend_handles_labels()
ax_train.legend(handles=handles + [solid_patch, dash_patch],
                labels=labels_leg + ["── Val Dice", "╌╌ Train Dice"],
                fontsize=7.5, loc="lower right")
ax_train.set_ylim(0.00, 0.80)

# ──────────────────────────────────────────────
#  7. Plot 4 — Best Dice Bar Chart (Final Ranking)
# ──────────────────────────────────────────────
ax_bar.set_title("Best Validation Dice — Final Ranking", fontsize=13, pad=10)
ax_bar.set_xlabel("Best Val Dice Score")

short_labels = [lbl.replace("\n", " ") for lbl in dfs.keys()]
best_dices   = [df["val_dice"].max() for df in dfs.values()]
best_ious    = [df["val_iou"].max() for df in dfs.values()]

y_pos = np.arange(len(short_labels))
bar_height = 0.38

bars_dice = ax_bar.barh(y_pos + bar_height / 2, best_dices,
                         height=bar_height, color=COLORS,
                         label="Best Val Dice", alpha=0.90)
bars_iou  = ax_bar.barh(y_pos - bar_height / 2, best_ious,
                         height=bar_height, color=COLORS,
                         label="Best Val IoU", alpha=0.50)

# Value annotations
for bar, val in zip(bars_dice, best_dices):
    ax_bar.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=9, color="#F1F5F9", fontweight="bold")

for bar, val in zip(bars_iou, best_ious):
    ax_bar.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=9, color="#94A3B8")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(short_labels, fontsize=9)
ax_bar.set_xlim(0.0, 0.78)
ax_bar.axvline(0.65, color="#FBBF24", linewidth=1.2, linestyle=":",
               alpha=0.8, label="Target (0.65)")

# Highlight winner
winner_idx = int(np.argmax(best_dices))
ax_bar.get_yticklabels()[winner_idx].set_color("#F472B6")
ax_bar.get_yticklabels()[winner_idx].set_fontweight("bold")
ax_bar.annotate("🏆 WINNER", xy=(best_dices[winner_idx], y_pos[winner_idx] + bar_height / 2),
                xytext=(best_dices[winner_idx] + 0.04, y_pos[winner_idx] + 0.35),
                color="#F472B6", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#F472B6", lw=1.5))

dice_patch = mpatches.Patch(color="#CBD5E1", alpha=0.9, label="Best Val Dice")
iou_patch  = mpatches.Patch(color="#CBD5E1", alpha=0.5, label="Best Val IoU")
target_line = plt.Line2D([0], [0], color="#FBBF24", linewidth=1.5,
                          linestyle=":", label="Target (0.65)")
ax_bar.legend(handles=[dice_patch, iou_patch, target_line],
              fontsize=9, loc="lower right")

# ──────────────────────────────────────────────
#  8. Save
# ──────────────────────────────────────────────
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅  Saved → {OUTPUT_FILE}")

# ──────────────────────────────────────────────
#  9. Print Summary Table
# ──────────────────────────────────────────────
print("\n" + "=" * 58)
print("  ABLATION STUDY — FINAL SUMMARY TABLE")
print("=" * 58)
print(f"  {'Model':<30} {'Best Dice':>10} {'Best IoU':>10}")
print("-" * 58)
for (label, df), bd, bi in zip(dfs.items(), best_dices, best_ious):
    short = label.replace("\n", " ")
    tag = " ← WINNER" if bd == max(best_dices) else ""
    print(f"  {short:<30} {bd:>10.4f} {bi:>10.4f}{tag}")
print("=" * 58)
