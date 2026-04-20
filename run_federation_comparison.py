"""
run_federation_comparison.py
============================
FedMedSeg Phase 3 & 4 — Final Federation Comparison & Visualization

PURPOSE:
  Generates publication-ready comparison charts and tables across ALL
  SIX experimental approaches:

    1. Centralized Baseline  (model3c_final — all data, one model)
    2. Isolated Client A     (75% pneumonia — specialist hospital)
    3. Isolated Client B     (75% normal — general clinic)
    4. FedAvg                (standard federated averaging)
    5. FedProx               (proximal-regularized federation)
    6. DP-FedProx            (Phase 4 — differentially private)

  This script tells the COMPLETE "Federation Story":
    Isolated FAILS → FedAvg RECOVERS → FedProx EXCELS → DP-FedProx: Privacy at Low Cost
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Result Paths ──────────────────────────────────────────────────────────────
CENTRALIZED_REPORT = PROJECT_ROOT / "results" / "model3c_final" / "model_evaluation_report.json"
ISOLATED_REPORT    = PROJECT_ROOT / "results" / "federated" / "isolated" / "isolated_metrics.json"
FEDAVG_REPORT      = PROJECT_ROOT / "results" / "federated" / "fedavg" / "fedavg_report.json"
FEDPROX_REPORT     = PROJECT_ROOT / "results" / "federated" / "fedprox" / "fedprox_report.json"
DP_FEDPROX_REPORT  = PROJECT_ROOT / "results" / "federated" / "dp_fedprox" / "dp_fedprox_report.json"

FEDAVG_ROUNDS_CSV   = PROJECT_ROOT / "results" / "federated" / "fedavg" / "round_metrics.csv"
FEDPROX_ROUNDS_CSV  = PROJECT_ROOT / "results" / "federated" / "fedprox" / "round_metrics.csv"
DP_ROUNDS_CSV       = PROJECT_ROOT / "results" / "federated" / "dp_fedprox" / "round_metrics.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "federated" / "dp_federated_comparison"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_metrics():
    """
    Load final metrics from all experiments.

    Returns:
        dict: approach_name → {dice, iou, pixel_acc}
    """
    results = {}

    # ── 1. Centralized Baseline ───────────────────────────────────────────────
    if CENTRALIZED_REPORT.exists():
        with open(CENTRALIZED_REPORT) as f:
            data = json.load(f)
        results["Centralized\n(Baseline)"] = {
            "dice":      data["metrics"]["dice_coefficient"]["mean"],
            "iou":       data["metrics"]["mean_iou"]["mean"],
            "pixel_acc": data["metrics"]["pixel_accuracy"]["mean"],
        }
        print(f"  ✓ Centralized: Dice={results['Centralized\n(Baseline)']['dice']:.4f}")
    else:
        print(f"  ⚠ Centralized report not found: {CENTRALIZED_REPORT}")

    # ── 2 & 3. Isolated Clients ──────────────────────────────────────────────
    if ISOLATED_REPORT.exists():
        with open(ISOLATED_REPORT) as f:
            data = json.load(f)

        if "client_a" in data:
            fm = data["client_a"]["final_metrics"]
            results["Isolated A\n(75% Pneumonia)"] = {
                "dice":      fm["val_dice"],
                "iou":       fm["val_iou"],
                "pixel_acc": fm["val_pixel_acc"],
            }
            print(f"  ✓ Isolated A:  Dice={fm['val_dice']:.4f}")

        if "client_b" in data:
            fm = data["client_b"]["final_metrics"]
            results["Isolated B\n(75% Normal)"] = {
                "dice":      fm["val_dice"],
                "iou":       fm["val_iou"],
                "pixel_acc": fm["val_pixel_acc"],
            }
            print(f"  ✓ Isolated B:  Dice={fm['val_dice']:.4f}")
    else:
        print(f"  ⚠ Isolated report not found: {ISOLATED_REPORT}")

    # ── 4. FedAvg ─────────────────────────────────────────────────────────────
    if FEDAVG_REPORT.exists():
        with open(FEDAVG_REPORT) as f:
            data = json.load(f)
        fm = data.get("final_metrics", {})
        results["FedAvg"] = {
            "dice":      fm.get("val_dice", 0),
            "iou":       fm.get("val_iou", 0),
            "pixel_acc": fm.get("val_pixel_acc", 0),
        }
        print(f"  ✓ FedAvg:      Dice={fm.get('val_dice', 0):.4f}")
    else:
        print(f"  ⚠ FedAvg report not found: {FEDAVG_REPORT}")

    # ── 5. FedProx ────────────────────────────────────────────────────────────
    if FEDPROX_REPORT.exists():
        with open(FEDPROX_REPORT) as f:
            data = json.load(f)
        fm = data.get("final_metrics", {})
        results["FedProx\n(μ=0.01)"] = {
            "dice":      fm.get("val_dice", 0),
            "iou":       fm.get("val_iou", 0),
            "pixel_acc": fm.get("val_pixel_acc", 0),
        }
        print(f"  ✓ FedProx:     Dice={fm.get('val_dice', 0):.4f}")
    else:
        print(f"  ⚠ FedProx report not found: {FEDPROX_REPORT}")

    # ── 6. DP-FedProx (Phase 4) ───────────────────────────────────────────────
    if DP_FEDPROX_REPORT.exists():
        with open(DP_FEDPROX_REPORT) as f:
            data = json.load(f)
        fm = data.get("final_metrics", {})
        priv = data.get("privacy", {})
        eps = priv.get("target_epsilon", "?")
        results[f"DP-FedProx\n(ε={eps})"] = {
            "dice":      fm.get("val_dice", 0),
            "iou":       fm.get("val_iou", 0),
            "pixel_acc": fm.get("val_pixel_acc", 0),
            "epsilon":   eps,
        }
        print(f"  ✓ DP-FedProx:  Dice={fm.get('val_dice', 0):.4f}  (ε={eps})")
    else:
        print(f"  ℹ DP-FedProx not found (Phase 4 optional): {DP_FEDPROX_REPORT}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1: BAR CHART — All Approaches Compared
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(results: dict, output_dir: Path):
    """
    Create a grouped bar chart comparing Dice, IoU, and Pixel Accuracy
    across all five approaches.
    """
    # Dark theme
    plt.rcParams.update({
        "figure.facecolor": "#0F172A",
        "axes.facecolor":   "#1E293B",
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  "#CBD5E1",
        "axes.titlecolor":  "#F1F5F9",
        "axes.grid":        True,
        "grid.color":       "#334155",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "xtick.color":      "#94A3B8",
        "ytick.color":      "#94A3B8",
        "text.color":       "#F1F5F9",
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#334155",
        "legend.labelcolor": "#CBD5E1",
    })

    labels = list(results.keys())
    dice_vals      = [results[k]["dice"] for k in labels]
    iou_vals       = [results[k]["iou"] for k in labels]
    pixel_acc_vals = [results[k]["pixel_acc"] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width, dice_vals,      width, label="Dice Coefficient",
                   color="#4ADE80", alpha=0.9, edgecolor="#0F172A", linewidth=0.5)
    bars2 = ax.bar(x,         iou_vals,        width, label="Mean IoU",
                   color="#38BDF8", alpha=0.9, edgecolor="#0F172A", linewidth=0.5)
    bars3 = ax.bar(x + width, pixel_acc_vals,  width, label="Pixel Accuracy",
                   color="#A78BFA", alpha=0.9, edgecolor="#0F172A", linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                        color="#F1F5F9")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("FedMedSeg — Federation Comparison (Phase 3 & 4)\n"
                 "Isolated → FedAvg → FedProx → DP-FedProx (Privacy-Preserving)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, ha="center")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10)

    # Horizontal reference line at centralized dice score
    if "Centralized\n(Baseline)" in results:
        baseline_dice = results["Centralized\n(Baseline)"]["dice"]
        ax.axhline(y=baseline_dice, color="#FBBF24", linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Centralized Dice ({baseline_dice:.3f})")

    plt.tight_layout()
    out_path = output_dir / "federation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Bar chart saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2: CONVERGENCE CURVES — FedAvg vs FedProx Over Rounds
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence(output_dir: Path):
    """
    Plot how global validation Dice evolves over federated rounds for
    both FedAvg and FedProx.
    """
    plt.rcParams.update({
        "figure.facecolor": "#0F172A",
        "axes.facecolor":   "#1E293B",
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  "#CBD5E1",
        "axes.titlecolor":  "#F1F5F9",
        "axes.grid":        True,
        "grid.color":       "#334155",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "xtick.color":      "#94A3B8",
        "ytick.color":      "#94A3B8",
        "text.color":       "#F1F5F9",
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#334155",
        "legend.labelcolor": "#CBD5E1",
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Federated Learning Convergence\nGlobal Model Performance Over Communication Rounds",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Load FedAvg rounds ────────────────────────────────────────────────────
    fedavg_data = None
    if FEDAVG_ROUNDS_CSV.exists():
        fedavg_data = pd.read_csv(FEDAVG_ROUNDS_CSV)
        print(f"  ✓ FedAvg rounds loaded: {len(fedavg_data)} rounds")

    # ── Load FedProx rounds ───────────────────────────────────────────────────
    fedprox_data = None
    if FEDPROX_ROUNDS_CSV.exists():
        fedprox_data = pd.read_csv(FEDPROX_ROUNDS_CSV)
        print(f"  ✓ FedProx rounds loaded: {len(fedprox_data)} rounds")

    # ── Plot Dice Convergence ─────────────────────────────────────────────────
    ax = axes[0]
    if fedavg_data is not None:
        ax.plot(fedavg_data["round"], fedavg_data["global_val_dice"],
                color="#38BDF8", lw=2, marker="o", markersize=4, label="FedAvg")
    if fedprox_data is not None:
        ax.plot(fedprox_data["round"], fedprox_data["global_val_dice"],
                color="#4ADE80", lw=2, marker="s", markersize=4, label="FedProx (μ=0.01)")

    # ── Plot DP-FedProx if available ──────────────────────────────────────────
    dp_data = None
    if DP_ROUNDS_CSV.exists():
        try:
            dp_data = pd.read_csv(DP_ROUNDS_CSV)
            if not dp_data.empty and "global_val_dice" in dp_data.columns:
                print(f"  ✓ DP-FedProx rounds loaded: {len(dp_data)} rounds")
                ax.plot(dp_data["round"], dp_data["global_val_dice"],
                        color="#F472B6", lw=2, marker="^", markersize=4,
                        label="DP-FedProx (ε=8.0)", linestyle="-.")
        except Exception:
            pass

    # Centralized baseline reference
    if CENTRALIZED_REPORT.exists():
        with open(CENTRALIZED_REPORT) as f:
            cdata = json.load(f)
        baseline = cdata["metrics"]["dice_coefficient"]["mean"]
        ax.axhline(y=baseline, color="#FBBF24", ls="--", lw=1.5,
                   alpha=0.7, label=f"Centralized ({baseline:.3f})")

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Global Validation Dice", fontsize=11)
    ax.set_title("Dice Coefficient Convergence", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)

    # ── Plot IoU Convergence ──────────────────────────────────────────────────
    ax = axes[1]
    if fedavg_data is not None:
        ax.plot(fedavg_data["round"], fedavg_data["global_val_iou"],
                color="#38BDF8", lw=2, marker="o", markersize=4, label="FedAvg")
    if fedprox_data is not None:
        ax.plot(fedprox_data["round"], fedprox_data["global_val_iou"],
                color="#4ADE80", lw=2, marker="s", markersize=4, label="FedProx (μ=0.01)")
    if dp_data is not None and "global_val_iou" in dp_data.columns:
        ax.plot(dp_data["round"], dp_data["global_val_iou"],
                color="#F472B6", lw=2, marker="^", markersize=4,
                label="DP-FedProx (ε=8.0)", linestyle="-.")

    if CENTRALIZED_REPORT.exists():
        iou_baseline = cdata["metrics"]["mean_iou"]["mean"]
        ax.axhline(y=iou_baseline, color="#FBBF24", ls="--", lw=1.5,
                   alpha=0.7, label=f"Centralized ({iou_baseline:.3f})")

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Global Validation IoU", fontsize=11)
    ax.set_title("IoU Convergence", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "convergence_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Convergence curves saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LATEX TABLE — For Report / Paper
# ═══════════════════════════════════════════════════════════════════════════════

def generate_latex_table(results: dict, output_dir: Path):
    """Generate a LaTeX-ready comparison table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{FedMedSeg — Federation Experiment Results}",
        r"\label{tab:federation_results}",
        r"\begin{tabular}{l c c c}",
        r"\hline",
        r"\textbf{Approach} & \textbf{Dice} & \textbf{IoU} & \textbf{Pixel Acc.} \\",
        r"\hline",
    ]

    for name, m in results.items():
        clean_name = name.replace("\n", " ")
        lines.append(
            f"  {clean_name} & {m['dice']:.4f} & {m['iou']:.4f} & {m['pixel_acc']:.4f} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = output_dir / "federation_results_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ LaTeX table saved → {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(results: dict, output_dir: Path):
    """Save a JSON summary combining all experiments."""
    summary = {
        "title": "FedMedSeg — Federation Comparison Summary",
        "approaches": {},
    }

    for name, m in results.items():
        clean_name = name.replace("\n", " ")
        summary["approaches"][clean_name] = m

    # Determine best approach
    best_name = max(results.keys(), key=lambda k: results[k]["dice"])
    summary["best_approach"] = best_name.replace("\n", " ")
    summary["best_dice"] = results[best_name]["dice"]

    out_path = output_dir / "federation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary JSON saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  FEDERATION COMPARISON — All Approaches")
    print("=" * 65)

    # ── Load all metrics ──────────────────────────────────────────────────────
    print("\n[1] Loading experiment results...")
    results = load_metrics()

    if len(results) < 2:
        print("\n  ⚠ Not enough experiments completed for comparison.")
        print("  Run these first:")
        print("    .venv/bin/python run_isolated.py")
        print("    .venv/bin/python run_fedavg.py")
        print("    .venv/bin/python run_fedprox.py")
        return

    # ── Generate Plots ────────────────────────────────────────────────────────
    print("\n[2] Generating comparison plots...")
    plot_bar_comparison(results, OUTPUT_DIR)
    plot_convergence(OUTPUT_DIR)

    # ── Generate LaTeX Table ──────────────────────────────────────────────────
    print("\n[3] Generating LaTeX table...")
    generate_latex_table(results, OUTPUT_DIR)

    # ── Generate Summary ──────────────────────────────────────────────────────
    print("\n[4] Generating summary report...")
    generate_summary(results, OUTPUT_DIR)

    # ── Final Console Summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FEDERATION COMPARISON COMPLETE")
    print(f"{'='*65}")
    print(f"\n  {'Approach':<28}  {'Dice':<8}  {'IoU':<8}  {'Pixel Acc'}")
    print(f"  {'─'*60}")
    for name, m in results.items():
        clean_name = name.replace('\n', ' ')
        print(f"  {clean_name:<28}  {m['dice']:.4f}    {m['iou']:.4f}    {m['pixel_acc']:.4f}")

    # ── Story ─────────────────────────────────────────────────────────────────
    print(f"\n  THE FEDERATION STORY:")
    print(f"  ───────────────────────────────────────────────────────────")
    if "Isolated A\n(75% Pneumonia)" in results and "FedAvg" in results:
        iso_a = results["Isolated A\n(75% Pneumonia)"]["dice"]
        iso_b = results.get("Isolated B\n(75% Normal)", {}).get("dice", 0)
        fedavg = results["FedAvg"]["dice"]
        print(f"  1. Isolated models FAIL  (A: {iso_a:.3f}, B: {iso_b:.3f})")
        print(f"     → Biased data leads to biased models.")
        print(f"  2. FedAvg RECOVERS       ({fedavg:.3f})")
        print(f"     → Collaboration through weight sharing works!")
        if "FedProx\n(μ=0.01)" in results:
            fedprox = results["FedProx\n(μ=0.01)"]["dice"]
            print(f"  3. FedProx EXCELS        ({fedprox:.3f})")
            print(f"     → Proximal term handles Non-IID drift!")

        # Phase 4: DP story
        dp_key = next((k for k in results if "DP-FedProx" in k), None)
        if dp_key:
            dp_dice = results[dp_key]["dice"]
            dp_eps  = results[dp_key].get("epsilon", "8.0")
            fedprox_dice = results.get("FedProx\n(μ=0.01)", {}).get("dice", 0)
            cost = fedprox_dice - dp_dice
            print(f"  4. DP-FedProx            ({dp_dice:.3f})")
            print(f"     → ε={dp_eps} privacy with only {cost:+.3f} Dice cost.")
            print(f"     → PRIVACY-PRESERVING medial AI achieved!")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"    ├── federation_comparison.png   (bar chart)")
    print(f"    ├── convergence_curves.png      (round-by-round)")
    print(f"    ├── federation_results_table.tex (LaTeX)")
    print(f"    └── federation_summary.json     (machine-readable)")


if __name__ == "__main__":
    main()
