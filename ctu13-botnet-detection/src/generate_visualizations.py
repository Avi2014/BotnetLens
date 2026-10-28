"""
src/generate_visualizations.py
Generates comparison charts for Phase 1 and Phase 2 results.

Reads the results CSVs produced by training scripts and outputs:
  - Binary algorithm comparison bar charts
  - Multiclass algorithm comparison bar charts
  - ROC comparison placeholder
  - Family-wise recall heatmap
  - Timing comparison

Usage:
    python src/generate_visualizations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import PROJECT_ROOT
from logging_utils import get_logger

logger = get_logger("generate_visualizations")

REPORTS  = PROJECT_ROOT / "reports"
FIGURES  = REPORTS / "figures"
P1_DIR   = REPORTS / "phase1"
P2_DIR   = REPORTS / "phase2"

PALETTE = sns.color_palette("tab10")


def _bar_comparison(df: pd.DataFrame, metrics: list[str], title: str,
                    out_path: Path, ylabel: str = "Score") -> None:
    """Bar chart comparing algorithms across multiple metrics."""
    df = df.set_index("algorithm")
    avail = [m for m in metrics if m in df.columns]
    if not avail:
        logger.warning("No metrics available for chart: %s", title)
        return
    x = np.arange(len(df))
    width = 0.8 / len(avail)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 2), 6))
    for i, metric in enumerate(avail):
        bars = ax.bar(x + i * width, df[metric], width, label=metric.replace("_", " ").title(),
                      alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(avail) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title,   fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved → %s", out_path)


def _timing_chart(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if "train_time_s" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["algorithm"], df["train_time_s"], color=PALETTE[:len(df)], alpha=0.85, edgecolor="black")
    ax.set_xlabel("Training Time (seconds)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Timing chart → %s", out_path)


def generate_phase1_charts() -> None:
    bin_csv = P1_DIR / "binary_results_all_algorithms.csv"
    mc_csv  = P1_DIR / "multiclass_results_all_algorithms.csv"

    if bin_csv.exists():
        df = pd.read_csv(bin_csv)
        _bar_comparison(df,
            ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"],
            "Binary Classification — Algorithm Comparison",
            FIGURES / "phase1_binary_comparison.png")
        _timing_chart(df, "Binary Classification — Training Time",
                      FIGURES / "phase1_binary_timing.png")
        logger.info("Phase 1 binary charts generated.")
    else:
        logger.warning("Binary results CSV not found: %s", bin_csv)

    if mc_csv.exists():
        df = pd.read_csv(mc_csv)
        _bar_comparison(df,
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
            "Multiclass Classification — Algorithm Comparison",
            FIGURES / "phase1_multiclass_comparison.png")
        _timing_chart(df, "Multiclass Classification — Training Time",
                      FIGURES / "phase1_multiclass_timing.png")
        logger.info("Phase 1 multiclass charts generated.")
    else:
        logger.warning("Multiclass results CSV not found: %s", mc_csv)


def generate_phase2_charts() -> None:
    anom_csv = P2_DIR / "anomaly_results_all_algorithms.csv"
    if not anom_csv.exists():
        logger.warning("Anomaly results CSV not found: %s", anom_csv)
        return

    df = pd.read_csv(anom_csv)
    _bar_comparison(df,
        ["fpr_on_benign", "tpr_on_known_botnet", "tpr_on_holdout"],
        "Phase 2 — Anomaly Detection Comparison",
        FIGURES / "phase2_anomaly_comparison.png",
        ylabel="Rate")
    _timing_chart(df, "Anomaly Detection — Training Time",
                  FIGURES / "phase2_anomaly_timing.png")

    # Score distribution overlay (if PNG exists in score_distributions)
    score_pngs = list((P2_DIR / "score_distributions").glob("*.png"))
    logger.info("Phase 2 charts generated. Score dist plots: %d", len(score_pngs))


def generate_family_heatmap() -> None:
    sig_csv = REPORTS / "family_analysis" / "family_feature_signatures.csv"
    if not sig_csv.exists():
        logger.warning("Family signatures CSV not found.")
        return

    df = pd.read_csv(sig_csv)
    pivot = df.pivot_table(index="family", columns="feature",
                           values="distinctiveness", aggfunc="max").fillna(0)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Family Feature Distinctiveness Heatmap", fontweight="bold", fontsize=13)
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Family",  fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    out = REPORTS / "family_analysis" / "family_feature_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Family heatmap → %s", out)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_phase1_charts()
    generate_phase2_charts()
    generate_family_heatmap()
    logger.info("All visualizations generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
