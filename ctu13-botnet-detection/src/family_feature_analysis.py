"""
src/family_feature_analysis.py
Per-family feature signature analysis.

For each botnet family:
  - Compares feature distributions vs all other classes
  - Computes distinctiveness scores (median difference / IQR)
  - Ranks features by discriminative power
  - Saves per-family profiles and boxplots

Usage:
    python src/family_feature_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import PROJECT_ROOT, PROCESSED_DIR, UBF_FEATURES
from logging_utils import get_logger
from feature_engineering import extract_features
from ubf_features import get_ubf

logger = get_logger("family_feature_analysis")

REPORTS_DIR = PROJECT_ROOT / "reports" / "family_analysis"
EPSILON = 1e-9


def _distinctiveness(family_vals: pd.Series, other_vals: pd.Series) -> float:
    """Distinctiveness = |median(family) - median(other)| / max(IQR(other), ε)"""
    med_diff = abs(family_vals.median() - other_vals.median())
    iqr      = float(stats.iqr(other_vals.dropna()))
    return float(med_diff / max(iqr, EPSILON))


def analyse_families(df: pd.DataFrame) -> dict:
    """Compute per-family feature signatures."""
    feats = extract_features(df)
    ubf   = get_ubf(feats)
    ubf["family"] = df["family"].values

    families    = [f for f in ubf["family"].unique() if f != "Benign"]
    signatures  = []
    top_features = []

    for family in sorted(families):
        fam_mask   = ubf["family"] == family
        other_mask = ~fam_mask
        fam_df     = ubf[fam_mask]
        oth_df     = ubf[other_mask]

        logger.info("Analysing family: %s  (n=%d)", family, len(fam_df))

        rows = []
        for feat in UBF_FEATURES:
            fam_vals = fam_df[feat].dropna()
            oth_vals = oth_df[feat].dropna()
            if len(fam_vals) == 0:
                continue

            disc  = _distinctiveness(fam_vals, oth_vals)
            direction = "higher" if fam_vals.median() > oth_vals.median() else "lower"

            try:
                kw_stat, kw_p = stats.kruskal(fam_vals, oth_vals)
            except Exception:
                kw_stat, kw_p = 0.0, 1.0

            rows.append({
                "family":       family,
                "feature":      feat,
                "distinctiveness": round(disc, 4),
                "family_median":   round(float(fam_vals.median()), 6),
                "other_median":    round(float(oth_vals.median()), 6),
                "family_iqr":      round(float(stats.iqr(fam_vals)), 6),
                "other_iqr":       round(float(stats.iqr(oth_vals)), 6),
                "kw_stat":         round(kw_stat, 4),
                "kw_pvalue":       round(kw_p, 6),
                "direction":       direction,
            })

        rows = sorted(rows, key=lambda r: r["distinctiveness"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        signatures.extend(rows)

        # Top 5 features per family
        top_features.append({
            "family":       family,
            "n_samples":    len(fam_df),
            "top_features": [r["feature"] for r in rows[:5]],
            "top_scores":   [r["distinctiveness"] for r in rows[:5]],
        })

    return {"signatures": signatures, "top_features": top_features}


def save_reports(analysis: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_df = pd.DataFrame(analysis["signatures"])
    sig_df.to_csv(out_dir / "family_feature_signatures.csv", index=False)

    top_df = pd.DataFrame(analysis["top_features"])
    top_df.to_csv(out_dir / "family_top_features.csv", index=False)

    # Markdown profile
    families = sig_df["family"].unique()
    lines = ["# Family Feature Profiles", ""]
    for fam in sorted(families):
        fam_rows = sig_df[sig_df["family"] == fam].head(7)
        lines += [f"## {fam}", "", "| Rank | Feature | Distinctiveness | Family Median | Other Median | Direction |",
                  "|---|---|---|---|---|---|"]
        for _, r in fam_rows.iterrows():
            lines.append(
                f"| {int(r['rank'])} | `{r['feature']}` | {r['distinctiveness']:.4f} | "
                f"{r['family_median']:.4f} | {r['other_median']:.4f} | {r['direction']} |"
            )
        lines += ["", "---", ""]

    (out_dir / "family_feature_profiles.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Family profiles → %s", out_dir / "family_feature_profiles.md")


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot per-family boxplots for top features."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        feats = extract_features(df)
        ubf   = get_ubf(feats)
        ubf["family"] = df["family"].values

        top_plot_features = ["duration", "bytes_per_second", "packets_per_second",
                             "bytes_ratio", "mean_packet_size"]

        for feat in top_plot_features:
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_df = ubf[["family", feat]].dropna()
            plot_df = plot_df[plot_df[feat] < plot_df[feat].quantile(0.99)]  # remove extreme outliers for viz
            order = plot_df.groupby("family")[feat].median().sort_values(ascending=False).index
            sns.boxplot(data=plot_df, x="family", y=feat, order=order, ax=ax,
                        hue="family", palette="tab10", legend=False, showfliers=False)
            ax.set_xlabel("Family",  fontweight="bold")
            ax.set_ylabel(feat,      fontweight="bold")
            ax.set_title(f"Feature: {feat} — per botnet family", fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / f"boxplot_{feat}.png", dpi=150, bbox_inches="tight")
            plt.close()
        logger.info("Boxplots saved → %s", out_dir)
    except Exception as exc:
        logger.warning("Boxplot generation failed: %s", exc)


CHUNKSIZE = 500_000
_CSV_DTYPES = {
    "binary_label": "int8", "scenario_id": "int16",
    "total_packets": "float32", "total_bytes": "float32",
    "src_bytes": "float32", "dst_bytes": "float32",
    "duration": "float32", "src_port": "float32",
    "dst_port": "float32",
}


def _load_analysis_sample(processed_path: Path, max_benign_per_chunk: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Stream CSV in chunks: retain all botnet flows plus a stratified benign sample to fit in RAM."""
    logger.info("Loading chunked sample from %s …", processed_path)
    chunks = []
    for i, chunk in enumerate(pd.read_csv(processed_path, chunksize=CHUNKSIZE, dtype=_CSV_DTYPES, low_memory=False)):
        mal = chunk[chunk["binary_label"] == 1]
        ben = chunk[chunk["binary_label"] == 0]
        if len(ben) > max_benign_per_chunk:
            ben = ben.sample(n=max_benign_per_chunk, random_state=seed + i)
        if len(mal) > 0 or len(ben) > 0:
            chunks.append(pd.concat([mal, ben], ignore_index=True))

    combined = pd.concat(chunks, ignore_index=True)
    logger.info("Sampled dataframe for family analysis: %d rows (%d malware, %d benign)",
                len(combined), int((combined["binary_label"] == 1).sum()), int((combined["binary_label"] == 0).sum()))
    return combined


def main() -> int:
    processed_path = PROCESSED_DIR / "ctu13_flows.csv"
    if not processed_path.exists():
        logger.error("Processed data not found: %s. Run clean_ctu13.py first.", processed_path)
        return 1

    df = _load_analysis_sample(processed_path)
    analysis = analyse_families(df)
    save_reports(analysis, REPORTS_DIR)
    plot_boxplots(df, REPORTS_DIR / "family_feature_boxplots")
    logger.info("Family feature analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
