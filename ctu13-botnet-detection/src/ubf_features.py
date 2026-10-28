"""
src/ubf_features.py
Unified Behavioral Feature (UBF) layer.

Ensures:
  - Same feature names at training and inference time
  - Same feature order always
  - Missing columns reported and zero-filled
  - UBF schema saved to JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import UBF_FEATURES, PROJECT_ROOT
from logging_utils import get_logger

logger = get_logger("ubf_features")

UBF_SCHEMA_PATH = PROJECT_ROOT / "config" / "feature_schema.json"
REPORTS_DIR     = PROJECT_ROOT / "reports" / "features"


def get_ubf(df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """Extract and order UBF columns from df.

    Args:
        df:            DataFrame with engineered features.
        fill_missing:  If True, missing UBF columns are zero-filled.

    Returns:
        DataFrame with exactly UBF_FEATURES columns in order.
    """
    missing = [c for c in UBF_FEATURES if c not in df.columns]
    if missing:
        if fill_missing:
            logger.warning("UBF missing columns (zero-filled): %s", missing)
            for col in missing:
                df = df.copy()
                df[col] = 0.0
        else:
            raise ValueError(f"UBF missing columns: {missing}")

    extra = [c for c in df.columns if c not in UBF_FEATURES]
    if extra:
        logger.debug("UBF: ignoring extra columns: %s", extra[:10])

    return df[UBF_FEATURES].copy()


def save_ubf_schema(out_path: Path | None = None) -> dict:
    schema = {
        "ubf_version": "1.0",
        "features":    UBF_FEATURES,
        "feature_count": len(UBF_FEATURES),
    }
    path = Path(out_path) if out_path else UBF_SCHEMA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("UBF schema saved → %s", path)
    return schema


def generate_ubf_reports(ubf_df: pd.DataFrame, out_dir: Path | None = None) -> None:
    """Generate UBF statistics and correlation reports."""
    out_dir = Path(out_dir) if out_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # statistics
    stats = ubf_df.describe().T
    stats["missing"] = ubf_df.isnull().sum()
    stats.to_csv(out_dir / "ubf_statistics.csv")
    logger.info("UBF statistics → %s", out_dir / "ubf_statistics.csv")

    # correlation matrix
    corr = ubf_df.corr()
    corr.to_csv(out_dir / "ubf_correlation_matrix.csv")
    logger.info("UBF correlation matrix → %s", out_dir / "ubf_correlation_matrix.csv")

    # correlation heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title("UBF Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        heatmap_path = out_dir / "ubf_correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("UBF heatmap → %s", heatmap_path)
    except Exception as exc:
        logger.warning("Could not generate heatmap: %s", exc)

    # manifest
    manifest = pd.DataFrame({
        "feature_name": UBF_FEATURES,
        "position":     range(len(UBF_FEATURES)),
        "ubf_version":  "1.0",
    })
    manifest.to_csv(out_dir / "ubf_manifest.csv", index=False)

    # definition markdown
    md_lines = [
        "# Unified Behavioral Feature (UBF) Definition",
        "",
        f"**Version:** 1.0  ",
        f"**Feature count:** {len(UBF_FEATURES)}",
        "",
        "## Feature List",
        "",
        "| Position | Feature Name |",
        "|---|---|",
    ]
    for i, feat in enumerate(UBF_FEATURES):
        md_lines.append(f"| {i:2d} | `{feat}` |")
    md_lines += [
        "",
        "## Notes",
        "",
        "- `NsisAy` is held out from supervised training as zero-day proxy.",
        "- All features are extracted from raw CTU-13 binetflow data.",
        "- Protocol and state columns are one-hot encoded.",
        "- Rate features use safe division (ε = 1e-9) for zero-duration flows.",
    ]
    (out_dir / "ubf_definition.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("UBF definition → %s", out_dir / "ubf_definition.md")
