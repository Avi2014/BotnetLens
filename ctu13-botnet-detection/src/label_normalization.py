"""
src/label_normalization.py
Documented label normalisation for CTU-13 raw Label column.

Raw labels seen in CTU-13:
  Background-*          → Benign  (binary=0)
  flow=Background-*     → Benign
  From-Botnet-*         → Botnet  (binary=1, family inferred from scenario)
  flow=From-Botnet-*    → Botnet
  Botnet                → Botnet
  Normal / Benign       → Benign

Family is inferred from scenario_id using CTU13_SCENARIO_FAMILY.
If family cannot be reliably inferred it is marked as "Unknown".
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import CTU13_SCENARIO_FAMILY, BENIGN_FAMILY, UNKNOWN_FAMILY, HOLDOUT_FAMILY
from logging_utils import get_logger

logger = get_logger("label_normalization")

# Each rule: (regex_pattern, binary_label, family_hint)
# Rules are applied top-down; first match wins.
_LABEL_RULES: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"background",   re.I), 0, BENIGN_FAMILY),
    (re.compile(r"\bbenign\b",   re.I), 0, BENIGN_FAMILY),
    (re.compile(r"\bnormal\b",   re.I), 0, BENIGN_FAMILY),
    (re.compile(r"from-botnet",  re.I), 1, ""),   # family from scenario
    (re.compile(r"\bbotnet\b",   re.I), 1, ""),
    (re.compile(r"malicious",    re.I), 1, ""),
]


def _match_rule(raw: str) -> tuple[int, str]:
    """Return (binary_label, family_hint) for a raw label string."""
    s = str(raw).strip()
    for pattern, binary, hint in _LABEL_RULES:
        if pattern.search(s):
            return binary, hint
    return -1, UNKNOWN_FAMILY   # unresolved


def normalise_single(raw_label: str, scenario_id: Any) -> dict[str, Any]:
    """Normalise one label value.

    Returns:
        {raw_label, binary_label, family, mapping_rule, status}
    """
    binary, hint = _match_rule(raw_label)

    if binary == -1:
        return {
            "raw_label":    raw_label,
            "binary_label": -1,
            "family":       UNKNOWN_FAMILY,
            "mapping_rule": "no_match",
            "status":       "unresolved",
        }

    if binary == 0:
        return {
            "raw_label":    raw_label,
            "binary_label": 0,
            "family":       BENIGN_FAMILY,
            "mapping_rule": "background/benign/normal",
            "status":       "ok",
        }

    # binary == 1: family from scenario
    try:
        sid = int(str(scenario_id).strip())
        family = CTU13_SCENARIO_FAMILY.get(sid, UNKNOWN_FAMILY)
    except (ValueError, TypeError):
        family = UNKNOWN_FAMILY

    status = "ok" if family != UNKNOWN_FAMILY else "unresolved_family"
    return {
        "raw_label":    raw_label,
        "binary_label": 1,
        "family":       family,
        "mapping_rule": "from-botnet→scenario_id",
        "status":       status,
    }


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary_label and family columns to df.

    Expects columns: label, scenario_id
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame missing 'label' column.")
    if "scenario_id" not in df.columns:
        logger.warning("'scenario_id' column missing; family inference may fail.")
        df = df.copy()
        df["scenario_id"] = "unknown"

    results = [
        normalise_single(row["label"], row["scenario_id"])
        for _, row in df[["label", "scenario_id"]].iterrows()
    ]
    result_df = pd.DataFrame(results)

    df = df.copy()
    df["binary_label"] = result_df["binary_label"].values
    df["family"]       = result_df["family"].values
    df["_mapping_rule"] = result_df["mapping_rule"].values
    df["_norm_status"]  = result_df["status"].values
    return df


def generate_label_mapping_report(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Write label_mapping.csv with per-raw-label stats."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    group = (
        df.groupby(["label", "binary_label", "family", "_mapping_rule", "_norm_status"])
        .size()
        .reset_index(name="row_count")
    )
    group.columns = [
        "raw_label", "binary_label", "family",
        "mapping_rule", "status", "row_count"
    ]
    group.to_csv(out_path, index=False)
    logger.info("Label mapping report → %s  (%d rows)", out_path, len(group))
    return group


def label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total":      len(df),
        "benign":     int((df["binary_label"] == 0).sum()),
        "botnet":     int((df["binary_label"] == 1).sum()),
        "unresolved": int((df["binary_label"] == -1).sum()),
        "families":   df["family"].value_counts().to_dict(),
    }
