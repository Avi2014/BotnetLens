"""
src/ctu13_schema.py
Schema normalisation layer for CTU-13 binetflow / CSV files.

Handles column-name variations without hardcoding a single layout.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from logging_utils import get_logger

logger = get_logger("ctu13_schema")

# ── Canonical column name mapping ─────────────────────────────────────────────
# Maps every known variant (lower-stripped) → canonical name
COLUMN_ALIASES: dict[str, str] = {
    # timestamps
    "starttime":      "start_time",
    "start_time":     "start_time",
    "timestamp":      "start_time",
    "time":           "start_time",
    # duration
    "dur":            "duration",
    "duration":       "duration",
    # protocol
    "proto":          "proto",
    "protocol":       "proto",
    # addresses
    "srcaddr":        "src_addr",
    "src_addr":       "src_addr",
    "dstaddr":        "dst_addr",
    "dst_addr":       "dst_addr",
    # ports
    "sport":          "src_port",
    "src_port":       "src_port",
    "dport":          "dst_port",
    "dst_port":       "dst_port",
    # direction
    "dir":            "direction",
    "direction":      "direction",
    # state
    "state":          "state",
    # tos
    "stos":           "src_tos",
    "dtos":           "dst_tos",
    "src_tos":        "src_tos",
    "dst_tos":        "dst_tos",
    # packet / byte counts
    "totpkts":        "total_packets",
    "tot_pkts":       "total_packets",
    "total_packets":  "total_packets",
    "totbytes":       "total_bytes",
    "tot_bytes":      "total_bytes",
    "total_bytes":    "total_bytes",
    "srcbytes":       "src_bytes",
    "src_bytes":      "src_bytes",
    # label
    "label":          "label",
    # flow id
    "flow_id":        "flow_id",
}

REQUIRED_COLUMNS = {"duration", "total_bytes", "src_bytes", "label"}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
        else:
            rename_map[col] = col  # keep as-is
    df = df.rename(columns=rename_map)
    return df


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Return list of missing required columns."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def inspect_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Return a schema summary dict."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_required": check_required_columns(df),
        "null_counts": df.isnull().sum().to_dict(),
    }
