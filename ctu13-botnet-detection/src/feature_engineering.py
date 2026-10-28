"""
src/feature_engineering.py
Flow-level feature extraction for CTU-13 data.

Produces all features required for the UBF representation.
Safe division is used throughout (epsilon = 1e-9).
No label or identifier columns are used as features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import UBF_FEATURES
from logging_utils import get_logger

logger = get_logger("feature_engineering")

EPSILON = 1e-9
FEATURE_MANIFEST_COLS = [
    "feature_name", "category", "source_column", "transformation",
    "data_type", "missing_value_policy", "leakage_risk", "used_in_ubf",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, EPSILON)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all engineered flow features from a cleaned CTU-13 DataFrame."""
    out = pd.DataFrame(index=df.index)

    # ── A. Basic flow features ────────────────────────────────────────────────
    out["duration"]       = df.get("duration", pd.Series(0.0, index=df.index)).fillna(0.0)
    out["src_bytes"]      = df.get("src_bytes", pd.Series(0.0, index=df.index)).fillna(0.0)
    out["dst_bytes"]      = df.get("dst_bytes", pd.Series(0.0, index=df.index)).fillna(0.0)
    out["total_bytes"]    = df.get("total_bytes", out["src_bytes"] + out["dst_bytes"]).fillna(0.0)
    out["total_packets"]  = df.get("total_packets", pd.Series(1.0, index=df.index)).fillna(1.0).replace(0, 1)

    # ── B. Rate features ──────────────────────────────────────────────────────
    dur = out["duration"].replace(0, EPSILON)
    out["bytes_per_second"]      = _safe_div(out["total_bytes"],   dur)
    out["packets_per_second"]    = _safe_div(out["total_packets"],  dur)
    out["src_bytes_per_second"]  = _safe_div(out["src_bytes"],     dur)
    out["dst_bytes_per_second"]  = _safe_div(out["dst_bytes"],     dur)

    # ── C. Ratio features ─────────────────────────────────────────────────────
    out["bytes_ratio"] = _safe_div(out["src_bytes"], out["total_bytes"].replace(0, EPSILON))

    # ── D. Packet-size features ───────────────────────────────────────────────
    pkts = out["total_packets"].replace(0, EPSILON)
    out["mean_packet_size"]     = _safe_div(out["total_bytes"], pkts)
    out["src_mean_packet_size"] = _safe_div(out["src_bytes"],   pkts)

    # ── E. Protocol one-hot ───────────────────────────────────────────────────
    proto = df.get("proto", pd.Series("unknown", index=df.index)).astype(str).str.lower().fillna("unknown")
    out["proto_tcp"]  = (proto == "tcp").astype(np.int8)
    out["proto_udp"]  = (proto == "udp").astype(np.int8)
    out["proto_icmp"] = (proto == "icmp").astype(np.int8)

    # ── F. Connection-state one-hot ───────────────────────────────────────────
    state = df.get("state", pd.Series("unknown", index=df.index)).astype(str).str.upper().fillna("UNKNOWN")
    out["state_SF"]  = (state == "SF").astype(np.int8)
    out["state_S0"]  = (state == "S0").astype(np.int8)
    out["state_REJ"] = (state == "REJ").astype(np.int8)

    # ── G. Replace inf / NaN ─────────────────────────────────────────────────
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Clip rate features to reasonable upper bound (prevents outlier skew)
    rate_cols = ["bytes_per_second", "packets_per_second",
                 "src_bytes_per_second", "dst_bytes_per_second"]
    for col in rate_cols:
        q99 = out[col].quantile(0.999)
        out[col] = out[col].clip(upper=max(q99, 1.0))

    logger.info("Feature extraction: %d rows → %d features", len(out), len(out.columns))
    return out


def build_feature_manifest(out_dir: Path) -> pd.DataFrame:
    """Write feature_manifest.csv."""
    manifest = [
        # (feature_name, category, source_column, transformation, dtype, missing_policy, leakage_risk, in_ubf)
        ("duration",              "basic",    "duration",      "direct",                 "float", "fill_0",    "no",  "yes"),
        ("src_bytes",             "basic",    "src_bytes",     "direct",                 "float", "fill_0",    "no",  "yes"),
        ("dst_bytes",             "basic",    "dst_bytes",     "total-src or direct",    "float", "fill_0",    "no",  "yes"),
        ("total_bytes",           "basic",    "total_bytes",   "direct",                 "float", "fill_0",    "no",  "yes"),
        ("total_packets",         "basic",    "total_packets", "direct",                 "float", "fill_1",    "no",  "yes"),
        ("bytes_per_second",      "rate",     "total_bytes/duration", "safe_div",        "float", "fill_0",    "no",  "yes"),
        ("packets_per_second",    "rate",     "total_packets/duration", "safe_div",      "float", "fill_0",    "no",  "yes"),
        ("src_bytes_per_second",  "rate",     "src_bytes/duration", "safe_div",         "float", "fill_0",    "no",  "no"),
        ("dst_bytes_per_second",  "rate",     "dst_bytes/duration", "safe_div",         "float", "fill_0",    "no",  "no"),
        ("bytes_ratio",           "ratio",    "src_bytes/total_bytes", "safe_div",       "float", "fill_0",    "no",  "yes"),
        ("mean_packet_size",      "pkt_size", "total_bytes/total_packets", "safe_div",   "float", "fill_0",    "no",  "yes"),
        ("src_mean_packet_size",  "pkt_size", "src_bytes/total_packets", "safe_div",    "float", "fill_0",    "no",  "yes"),
        ("proto_tcp",             "protocol", "proto",         "one_hot",                "int8",  "fill_0",    "no",  "yes"),
        ("proto_udp",             "protocol", "proto",         "one_hot",                "int8",  "fill_0",    "no",  "yes"),
        ("proto_icmp",            "protocol", "proto",         "one_hot",                "int8",  "fill_0",    "no",  "yes"),
        ("state_SF",              "state",    "state",         "one_hot",                "int8",  "fill_0",    "no",  "yes"),
        ("state_S0",              "state",    "state",         "one_hot",                "int8",  "fill_0",    "no",  "yes"),
        ("state_REJ",             "state",    "state",         "one_hot",                "int8",  "fill_0",    "no",  "yes"),
    ]
    cols = FEATURE_MANIFEST_COLS + ["selected_for_model"]
    df = pd.DataFrame(manifest, columns=cols)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "feature_manifest.csv", index=False)
    logger.info("Feature manifest → %s", out_dir / "feature_manifest.csv")
    return df
