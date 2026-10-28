"""
src/config.py
Central configuration loader for the CTU-13 botnet detection project.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
SRC_DIR      = PROJECT_ROOT / "src"

# ── Default paths ─────────────────────────────────────────────────────────────
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR    = DATA_DIR / "splits"

# ── CTU-13 scenario → family mapping ─────────────────────────────────────────
# Source: https://www.stratosphereips.org/datasets-ctu13
CTU13_SCENARIO_FAMILY: dict[int, str] = {
    1:  "Neris",
    2:  "Neris",
    3:  "Rbot",
    4:  "Rbot",
    5:  "Virut",
    6:  "Menti",
    7:  "Sogou",
    8:  "Murlo",
    9:  "Neris",
    10: "Rbot",
    11: "Rbot",
    12: "NsisAy",
    13: "Virut",
}

ALL_FAMILIES  = sorted(set(CTU13_SCENARIO_FAMILY.values()))
BENIGN_FAMILY = "Benign"
UNKNOWN_FAMILY = "Unknown"

# ── Holdout family (zero-day proxy) ──────────────────────────────────────────
# NsisAy: ultra-short duration, most aggressive rate — most distinctive family
HOLDOUT_FAMILY = "NsisAy"

# ── UBF features ─────────────────────────────────────────────────────────────
UBF_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "total_bytes",
    "total_packets",
    "bytes_per_second",
    "packets_per_second",
    "bytes_ratio",
    "mean_packet_size",
    "src_mean_packet_size",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "state_SF",
    "state_S0",
    "state_REJ",
]

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Label normalization rules ─────────────────────────────────────────────────
# Maps raw CTU-13 Label substrings → (binary, family_hint)
LABEL_RULES: list[tuple[str, int, str]] = [
    # (substring_lower, binary_label, family_hint)
    ("background",  0, "Benign"),
    ("benign",      0, "Benign"),
    ("normal",      0, "Benign"),
    ("from-botnet", 1, ""),      # family inferred from scenario
    ("botnet",      1, ""),
    ("malicious",   1, ""),
]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load default.yaml, optionally overriding with user config."""
    default = CONFIG_DIR / "default.yaml"
    cfg = load_yaml(default) if default.exists() else {}
    if config_path and Path(config_path).exists():
        override = load_yaml(Path(config_path))
        cfg.update(override)
    return cfg


def load_feature_schema(schema_path: Path | None = None) -> dict[str, Any]:
    path = schema_path or (CONFIG_DIR / "feature_schema.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
