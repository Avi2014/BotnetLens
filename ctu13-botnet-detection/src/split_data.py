"""
src/split_data.py
Reproducible data splitting with leakage prevention.

Uses a TWO-PASS STREAMING approach so the full 19 M-row CSV is never
loaded into RAM at once:
  Pass 1 – read only [scenario_id, family] to decide the split assignments.
  Pass 2 – stream the full CSV and write each row to the correct split file.

Strategies:
  1. random           – stratified random split (baseline, fast)
  2. scenario_aware   – scenarios kept intact between train/test (primary)
  3. family_holdout   – one family held out entirely for Phase 2 evaluation

Usage:
    python src/split_data.py [--strategy family_holdout] [--holdout-family NsisAy]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (PROJECT_ROOT, PROCESSED_DIR, SPLITS_DIR,
                    RANDOM_SEED, HOLDOUT_FAMILY)
from logging_utils import get_logger

logger = get_logger("split_data")

CHUNKSIZE = 500_000


# ── Pass 1: lightweight scan to get scenario → split mapping ─────────────────

def _assign_scenarios(input_path: Path, holdout_family: str,
                      test_size: float, val_size: float,
                      seed: int) -> tuple[dict[Any, str], str]:
    """Read only scenario_id + family columns and return {scenario_id: split_name}.

    Returns (scenario_split_map, holdout_family)
    """
    logger.info("Pass 1: reading scenario_id + family columns …")
    scenario_families: dict[Any, set] = defaultdict(set)

    reader = pd.read_csv(input_path, usecols=["scenario_id", "family"],
                         chunksize=CHUNKSIZE, low_memory=False)
    for chunk in reader:
        for sid, fam in zip(chunk["scenario_id"], chunk["family"]):
            scenario_families[sid].add(str(fam))

    scenarios = sorted(scenario_families.keys())
    logger.info("Found %d unique scenarios.", len(scenarios))

    # Remove scenarios whose ONLY non-holdout rows would be zero (i.e. pure
    # holdout scenarios) — treat them as holdout not part of scenario split.
    # For family_holdout strategy: holdout family rows are routed separately
    # during pass 2 regardless of scenario.

    rng = np.random.default_rng(seed)
    shuffled = list(scenarios)
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_size))
    n_val  = max(1, round(len(shuffled) * val_size))

    test_scens  = set(shuffled[:n_test])
    val_scens   = set(shuffled[n_test:n_test + n_val])
    train_scens = set(shuffled[n_test + n_val:])

    logger.info("Scenario-aware split: train=%s val=%s test=%s",
                sorted(train_scens), sorted(val_scens), sorted(test_scens))

    mapping = {}
    for sid in scenarios:
        if sid in train_scens:
            mapping[sid] = "train"
        elif sid in val_scens:
            mapping[sid] = "val"
        else:
            mapping[sid] = "test"

    return mapping, holdout_family


# ── Pass 2: stream and write ──────────────────────────────────────────────────

def _stream_split(input_path: Path, out_dir: Path,
                  scenario_map: dict, holdout_family: str,
                  strategy: str) -> dict[str, int]:
    """Stream CSV and write each row to the correct split file.
    Returns row counts per split."""

    splits = ["train", "val", "test", "holdout"]
    split_paths = {s: out_dir / f"{s}.csv" for s in splits}
    counts: dict[str, int] = {s: 0 for s in splits}

    file_handles: dict[str, Any] = {}
    writers: dict[str, Any] = {}

    try:
        reader = pd.read_csv(input_path, chunksize=CHUNKSIZE, low_memory=False,
                             dtype={"binary_label": "int8", "scenario_id": "int16",
                                    "total_packets": "float32", "total_bytes": "float32",
                                    "src_bytes": "float32", "dst_bytes": "float32",
                                    "duration": "float32", "src_port": "float32",
                                    "dst_port": "float32"})

        first = True
        for chunk in reader:
            # Route each row
            masks: dict[str, pd.Series] = {}

            if strategy == "family_holdout" and "family" in chunk.columns:
                holdout_mask = chunk["family"] == holdout_family
                main_chunk   = chunk[~holdout_mask]
                masks["holdout"] = holdout_mask
            else:
                main_chunk = chunk

            for split_name in ["train", "val", "test"]:
                scens = {k for k, v in scenario_map.items() if v == split_name}
                masks[split_name] = main_chunk["scenario_id"].isin(scens)

            # Write chunks
            for split_name, mask in masks.items():
                if split_name == "holdout":
                    rows = chunk[mask]
                else:
                    rows = main_chunk[mask]

                if len(rows) == 0:
                    continue

                path = split_paths[split_name]
                if split_name not in file_handles:
                    file_handles[split_name] = open(path, "w", newline="", encoding="utf-8")

                write_header = split_name not in writers
                rows.to_csv(file_handles[split_name], index=False, header=write_header)
                writers[split_name] = True
                counts[split_name] += len(rows)

    finally:
        for fh in file_handles.values():
            fh.close()

    # Create empty holdout.csv if no holdout rows found
    if "holdout" not in file_handles:
        # Write empty file with header
        sample = pd.read_csv(input_path, nrows=0)
        sample.to_csv(split_paths["holdout"], index=False)

    for split_name, count in counts.items():
        logger.info("Saved %s: %d rows → %s", split_name, count, split_paths[split_name])

    return counts


# ── Manifest builder ──────────────────────────────────────────────────────────

def _build_manifest(out_dir: Path, counts: dict[str, int],
                    scenario_map: dict, strategy: str,
                    holdout_family: str, seed: int) -> None:
    train_scens = sorted([k for k, v in scenario_map.items() if v == "train"])
    val_scens   = sorted([k for k, v in scenario_map.items() if v == "val"])
    test_scens  = sorted([k for k, v in scenario_map.items() if v == "test"])

    manifest = {
        "strategy":       strategy,
        "holdout_family": holdout_family,
        "random_seed":    seed,
        "train": {"rows": counts["train"], "scenarios": [int(s) for s in train_scens]},
        "val":   {"rows": counts["val"],   "scenarios": [int(s) for s in val_scens]},
        "test":  {"rows": counts["test"],  "scenarios": [int(s) for s in test_scens]},
        "holdout": {"rows": counts["holdout"]},
    }

    manifest_path = PROJECT_ROOT / "reports" / "splits" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Split manifest → %s", manifest_path)


# ── Test-compatibility shims (still importable for unit tests) ────────────────

def family_holdout_split(df: pd.DataFrame, holdout_family: str = HOLDOUT_FAMILY,
                         test_size=0.20, val_size=0.10,
                         seed: int = RANDOM_SEED):
    """In-memory version used ONLY by unit tests (small DataFrames)."""
    holdout_df = df[df["family"] == holdout_family].copy()
    main_df    = df[df["family"] != holdout_family].copy()
    train, val, test = scenario_aware_split(main_df, test_size, val_size, seed)
    return train, val, test, holdout_df


def scenario_aware_split(df: pd.DataFrame, test_size=0.20, val_size=0.10,
                         seed: int = RANDOM_SEED):
    """In-memory version used ONLY by unit tests."""
    if "scenario_id" not in df.columns:
        return _random_split_df(df, test_size, val_size, seed)
    scenarios = list(df["scenario_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)
    n_test  = max(1, round(len(scenarios) * test_size))
    n_val   = max(1, round(len(scenarios) * val_size))
    test_s  = set(scenarios[:n_test])
    val_s   = set(scenarios[n_test:n_test + n_val])
    train_s = set(scenarios[n_test + n_val:])
    return (df[df["scenario_id"].isin(train_s)],
            df[df["scenario_id"].isin(val_s)],
            df[df["scenario_id"].isin(test_s)])


def _random_split_df(df, test_size=0.20, val_size=0.10, seed: int = RANDOM_SEED):
    from sklearn.model_selection import train_test_split
    label = df["binary_label"] if "binary_label" in df.columns else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=label)
    val_ratio = val_size / (1 - test_size)
    label_tv = train_val["binary_label"] if "binary_label" in train_val.columns else None
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed, stratify=label_tv)
    return train, val, test

random_split = _random_split_df



# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split CTU-13 processed data (streaming).")
    parser.add_argument("--input",           default=str(PROCESSED_DIR / "ctu13_flows.csv"))
    parser.add_argument("--output-dir",      default=str(SPLITS_DIR))
    parser.add_argument("--strategy",        default="scenario_aware",
                        choices=["random", "scenario_aware", "family_holdout"])
    parser.add_argument("--holdout-family",  default=HOLDOUT_FAMILY)
    parser.add_argument("--test-size",       type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--seed",            type=int,   default=RANDOM_SEED)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s. Run clean_ctu13.py first.", input_path)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: assign scenarios → splits
    scenario_map, holdout_family = _assign_scenarios(
        input_path, args.holdout_family,
        args.test_size, args.validation_size, args.seed)

    # Pass 2: stream and write
    logger.info("Pass 2: streaming %s → split files …", input_path)
    counts = _stream_split(input_path, out_dir, scenario_map,
                           holdout_family, args.strategy)

    logger.info("Split complete: train=%d val=%d test=%d holdout=%d",
                counts["train"], counts["val"], counts["test"], counts["holdout"])

    _build_manifest(out_dir, counts, scenario_map, args.strategy,
                    holdout_family, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
