"""
src/extract_ctu13.py
Parse raw CTU-13 .binetflow files → interim/ctu13_raw.csv

Steps:
  1. Discover all scenario folders under data/raw
  2. Read each .binetflow file
  3. Normalise column names (ctu13_schema)
  4. Attach scenario_id
  5. Normalise labels (label_normalization)
  6. Concatenate and save to data/interim/ctu13_raw.csv

Usage:
    python src/extract_ctu13.py [--data-dir data/raw] [--output-dir data/interim]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, RAW_DIR, INTERIM_DIR, CTU13_SCENARIO_FAMILY
from logging_utils import get_logger
from dataset_discovery import discover
from ctu13_schema import normalise_columns, check_required_columns, inspect_schema
from label_normalization import normalise_labels, generate_label_mapping_report, label_distribution

logger = get_logger("extract_ctu13", log_file=PROJECT_ROOT / "reports" / "dataset" / "extract.log")

# Expected CTU-13 subfolder containing scenario dirs
CTU13_SUBDIR = "CTU-13-Dataset.tar"


def _find_ctu13_root(raw_dir: Path) -> Path | None:
    """Try to locate the CTU-13-Dataset folder."""
    for candidate in [
        raw_dir / CTU13_SUBDIR,
        raw_dir / "CTU-13-Dataset",
        raw_dir,
    ]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def load_one_scenario(scenario_dir: Path, scenario_id: str) -> pd.DataFrame | None:
    """Load all flow files from one scenario directory."""
    frames = []
    for fp in sorted(scenario_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".binetflow", ".csv"}:
            continue
        try:
            df = pd.read_csv(fp, low_memory=False)
            df = normalise_columns(df)
            missing = check_required_columns(df)
            if missing:
                logger.warning("Scenario %s  %s  missing columns: %s — skipped", scenario_id, fp.name, missing)
                continue
            df["scenario_id"] = scenario_id
            df["source_file"] = fp.name
            logger.info("  Loaded scenario=%s  rows=%d  file=%s", scenario_id, len(df), fp.name)
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", fp, exc)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract CTU-13 raw data.")
    parser.add_argument("--data-dir",    default=str(RAW_DIR))
    parser.add_argument("--output-dir",  default=str(INTERIM_DIR))
    parser.add_argument("--max-rows",    type=int, default=None, help="Limit rows per scenario (dev mode)")
    args = parser.parse_args(argv)

    raw_dir  = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctu13_root = _find_ctu13_root(raw_dir)
    if ctu13_root is None:
        logger.error("Cannot locate CTU-13-Dataset directory under %s. Exiting.", raw_dir)
        return 1

    logger.info("CTU-13 root: %s", ctu13_root)

    t0 = time.time()
    all_frames = []
    scenario_stats = []

    # Try numeric scenario folders (1..13) first, then fall back to discovery
    found_scenarios = 0
    for sid in range(1, 14):
        sdir = ctu13_root / str(sid)
        if not sdir.exists():
            logger.warning("Scenario dir %s not found — skipped.", sdir)
            continue
        found_scenarios += 1
        df = load_one_scenario(sdir, str(sid))
        if df is None:
            logger.warning("No usable files in scenario %d.", sid)
            continue
        if args.max_rows:
            df = df.head(args.max_rows)
        scenario_stats.append({
            "scenario_id": str(sid),
            "family": CTU13_SCENARIO_FAMILY.get(sid, "Unknown"),
            "rows_raw": len(df),
        })
        all_frames.append(df)

    if not all_frames:
        logger.error("No data loaded from any scenario. Exiting.")
        return 1

    raw_df = pd.concat(all_frames, ignore_index=True)
    logger.info("Total raw rows before normalisation: %d", len(raw_df))

    # Normalise labels
    raw_df = normalise_labels(raw_df)

    dist = label_distribution(raw_df)
    logger.info("Label distribution: benign=%d  botnet=%d  unresolved=%d",
                dist["benign"], dist["botnet"], dist["unresolved"])
    logger.info("Family counts: %s", dist["families"])

    # Save interim
    out_path = out_dir / "ctu13_raw.csv"
    raw_df.to_csv(out_path, index=False)
    logger.info("Saved interim dataset → %s  (%d rows, %d cols)", out_path, *raw_df.shape)

    # Reports
    report_dir = PROJECT_ROOT / "reports" / "dataset"
    report_dir.mkdir(parents=True, exist_ok=True)
    generate_label_mapping_report(raw_df, report_dir / "label_mapping.csv")

    # Scenario stats
    import json
    stats_path = report_dir / "scenario_stats.json"
    with open(stats_path, "w") as f:
        json.dump(scenario_stats, f, indent=2)
    logger.info("Scenario stats → %s", stats_path)

    elapsed = time.time() - t0
    logger.info("Extraction complete in %.1f s.", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
