"""
src/dataset_discovery.py
Recursively discover CTU-13 .binetflow / flow CSV files.

Usage:
    python src/dataset_discovery.py --data-dir data/raw
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from logging_utils import get_logger
from config import PROJECT_ROOT, RAW_DIR

logger = get_logger("dataset_discovery")

# Extensions we consider valid CTU-13 flow files
VALID_EXTENSIONS = {".binetflow", ".csv"}

# Minimum columns that must be present (case-insensitive, after normalisation)
REQUIRED_COLUMNS_MIN = {"label"}   # at least a label column
FLOW_COLUMNS_HINT = {
    "starttime", "dur", "duration", "proto", "protocol",
    "srcaddr", "src_addr", "totpkts", "total_packets",
    "totbytes", "total_bytes", "srcbytes", "src_bytes", "label",
}


def _is_valid_flow_file(path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False, f"extension {path.suffix!r} not in {VALID_EXTENSIONS}"
    if path.stat().st_size < 100:
        return False, "file too small (<100 bytes)"
    # Try to read header
    try:
        header = pd.read_csv(path, nrows=0)
        cols_lower = {c.strip().lower() for c in header.columns}
        overlap = cols_lower & FLOW_COLUMNS_HINT
        if len(overlap) < 2:
            return False, f"too few recognisable flow columns (found {overlap})"
        return True, "ok"
    except Exception as exc:
        return False, f"parse error: {exc}"


def _infer_scenario_id(path: Path) -> str:
    """Infer CTU-13 scenario ID from file path."""
    # Check parent directory name
    for part in reversed(path.parts):
        if part.isdigit() and 1 <= int(part) <= 13:
            return part
    # Check filename for digits
    stem = path.stem
    for token in stem.replace("-", "_").split("_"):
        if token.isdigit() and 1 <= int(token) <= 13:
            return token
    return "unknown"


def discover(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Walk data_dir and return:
      accepted: list of dicts with path, scenario_id, size_bytes
      rejected: list of dicts with path, reason
    """
    accepted, rejected = [], []
    data_dir = Path(data_dir)

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        return accepted, rejected

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue  # silently skip non-flow extensions
        valid, reason = _is_valid_flow_file(path)
        if valid:
            scenario_id = _infer_scenario_id(path)
            accepted.append({
                "path": str(path),
                "scenario_id": scenario_id,
                "size_bytes": path.stat().st_size,
                "filename": path.name,
            })
            logger.info("ACCEPTED  scenario=%s  %s", scenario_id, path.name)
        else:
            rejected.append({"path": str(path), "reason": reason})
            logger.warning("REJECTED  %s  reason=%s", path.name, reason)

    return accepted, rejected


def save_discovery_report(accepted: list[dict], rejected: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_path = out_dir / "discovered_files.csv"
    rej_path = out_dir / "rejected_files.csv"

    with open(acc_path, "w", newline="", encoding="utf-8") as f:
        if accepted:
            writer = csv.DictWriter(f, fieldnames=accepted[0].keys())
            writer.writeheader()
            writer.writerows(accepted)
    logger.info("Wrote %d accepted entries → %s", len(accepted), acc_path)

    with open(rej_path, "w", newline="", encoding="utf-8") as f:
        if rejected:
            writer = csv.DictWriter(f, fieldnames=["path", "reason"])
            writer.writeheader()
            writer.writerows(rejected)
    logger.info("Wrote %d rejected entries → %s", len(rejected), rej_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover CTU-13 flow files.")
    parser.add_argument("--data-dir", default=str(RAW_DIR), help="Root data directory to scan")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "dataset"))
    args = parser.parse_args(argv)

    accepted, rejected = discover(Path(args.data_dir))

    if not accepted:
        logger.error("No usable CTU-13 files found in %s. Exiting.", args.data_dir)
        return 1

    save_discovery_report(accepted, rejected, Path(args.output_dir))
    logger.info("Discovery complete: %d accepted, %d rejected.", len(accepted), len(rejected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
