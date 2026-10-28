"""
src/clean_ctu13.py
Data cleaning pipeline for CTU-13 interim data.

Processes data in CHUNKS to avoid OOM on 19 M-row files.

Input:  data/interim/ctu13_raw.csv
Output: data/processed/ctu13_flows.csv
        reports/dataset/data_quality_report.md
        reports/dataset/data_quality_summary.csv
        reports/dataset/cleaning_log.csv

Usage:
    python src/clean_ctu13.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, INTERIM_DIR, PROCESSED_DIR
from logging_utils import get_logger

logger = get_logger("clean_ctu13", log_file=PROJECT_ROOT / "reports" / "dataset" / "clean.log")

EPSILON   = 1e-9
CHUNKSIZE = 500_000          # rows per chunk — tune if still tight on RAM


# ── Per-row cleaning (stateless, applied to every chunk) ─────────────────────

def _safe_numeric(series: pd.Series, col: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    return converted


def clean_chunk(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean one chunk.  Returns (cleaned_chunk, step_counts_dict)."""
    counts: dict[str, int] = defaultdict(int)
    n_in = len(df)
    counts["rows_in"] += n_in

    # 1. Normalise proto / direction
    if "proto" in df.columns:
        df["proto"] = df["proto"].astype(str).str.strip().str.lower()
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.strip()

    # 2. Replace dash/empty → NaN
    df = df.replace(r"^\s*-\s*$", np.nan, regex=True)
    df = df.replace("", np.nan)

    # 3. Parse numeric columns (coerce to float to avoid int64 OOM spike)
    numeric_cols = ["duration", "total_bytes", "src_bytes", "total_packets",
                    "src_tos", "dst_tos", "src_port", "dst_port"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Replace ±inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # 5. Remove rows with negative bytes / packets
    mask_neg = pd.Series(False, index=df.index)
    for col in ["total_bytes", "src_bytes", "total_packets"]:
        if col in df.columns:
            mask_neg |= (df[col].fillna(0) < 0)
    removed_neg = mask_neg.sum()
    df = df[~mask_neg]
    counts["remove_negative_bytes_pkts"] += int(removed_neg)

    # 6. Remove rows with negative duration
    if "duration" in df.columns:
        mask_dur = df["duration"].fillna(0) < 0
        counts["remove_negative_duration"] += int(mask_dur.sum())
        df = df[~mask_dur]

    # 7. Fill NaN durations / bytes / packets with 0
    if "duration" in df.columns:
        df["duration"] = df["duration"].fillna(0.0)
    for col in ["total_bytes", "src_bytes", "total_packets"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 8. Derive dst_bytes if missing
    if "dst_bytes" not in df.columns and all(c in df.columns for c in ["total_bytes", "src_bytes"]):
        df["dst_bytes"] = (df["total_bytes"] - df["src_bytes"]).clip(lower=0)

    # 9. Remove unresolved labels (binary_label == -1)
    if "binary_label" in df.columns:
        mask_bad = df["binary_label"] == -1
        counts["remove_unresolved_labels"] += int(mask_bad.sum())
        df = df[~mask_bad]

    # 10. Drop duplicate rows within chunk
    n_before_dup = len(df)
    df = df.drop_duplicates()
    counts["drop_duplicates"] += n_before_dup - len(df)

    counts["rows_out"] += len(df)
    return df.reset_index(drop=True), counts


# ── Aggregated quality stats (accumulated across chunks) ─────────────────────

class QualityAccumulator:
    def __init__(self):
        self.rows_raw   = 0
        self.rows_clean = 0
        self.null_total = 0
        self.benign     = 0
        self.botnet     = 0
        self.families: dict[str, int] = defaultdict(int)
        self.scenarios: dict[str, int] = defaultdict(int)
        self.step_totals: dict[str, int] = defaultdict(int)

    def accumulate(self, raw_chunk: pd.DataFrame, clean_chunk: pd.DataFrame,
                   step_counts: dict) -> None:
        self.rows_raw   += len(raw_chunk)
        self.rows_clean += len(clean_chunk)
        self.null_total += int(clean_chunk.isnull().sum().sum())
        if "binary_label" in clean_chunk.columns:
            self.benign += int((clean_chunk["binary_label"] == 0).sum())
            self.botnet += int((clean_chunk["binary_label"] == 1).sum())
        if "family" in clean_chunk.columns:
            for fam, cnt in clean_chunk["family"].value_counts().items():
                self.families[str(fam)] += int(cnt)
        if "scenario_id" in clean_chunk.columns:
            for sc, cnt in clean_chunk["scenario_id"].value_counts().items():
                self.scenarios[str(sc)] += int(cnt)
        for k, v in step_counts.items():
            self.step_totals[k] += v


# ── Report writers ────────────────────────────────────────────────────────────

def save_reports(acc: QualityAccumulator, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cleaning_log.csv
    step_map = [
        ("remove_negative_bytes_pkts", "negative byte or packet count"),
        ("remove_negative_duration",   "negative duration"),
        ("remove_unresolved_labels",   "binary_label == -1"),
        ("drop_duplicates",            "exact duplicate rows within chunk"),
    ]
    with open(out_dir / "cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "rows_removed", "reason"])
        writer.writeheader()
        for step, reason in step_map:
            writer.writerow({"step": step,
                             "rows_removed": acc.step_totals.get(step, 0),
                             "reason": reason})

    # data_quality_summary.csv
    summary_df = pd.DataFrame([
        {"phase": "raw",   "rows": acc.rows_raw,   "benign": "?",        "botnet": "?"},
        {"phase": "clean", "rows": acc.rows_clean, "benign": acc.benign, "botnet": acc.botnet,
         "null_total": acc.null_total},
    ])
    summary_df.to_csv(out_dir / "data_quality_summary.csv", index=False)

    # data_quality_report.md
    removed = acc.rows_raw - acc.rows_clean
    lines = [
        "# Data Quality Report — CTU-13",
        "",
        f"**Raw rows:** {acc.rows_raw:,}",
        f"**Clean rows:** {acc.rows_clean:,}",
        f"**Rows removed:** {removed:,}",
        "",
        "## Cleaning Steps",
        "",
    ]
    for step, reason in step_map:
        n = acc.step_totals.get(step, 0)
        lines.append(f"- **{step}**: removed {n:,} rows ({reason})")
    lines += [
        "",
        "## Clean Dataset Label Distribution",
        "",
        f"- Benign: {acc.benign:,}",
        f"- Botnet: {acc.botnet:,}",
        "",
        "## Family Distribution",
        "",
    ]
    for fam, cnt in sorted(acc.families.items()):
        lines.append(f"- {fam}: {cnt:,}")
    (out_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Quality reports saved → %s", out_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean CTU-13 interim data (chunked).")
    parser.add_argument("--input",      default=str(INTERIM_DIR / "ctu13_raw.csv"))
    parser.add_argument("--output",     default=str(PROCESSED_DIR / "ctu13_flows.csv"))
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "dataset"))
    parser.add_argument("--chunksize",  type=int, default=CHUNKSIZE)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s\nRun extract_ctu13.py first.", input_path)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0  = time.time()
    acc = QualityAccumulator()
    chunk_i    = 0
    total_in   = 0
    total_out  = 0
    first_chunk = True

    logger.info("Chunked cleaning: %s  (chunksize=%d)", input_path, args.chunksize)

    reader = pd.read_csv(input_path, low_memory=False, chunksize=args.chunksize)

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        for raw_chunk in reader:
            chunk_i   += 1
            total_in  += len(raw_chunk)

            clean_df, step_counts = clean_chunk(raw_chunk.copy())
            acc.accumulate(raw_chunk, clean_df, step_counts)
            total_out += len(clean_df)

            # Write header only on first chunk
            clean_df.to_csv(out_f, index=False, header=first_chunk)
            first_chunk = False

            if chunk_i % 5 == 0 or chunk_i == 1:
                logger.info("  chunk %3d | in=%8d | out=%8d | running total=%d",
                            chunk_i, len(raw_chunk), len(clean_df), total_out)

    logger.info("Cleaning complete: %d → %d rows (removed %d) in %.1f s",
                total_in, total_out, total_in - total_out, time.time() - t0)

    save_reports(acc, Path(args.report_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
