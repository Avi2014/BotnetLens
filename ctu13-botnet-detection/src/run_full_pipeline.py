"""
src/run_full_pipeline.py
Master pipeline runner — executes all phases end to end.

Usage:
    python src/run_full_pipeline.py [--data-dir data/raw] [--skip-extraction]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import PROJECT_ROOT, PROCESSED_DIR, SPLITS_DIR, HOLDOUT_FAMILY, RANDOM_SEED
from logging_utils import get_logger

logger = get_logger("run_full_pipeline",
                    log_file=PROJECT_ROOT / "reports" / "pipeline.log")


def _run_script(script: str, args: list[str] = [], label: str = "") -> bool:
    """Run a src script via the same Python interpreter."""
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / script)] + args
    logger.info("=" * 60)
    logger.info("STEP: %s", label or script)
    logger.info("CMD:  %s", " ".join(cmd))
    t0 = time.time()
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("  [OK]  %s  done in %.1f s", label, elapsed)
        return True
    else:
        logger.error("  [FAIL] %s  FAILED (exit %d) after %.1f s", label, result.returncode, elapsed)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full CTU-13 botnet detection pipeline.")
    parser.add_argument("--data-dir",        default="data/raw")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extract step (raw CSV already exists)")
    parser.add_argument("--skip-cleaning",   action="store_true",
                        help="Skip clean step (processed CSV already exists)")
    parser.add_argument("--skip-training",   action="store_true",
                        help="Skip model training (report only)")
    parser.add_argument("--strategy",        default="family_holdout",
                        choices=["random", "scenario_aware", "family_holdout"])
    parser.add_argument("--holdout-family",  default=HOLDOUT_FAMILY)
    parser.add_argument("--seed",            type=int, default=RANDOM_SEED)
    parser.add_argument("--max-rows",        type=int, default=None,
                        help="Limit rows per scenario (dev/test mode)")
    args = parser.parse_args(argv)

    pipeline_start = time.time()
    steps_ok  = []
    steps_fail = []

    def _step(script, step_args, label):
        ok = _run_script(script, step_args, label)
        (steps_ok if ok else steps_fail).append(label)
        return ok

    # ── Phase 0: Extract ──────────────────────────────────────────────────────
    if not args.skip_extraction:
        extract_args = ["--data-dir", args.data_dir]
        if args.max_rows:
            extract_args += ["--max-rows", str(args.max_rows)]
        if not _step("extract_ctu13.py", extract_args, "Phase 0: Extract CTU-13"):
            logger.error("Extraction failed. Pipeline cannot continue.")
            return 1
    else:
        logger.info("Skipping extraction (--skip-extraction) — using existing ctu13_raw.csv.")

    # ── Phase 0: Clean ────────────────────────────────────────────────────────
    if not args.skip_cleaning:
        if not _step("clean_ctu13.py", [], "Phase 0: Clean CTU-13"):
            logger.error("Cleaning failed. Pipeline cannot continue.")
            return 1
    else:
        logger.info("Skipping cleaning (--skip-cleaning) — using existing ctu13_flows.csv.")
        processed_path = PROCESSED_DIR / "ctu13_flows.csv"
        if not processed_path.exists():
            logger.error("Processed data not found: %s", processed_path)
            return 1

    # ── Phase 0: Feature analysis (family signatures) ─────────────────────────
    _step("family_feature_analysis.py", [], "Phase 0: Family Feature Analysis")

    # ── Phase 0: Split ────────────────────────────────────────────────────────
    split_args = [
        "--strategy", args.strategy,
        "--holdout-family", args.holdout_family,
        "--seed", str(args.seed),
    ]
    if not _step("split_data.py", split_args, "Phase 0: Data Splitting"):
        logger.error("Splitting failed. Pipeline cannot continue.")
        return 1

    if args.skip_training:
        logger.info("Skipping training (--skip-training).")
    else:
        # ── Phase 1: Binary training ──────────────────────────────────────────
        if not _step("train_phase1_binary.py",
                     ["--seed", str(args.seed)],
                     "Phase 1: Binary Classification"):
            steps_fail.append("Phase 1: Binary")

        # ── Phase 1: Multiclass training ──────────────────────────────────────
        if not _step("train_phase1_multiclass.py",
                     ["--seed", str(args.seed)],
                     "Phase 1: Multiclass Classification"):
            steps_fail.append("Phase 1: Multiclass")

        # ── Phase 2: Anomaly training ─────────────────────────────────────────
        if not _step("train_phase2_anomaly.py",
                     ["--seed", str(args.seed)],
                     "Phase 2: Anomaly Detection"):
            steps_fail.append("Phase 2: Anomaly")

    # ── Visualizations ────────────────────────────────────────────────────────
    _step("generate_visualizations.py", [], "Generate Visualizations")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1f s", elapsed)
    logger.info("Steps succeeded: %d", len(steps_ok))
    logger.info("Steps failed:    %d", len(steps_fail))
    if steps_fail:
        logger.warning("Failed steps: %s", steps_fail)

    summary = {
        "total_time_s": round(elapsed, 2),
        "holdout_family": args.holdout_family,
        "strategy": args.strategy,
        "steps_ok": steps_ok,
        "steps_failed": steps_fail,
    }
    summary_path = PROJECT_ROOT / "reports" / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Pipeline summary → %s", summary_path)

    return 0 if not steps_fail else 1


if __name__ == "__main__":
    sys.exit(main())
