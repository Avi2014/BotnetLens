"""
src/demo_inject.py
Live Demo Injector — feeds real CTU-13 Neris flow features into live_detections.csv
so hardware_monitor (T4) shows realistic KNOWN_BOTNET detections.

Usage:
    # Single injection level (50% botnet):
    python src/demo_inject.py --injection 50 --flows 1000

    # Full sweep 0% → 100% (best for professor demo):
    python src/demo_inject.py --sweep --flows 500

    # Fast mode (no delay between flows):
    python src/demo_inject.py --injection 80 --flows 2000 --delay 0
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT
from logging_utils import get_logger
from cascade_inference import CascadeEngine

logger = get_logger("demo_inject")

# ── Paths ────────────────────────────────────────────────────────────────────
POOL_NERIS  = PROJECT_ROOT / "data" / "interim" / "pool_neris.csv"
POOL_BENIGN = PROJECT_ROOT / "data" / "interim" / "pool_benign_sample.csv"
OUT_CSV     = PROJECT_ROOT / "reports" / "live" / "live_detections.csv"

INJECTION_RATES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Fake IPs for demo — mimic RPi → Laptop hardware traffic
SRC_IP = "192.168.1.2"   # RPi
DST_IP = "192.168.1.3"   # Laptop


def load_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading pools from cache ...")
    if not POOL_NERIS.exists() or not POOL_BENIGN.exists():
        logger.error("Pool cache missing! Run inject_experiment.py first to build cache.")
        sys.exit(1)
    neris  = pd.read_csv(POOL_NERIS)
    benign = pd.read_csv(POOL_BENIGN)
    logger.info(f"Cache hit — Neris: {len(neris):,}  Benign: {len(benign):,}")
    return neris, benign


def build_mix(neris: pd.DataFrame, benign: pd.DataFrame,
              injection_pct: int, n_flows: int) -> pd.DataFrame:
    n_bot    = round(n_flows * injection_pct / 100)
    n_benign = n_flows - n_bot
    parts = []
    if n_bot > 0:
        parts.append(neris.sample(min(n_bot, len(neris)), replace=True))
    if n_benign > 0:
        parts.append(benign.sample(min(n_benign, len(benign)), replace=True))
    mix = pd.concat(parts, ignore_index=True).sample(frac=1).reset_index(drop=True)
    return mix


def add_network_columns(results_df: pd.DataFrame, mix_df: pd.DataFrame,
                        injection_pct: int) -> pd.DataFrame:
    """Add src/dst IP/port/proto columns to match live_detections.csv format."""
    import random
    n = len(results_df)
    results_df = results_df.copy()

    # Timestamps — spread over realistic time window
    base_ts = datetime.now(timezone.utc).timestamp()
    results_df["timestamp"] = [
        datetime.fromtimestamp(base_ts + i * 0.05, tz=timezone.utc).isoformat()
        for i in range(n)
    ]

    # IPs
    results_df["src_ip"]  = SRC_IP
    results_df["dst_ip"]  = DST_IP

    # Ports — botnet traffic goes to 6667 (IRC), benign to 80/443
    is_bot = (results_df["final_category"] == "KNOWN_BOTNET")
    results_df["src_port"] = [random.randint(32768, 60999) for _ in range(n)]
    results_df["dst_port"] = [
        6667 if bot else random.choice([80, 443, 53])
        for bot in is_bot
    ]
    results_df["proto"] = "tcp"

    return results_df


def write_to_live_csv(results_df: pd.DataFrame, append: bool = False) -> None:
    """Write results to live_detections.csv in the exact format hardware_monitor expects."""
    # Columns that live_detections.csv has (from realtime_sniffer.py)
    csv_cols = [
        "timestamp",
        "phase1_binary_prediction", "phase1_binary_confidence",
        "phase1_family_prediction", "phase1_family_confidence",
        "phase1_decision",
        "phase2_executed", "phase2_prediction",
        "phase2_anomaly_score", "phase2_threshold",
        "final_label", "final_category", "explanation",
        "src_ip", "dst_ip", "src_port", "dst_port", "proto",
    ]

    # Keep only columns that exist in results_df + our added ones
    out_cols = [c for c in csv_cols if c in results_df.columns]
    out_df = results_df[out_cols]

    mode   = "a" if append else "w"
    header = not append or not OUT_CSV.exists()
    out_df.to_csv(OUT_CSV, mode=mode, index=False, header=header)


def run_level(engine: CascadeEngine,
              neris: pd.DataFrame, benign: pd.DataFrame,
              injection_pct: int, n_flows: int,
              append: bool = False) -> dict:

    n_bot    = round(n_flows * injection_pct / 100)
    n_benign = n_flows - n_bot
    logger.info(f"--- Injection {injection_pct}%: {n_bot} Neris + {n_benign} Benign ---")

    mix_df     = build_mix(neris, benign, injection_pct, n_flows)
    results_df = engine.predict_batch(mix_df)
    results_df = add_network_columns(results_df, mix_df, injection_pct)

    write_to_live_csv(results_df, append=append)

    # Metrics
    cats  = results_df["final_category"].value_counts().to_dict()
    known = cats.get("KNOWN_BOTNET", 0)
    novel = cats.get("NOVEL_BEHAVIOR", 0)
    benig = cats.get("BENIGN", 0)
    tpr   = known / max(1, n_bot) * 100 if n_bot > 0 else 0.0
    fpr_denom = max(1, n_benign)
    fpr   = cats.get("KNOWN_BOTNET", 0) / fpr_denom * 100 if n_benign > 0 else 0.0

    logger.info(f"  TPR={tpr:.1f}%  KNOWN={known}  NOVEL={novel}  BENIGN={benig}")
    return {"injection_pct": injection_pct, "tpr": tpr, "known": known,
            "novel": novel, "benign": benig}


def print_summary(metrics: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f" {'Inj%':>5}  {'TPR%':>8}  {'KNOWN':>7}  {'NOVEL':>7}")
    print("-" * 60)
    for m in metrics:
        print(f" {m['injection_pct']:>5}  {m['tpr']:>7.1f}%  {m['known']:>7}  {m['novel']:>7}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="BotnetLens Live Demo Injector — shows KNOWN_BOTNET in hardware_monitor"
    )
    parser.add_argument("--injection", type=int, default=50,
                        choices=[0,10,20,30,40,50,60,70,80,90,100],
                        help="Single injection level (default 50%%)")
    parser.add_argument("--flows", type=int, default=1000,
                        help="Flows per level (default 1000)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to pause between sweep levels (default 2)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep 0%%→100%% with live updates to hardware_monitor")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing live_detections.csv (default: overwrite)")
    args = parser.parse_args()

    # Load engine & pools
    engine = CascadeEngine()
    engine.load()
    neris, benign = load_pools()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        logger.info("=" * 60)
        logger.info(f"SWEEP MODE  rates={INJECTION_RATES}  flows/step={args.flows}")
        logger.info("hardware_monitor (T4) will update after each step")
        logger.info("=" * 60)

        metrics = []
        for i, rate in enumerate(INJECTION_RATES):
            m = run_level(engine, neris, benign,
                          injection_pct=rate,
                          n_flows=args.flows,
                          append=(i > 0 or args.append))
            metrics.append(m)

            if rate < 100 and args.delay > 0:
                logger.info(f"Pausing {args.delay}s — watch T4 update ...")
                time.sleep(args.delay)

        print_summary(metrics)
        logger.info(f"Sweep complete! Results written to {OUT_CSV}")

    else:
        # Single level
        m = run_level(engine, neris, benign,
                      injection_pct=args.injection,
                      n_flows=args.flows,
                      append=args.append)
        logger.info(f"Done! Check hardware_monitor (T4) for KNOWN_BOTNET detections.")


if __name__ == "__main__":
    main()
