"""
src/hardware_monitor.py
BotnetLens Hardware Monitor — live terminal dashboard.

Reads the live_detections.csv in real-time (written by realtime_sniffer.py
or inject_experiment.py) and renders a constantly-updating terminal dashboard
showing:
  - Total flows classified
  - Detection counts by category
  - Rolling detection rate (last 60 seconds)
  - Recent alerts (KNOWN_BOTNET / NOVEL_BEHAVIOR)
  - Inference latency stats

Works standalone: can also tail the injection sweep results for demo.

Usage:
    # Monitor live sniffer output:
    python src/hardware_monitor.py

    # Monitor a specific CSV:
    python src/hardware_monitor.py --input reports/injection/run_050pct_predictions.csv

    # Dashboard refresh rate:
    python src/hardware_monitor.py --refresh 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT
from logging_utils import get_logger

logger = get_logger("hardware_monitor")

DEFAULT_INPUT  = PROJECT_ROOT / "reports" / "live" / "live_detections.csv"
REFRESH_S      = 2
ALERT_WINDOW_S = 60     # rolling window for detection rate
MAX_ALERTS     = 15     # max recent alerts to show

# ANSI colors
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CATEGORY_COLORS = {
    "KNOWN_BOTNET":   RED,
    "NOVEL_BEHAVIOR": YELLOW,
    "BENIGN":         GREEN,
    "UNCERTAIN":      DIM,
    "INVALID_INPUT":  DIM,
}


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _bar(value: float, width: int = 30, fill="█", empty="░") -> str:
    filled = round(value * width)
    return fill * filled + empty * (width - filled)


def _render_dashboard(df: pd.DataFrame, start_time: float, input_path: Path):
    _clear()
    now = time.time()
    elapsed = now - start_time

    total = len(df)
    cat_counts = df["final_category"].value_counts().to_dict() if total > 0 else {}

    known  = cat_counts.get("KNOWN_BOTNET",   0)
    novel  = cat_counts.get("NOVEL_BEHAVIOR", 0)
    benign = cat_counts.get("BENIGN",         0)
    unc    = cat_counts.get("UNCERTAIN",       0)

    tpr = (known + novel) / max(total, 1)

    # Rolling window
    recent_alerts = 0
    if total > 0 and "timestamp" in df.columns:
        try:
            ts_col = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            cutoff  = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=ALERT_WINDOW_S)
            recent_mask = ts_col > cutoff
            recent_df   = df[recent_mask]
            recent_alerts = recent_df["final_category"].isin(
                ["KNOWN_BOTNET", "NOVEL_BEHAVIOR"]).sum()
        except Exception:
            pass

    # Banner
    print(f"{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          BotnetLens  ·  Real-Time Detection Dashboard        ║")
    print("╚══════════════════════════════════════════════════════════════╝" + RESET)
    print(f"  {DIM}Input  : {input_path}{RESET}")
    print(f"  {DIM}Uptime : {int(elapsed//60):02d}m {int(elapsed%60):02d}s  |  "
          f"Last refresh: {datetime.now().strftime('%H:%M:%S')}{RESET}")
    print()

    # Summary stats
    print(f"{BOLD}  ── Flow Statistics ──────────────────────────────────────────{RESET}")
    print(f"  Total Flows Classified : {BOLD}{total:>8,}{RESET}")
    print(f"  Rolling Alerts (60s)   : {BOLD}{YELLOW}{recent_alerts:>8,}{RESET}")
    print()

    # Category breakdown with bars
    print(f"{BOLD}  ── Cascade Detection Breakdown ──────────────────────────────{RESET}")
    categories = [
        ("KNOWN_BOTNET",   known,  RED),
        ("NOVEL_BEHAVIOR", novel,  YELLOW),
        ("BENIGN",         benign, GREEN),
        ("UNCERTAIN",      unc,    DIM),
    ]
    for cat, cnt, color in categories:
        pct  = cnt / max(total, 1)
        bar  = _bar(pct, width=25)
        print(f"  {color}{cat:<22}{RESET}  {bar}  "
              f"{color}{cnt:>6,}  ({pct*100:5.1f}%){RESET}")
    print()

    # Overall detection gauge
    print(f"{BOLD}  ── Overall Detection Rate ───────────────────────────────────{RESET}")
    gauge_color = RED if tpr > 0.5 else YELLOW if tpr > 0.1 else GREEN
    print(f"  Detection Rate  {gauge_color}{_bar(tpr, width=35)}  {tpr*100:5.1f}%{RESET}")
    print()

    # Recent alerts table
    print(f"{BOLD}  ── Recent Alerts (last {MAX_ALERTS}) ──────────────────────────────{RESET}")
    if total > 0:
        alert_df = df[df["final_category"].isin(["KNOWN_BOTNET", "NOVEL_BEHAVIOR"])].tail(MAX_ALERTS)
        if len(alert_df) == 0:
            print(f"  {DIM}  No alerts yet.{RESET}")
        else:
            print(f"  {'Category':<22}  {'Family':<12}  {'Confidence':>10}  {'Source IP':<18}  {'Dst IP':<18}")
            print(f"  {'─'*22}  {'─'*12}  {'─'*10}  {'─'*18}  {'─'*18}")
            for _, row in alert_df.iterrows():
                cat    = str(row.get("final_category", "?"))
                family = str(row.get("phase1_family_prediction", "?"))[:12]
                conf   = row.get("phase1_binary_confidence", 0)
                src    = str(row.get("src_ip", "?"))[:18]
                dst    = str(row.get("dst_ip", "?"))[:18]
                color  = CATEGORY_COLORS.get(cat, "")
                print(f"  {color}{cat:<22}{RESET}  {family:<12}  {conf:>10.4f}  {src:<18}  {dst:<18}")
    else:
        print(f"  {DIM}  Waiting for data…{RESET}")

    print()
    print(f"  {DIM}Press Ctrl+C to stop monitoring.{RESET}")


def _load_csv_safe(path: Path) -> pd.DataFrame:
    """Load CSV, tolerate partial writes."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def main(argv=None):
    parser = argparse.ArgumentParser(description="BotnetLens live terminal dashboard.")
    parser.add_argument("--input",   default=str(DEFAULT_INPUT),
                        help="CSV file to monitor (written by realtime_sniffer.py or inject_experiment.py)")
    parser.add_argument("--refresh", type=float, default=REFRESH_S,
                        help="Dashboard refresh interval in seconds")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    start_time = time.time()

    print(f"BotnetLens Hardware Monitor")
    print(f"Watching: {input_path}")
    print(f"Refresh : every {args.refresh}s")
    print("Waiting for data…")

    try:
        while True:
            df = _load_csv_safe(input_path) if input_path.exists() else pd.DataFrame()
            _render_dashboard(df, start_time, input_path)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print(f"\nMonitor stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
