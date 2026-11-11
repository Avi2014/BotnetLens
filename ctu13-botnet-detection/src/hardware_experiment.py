"""
src/hardware_experiment.py
BotnetLens Hardware Testbed Experiment Controller — runs on Monitor PC.

Coordinates the full injection sweep on real hardware:
  - Connects to Raspberry Pi via SSH to start/stop bot_agent at each level
  - Captures live traffic via realtime_sniffer.py
  - Runs cascade inference on captured flows
  - Records TPR/FPR/KNOWN/NOVEL at each injection level (0%–100%)
  - Saves results to reports/hardware/

Requires:
  - Raspberry Pi running cc_server.py
  - Managed switch with port mirroring configured
  - paramiko (pip install paramiko) for SSH control of RPi

Usage (on Monitor PC):
    python src/hardware_experiment.py --rpi-ip 192.168.1.2 --iface Ethernet
    python src/hardware_experiment.py --rpi-ip 192.168.1.2 --iface Ethernet --duration 60
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
SRC      = ROOT / "src"
OUT_DIR  = ROOT / "reports" / "hardware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INJECTION_RATES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_one_level(rate: int, rpi_ip: str, rpi_user: str, rpi_pass: str,
                   iface: str, duration: int) -> dict:
    """
    1. SSH into RPi → start bot_agent.py at given injection rate
    2. Capture live traffic for `duration` seconds
    3. Stop bot_agent on RPi
    4. Parse captured predictions → compute TPR / FPR
    """
    log(f"{'='*60}")
    log(f"Injection {rate}%: starting bot_agent on RPi ...")

    # ── Start bot agent on RPi via SSH ──────────────────────────────────────
    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(rpi_ip, username=rpi_user, password=rpi_pass, timeout=10)

        bot_cmd = (
            f"python3 ~/botnetlens/src/bot_agent.py "
            f"--cc-host {rpi_ip} --injection {rate} &"
        )
        ssh.exec_command(bot_cmd)
        log(f"  Bot agent started on RPi at {rate}% injection")
        time.sleep(3)  # let bots connect to C&C

    except ImportError:
        log("  [WARN] paramiko not installed — start bot_agent.py on RPi manually")
        log(f"  Run on RPi: python3 src/bot_agent.py --cc-host {rpi_ip} --injection {rate}")
        input("  Press Enter when bot_agent is running on RPi ...")
        ssh = None

    # ── Capture live traffic ──────────────────────────────────────────────────
    cap_csv = OUT_DIR / f"hw_run_{rate:03d}pct_predictions.csv"
    log(f"  Capturing live traffic on '{iface}' for {duration}s ...")

    sniffer_proc = subprocess.Popen(
        [
            sys.executable,
            str(SRC / "realtime_sniffer.py"),
            "--iface", iface,
            "--duration", str(duration),
            "--out", str(cap_csv),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sniffer_proc.wait()
    log(f"  Capture complete → {cap_csv.name}")

    # ── Stop bot agent on RPi ─────────────────────────────────────────────────
    if ssh:
        ssh.exec_command("pkill -f bot_agent.py")
        ssh.close()
        log(f"  Bot agent stopped on RPi")
    else:
        input("  Press Enter after stopping bot_agent.py on RPi ...")

    # ── Parse results ─────────────────────────────────────────────────────────
    if not cap_csv.exists():
        log(f"  [WARN] No capture file found — skipping level {rate}%")
        return {}

    import pandas as pd
    df = pd.read_csv(cap_csv)
    if df.empty:
        log(f"  [WARN] Empty capture — skipping level {rate}%")
        return {}

    # Expected columns from realtime_sniffer: prediction, true_label (if labelled)
    # In hardware mode we estimate by traffic source port
    total       = len(df)
    known_bot   = (df.get("prediction", pd.Series()) == "KNOWN_BOTNET").sum()
    novel_beh   = (df.get("prediction", pd.Series()) == "NOVEL_BEHAVIOR").sum()
    benign_cnt  = (df.get("prediction", pd.Series()) == "BENIGN").sum()

    # TPR: bot flows caught / total bot flows
    # In hardware we identify bot flows by dst_port == 6667
    bot_flows    = df[df.get("dst_port", pd.Series(0)) == 6667] if "dst_port" in df.columns else pd.DataFrame()
    benign_flows = df[df.get("dst_port", pd.Series(0)) != 6667] if "dst_port" in df.columns else pd.DataFrame()

    tpr = (known_bot + novel_beh) / max(len(bot_flows), 1) if len(bot_flows) > 0 else 0.0
    fpr = (df[df.get("dst_port", pd.Series(0)) != 6667].get("prediction", pd.Series())
               .isin(["KNOWN_BOTNET", "NOVEL_BEHAVIOR"]).sum()) / max(len(benign_flows), 1)

    m = {
        "injection_pct":        rate,
        "total_flows":          total,
        "tpr":                  round(tpr, 4),
        "fpr":                  round(fpr, 4),
        "known_botnet_count":   int(known_bot),
        "novel_behavior_count": int(novel_beh),
        "benign_count":         int(benign_cnt),
    }
    log(f"  TPR={tpr*100:.1f}%  FPR={fpr*100:.1f}%  "
        f"KNOWN={known_bot}  NOVEL={novel_beh}  TOTAL={total}")
    return m


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="BotnetLens Hardware Experiment")
    parser.add_argument("--rpi-ip",   default="192.168.1.2",
                        help="IP address of Raspberry Pi on lab network")
    parser.add_argument("--rpi-user", default="pi",
                        help="SSH username for Raspberry Pi")
    parser.add_argument("--rpi-pass", default="raspberry",
                        help="SSH password for Raspberry Pi")
    parser.add_argument("--iface",    default="Ethernet",
                        help="Network interface on Monitor PC to sniff")
    parser.add_argument("--duration", type=int, default=60,
                        help="Seconds to capture traffic per injection level")
    parser.add_argument("--rates",    nargs="+", type=int, default=INJECTION_RATES,
                        help="Injection rates to test (e.g. 0 10 50 100)")
    args = parser.parse_args(argv)

    log("BotnetLens Hardware Testbed Experiment")
    log(f"RPi: {args.rpi_ip} | Interface: {args.iface} | Duration/level: {args.duration}s")
    log(f"Injection rates: {args.rates}")
    log("Make sure:")
    log("  1. RPi is connected to switch Port 1")
    log("  2. Monitor PC is connected to switch Port 2")
    log("  3. Port mirror: Port 1 → Port 2 is configured in switch web UI")
    log("  4. cc_server.py is running on RPi: python3 src/cc_server.py")
    input("\nPress Enter to begin experiment ...")

    all_metrics = []
    for rate in args.rates:
        m = run_one_level(
            rate, args.rpi_ip, args.rpi_user, args.rpi_pass,
            args.iface, args.duration,
        )
        if m:
            all_metrics.append(m)
        time.sleep(5)  # cooldown between levels

    # ── Save summary CSV ──────────────────────────────────────────────────────
    if all_metrics:
        out_csv = OUT_DIR / "hardware_sweep_results.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        log(f"\nResults → {out_csv}")

        # Print summary table
        print("\n" + "=" * 65)
        print(f"{'Inj%':>5}  {'TPR%':>8}  {'FPR%':>8}  {'KNOWN':>7}  {'NOVEL':>7}  {'TOTAL':>7}")
        print("-" * 65)
        for m in all_metrics:
            print(f"{m['injection_pct']:>5}  {m['tpr']*100:>7.1f}%  {m['fpr']*100:>7.1f}%  "
                  f"{m['known_botnet_count']:>7}  {m['novel_behavior_count']:>7}  "
                  f"{m['total_flows']:>7}")
        print("=" * 65)

    log("Hardware experiment complete.")


if __name__ == "__main__":
    main()

