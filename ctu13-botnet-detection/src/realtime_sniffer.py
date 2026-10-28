"""
src/realtime_sniffer.py
Real-time packet sniffer and UBF feature extractor for BotnetLens.

Captures live packets on a network interface, aggregates them into
5-tuple flows, extracts the 16 UBF features, and feeds each completed
flow into the CascadeEngine for real-time botnet detection.

Works on Windows (Npcap) and Linux (libpcap).
Install scapy:  pip install scapy

Usage:
    # List interfaces first:
    python src/realtime_sniffer.py --list-ifaces

    # Sniff on interface (by index or name):
    python src/realtime_sniffer.py --iface "Ethernet" --duration 60
    python src/realtime_sniffer.py --iface 1 --output reports/live_detections.csv

    # Dry-run using a PCAP file (no live capture needed):
    python src/realtime_sniffer.py --pcap path/to/capture.pcap
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import threading
import queue
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT
from logging_utils import get_logger
from cascade_inference import CascadeEngine

logger = get_logger("realtime_sniffer")

# Flow timeout: treat a flow as complete if no new packets for this many seconds
FLOW_TIMEOUT_S = 5.0

# How often to flush timed-out flows (seconds)
FLUSH_INTERVAL_S = 2.0

# Protocol map
PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp"}

OUT_DIR = PROJECT_ROOT / "reports" / "live"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Flow tracker ──────────────────────────────────────────────────────────────

class FlowRecord:
    """Accumulates packet-level stats into a single flow record."""
    __slots__ = ("src_ip", "dst_ip", "src_port", "dst_port", "proto",
                 "start_ts", "last_ts", "pkt_count", "src_bytes", "dst_bytes",
                 "total_bytes", "total_packets", "flags")

    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto, ts):
        self.src_ip  = src_ip;  self.dst_ip  = dst_ip
        self.src_port = src_port; self.dst_port = dst_port
        self.proto   = proto
        self.start_ts = ts;   self.last_ts = ts
        self.pkt_count   = 0
        self.src_bytes   = 0;  self.dst_bytes  = 0
        self.total_bytes = 0;  self.total_packets = 0
        self.flags = set()

    def update(self, payload_len: int, direction: str, ts: float, flags=None):
        self.last_ts = ts
        self.pkt_count += 1
        self.total_packets += 1
        self.total_bytes   += payload_len
        if direction == "fwd":
            self.src_bytes += payload_len
        else:
            self.dst_bytes += payload_len
        if flags:
            self.flags.update(flags)

    def to_dict(self) -> dict:
        duration = max(self.last_ts - self.start_ts, 1e-6)
        return {
            "src_ip":          self.src_ip,
            "dst_ip":          self.dst_ip,
            "src_port":        self.src_port,
            "dst_port":        self.dst_port,
            "proto":           self.proto,
            "duration":        round(duration, 6),
            "total_packets":   self.total_packets,
            "total_bytes":     self.total_bytes,
            "src_bytes":       self.src_bytes,
            "dst_bytes":       self.dst_bytes,
            "timestamp":       datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat(),
        }


class FlowTracker:
    """5-tuple flow aggregator with timeout-based completion."""

    def __init__(self, timeout_s: float = FLOW_TIMEOUT_S):
        self._flows: dict[tuple, FlowRecord] = {}
        self._timeout = timeout_s
        self._lock = threading.Lock()

    def _key(self, src_ip, dst_ip, src_port, dst_port, proto):
        # Bidirectional: normalize key so A->B and B->A merge
        if (src_ip, src_port) > (dst_ip, dst_port):
            src_ip, dst_ip = dst_ip, src_ip
            src_port, dst_port = dst_port, src_port
        return (src_ip, dst_ip, src_port, dst_port, proto)

    def add_packet(self, src_ip, dst_ip, src_port, dst_port, proto,
                   payload_len, ts, flags=None):
        key = self._key(src_ip, dst_ip, src_port, dst_port, proto)
        direction = "fwd" if (src_ip, src_port) <= (dst_ip, dst_port) else "rev"
        with self._lock:
            if key not in self._flows:
                self._flows[key] = FlowRecord(src_ip, dst_ip, src_port, dst_port, proto, ts)
            self._flows[key].update(payload_len, direction, ts, flags)

    def flush_completed(self, now: float) -> list[dict]:
        """Return and remove flows that have timed out."""
        completed = []
        with self._lock:
            expired = [k for k, v in self._flows.items()
                       if (now - v.last_ts) >= self._timeout]
            for k in expired:
                completed.append(self._flows.pop(k).to_dict())
        return completed

    def flush_all(self) -> list[dict]:
        """Return all remaining flows (end of capture)."""
        with self._lock:
            flows = [v.to_dict() for v in self._flows.values()]
            self._flows.clear()
        return flows


# ── Packet callback ───────────────────────────────────────────────────────────

def _make_packet_handler(tracker: FlowTracker):
    """Returns a scapy packet callback that feeds the tracker."""
    try:
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.l2 import Ether
    except ImportError:
        logger.error("scapy not installed. Run: pip install scapy")
        sys.exit(1)

    def handler(pkt):
        try:
            if not pkt.haslayer(IP):
                return
            ip   = pkt[IP]
            ts   = float(pkt.time)
            proto_num = ip.proto
            proto = PROTO_MAP.get(proto_num, str(proto_num))

            src_ip = ip.src;  dst_ip = ip.dst
            src_port = 0;     dst_port = 0;  flags = None
            payload_len = len(ip.payload)

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = tcp.sport; dst_port = tcp.dport
                flags = {f for f in ["S","A","F","R","P","U"] if getattr(tcp.flags, f, False)}
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                src_port = udp.sport; dst_port = udp.dport

            tracker.add_packet(src_ip, dst_ip, src_port, dst_port, proto,
                                payload_len, ts, flags)
        except Exception:
            pass

    return handler


# ── Live sniffer thread ───────────────────────────────────────────────────────

def _sniff_thread(iface, duration, tracker, stop_event, pcap=None):
    try:
        from scapy.all import sniff
    except ImportError:
        logger.error("scapy not installed."); return

    handler = _make_packet_handler(tracker)
    kwargs = {"prn": handler, "store": False}
    if pcap:
        kwargs["offline"] = pcap
    else:
        kwargs["iface"]   = iface
        kwargs["timeout"] = duration
    logger.info("Sniffer started (iface=%s  pcap=%s  duration=%ss)", iface, pcap, duration)
    from scapy.all import sniff
    sniff(**kwargs, stop_filter=lambda _: stop_event.is_set())
    stop_event.set()
    logger.info("Sniffer stopped.")


# ── Main capture loop ─────────────────────────────────────────────────────────

def run_capture(iface, duration, engine: CascadeEngine, output_csv: Path,
                pcap=None, verbose=True):
    tracker    = FlowTracker()
    stop_event = threading.Event()
    results    = []

    # Start sniffer in background thread
    t = threading.Thread(target=_sniff_thread,
                         args=(iface, duration, tracker, stop_event, pcap),
                         daemon=True)
    t.start()

    # CSV writer
    fieldnames = None
    csv_file   = None
    writer     = None
    if output_csv:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")

    try:
        start = time.time()
        while not stop_event.is_set() or tracker._flows:
            time.sleep(FLUSH_INTERVAL_S)
            now = time.time()
            completed = tracker.flush_completed(now)
            if stop_event.is_set():
                completed += tracker.flush_all()

            for flow_dict in completed:
                try:
                    result = engine.predict_one(flow_dict)
                    result.update({k: flow_dict.get(k, "") for k in
                                   ["src_ip", "dst_ip", "src_port", "dst_port", "proto"]})
                    results.append(result)

                    if verbose:
                        cat   = result.get("final_category", "?")
                        color = {"KNOWN_BOTNET": "[!BOTNET!]",
                                 "NOVEL_BEHAVIOR": "[?NOVEL?]",
                                 "BENIGN": "[ OK    ]"}.get(cat, "[  ?   ]")
                        print(f"{color}  {flow_dict['src_ip']}:{flow_dict['src_port']} -> "
                              f"{flow_dict['dst_ip']}:{flow_dict['dst_port']}  "
                              f"proto={flow_dict['proto']}  "
                              f"conf={result.get('phase1_binary_confidence',0):.3f}  "
                              f"family={result.get('phase1_family_prediction','?')}")

                    if csv_file:
                        if writer is None:
                            fieldnames = list(result.keys())
                            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                            writer.writeheader()
                        writer.writerow({k: result.get(k, "") for k in fieldnames})
                        csv_file.flush()
                except Exception as exc:
                    logger.debug("Flow inference error: %s", exc)

            if duration and (time.time() - start) >= duration + 3:
                break
    finally:
        if csv_file:
            csv_file.close()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="BotnetLens real-time packet sniffer.")
    parser.add_argument("--iface",    default=None, help="Network interface name or index")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")
    parser.add_argument("--pcap",     default=None, help="Offline PCAP file instead of live capture")
    parser.add_argument("--output",   default=str(OUT_DIR / "live_detections.csv"))
    parser.add_argument("--binary-model",     default="lightgbm")
    parser.add_argument("--multiclass-model", default="lightgbm")
    parser.add_argument("--anomaly-model",    default="isolation_forest")
    parser.add_argument("--confidence",       type=float, default=0.70)
    parser.add_argument("--list-ifaces",      action="store_true")
    parser.add_argument("--quiet",            action="store_true")
    args = parser.parse_args(argv)

    if args.list_ifaces:
        try:
            from scapy.all import get_if_list
            for i, name in enumerate(get_if_list()):
                print(f"  [{i}] {name}")
        except ImportError:
            print("scapy not installed. Run: pip install scapy")
        return 0

    if not args.iface and not args.pcap:
        parser.error("Provide --iface <name> for live capture or --pcap <file> for offline.")

    engine = CascadeEngine(
        binary_model_name=args.binary_model,
        multiclass_model_name=args.multiclass_model,
        anomaly_model_name=args.anomaly_model,
        confidence_threshold=args.confidence,
    )
    try:
        engine.load()
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc); return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nBotnetLens Real-Time Sniffer")
    print(f"  Interface : {args.iface or 'OFFLINE: ' + str(args.pcap)}")
    print(f"  Duration  : {args.duration}s")
    print(f"  Output    : {out_path}")
    print(f"  Models    : binary={args.binary_model}  anomaly={args.anomaly_model}")
    print("-" * 60)

    results = run_capture(
        iface=args.iface, duration=args.duration, engine=engine,
        output_csv=out_path, pcap=args.pcap, verbose=not args.quiet,
    )

    if results:
        df = pd.DataFrame(results)
        cats = df["final_category"].value_counts()
        print("\n--- Capture Summary ---")
        print(f"  Total flows classified : {len(df)}")
        for cat, cnt in cats.items():
            print(f"  {cat:<22}: {cnt}")
        print(f"  Results saved -> {out_path}")
    else:
        print("No flows captured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
