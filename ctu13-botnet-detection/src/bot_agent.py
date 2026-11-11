"""
src/bot_agent.py
Simulated Neris Bot Agent — runs on Raspberry Pi 4.

Mimics exactly how the Neris botnet behaves:
  - Connects to C&C server on port 6667 (IRC)
  - Registers with NICK/USER (like real IRC client)
  - JOINs the botnet channel
  - Sends periodic PING/PONG heartbeats (Neris C&C keepalive pattern)
  - Generates traffic that matches Neris flow features:
      • Short duration flows
      • Small fixed payload sizes (~100-400 bytes)
      • Regular timing intervals (every 30s)
      • TCP on port 6667

This generates REAL network traffic on the actual hardware interface.
The managed switch port mirror copies it to the monitor PC's interface.
BotnetLens realtime_sniffer.py captures and detects it.

⚠️  FOR ACADEMIC USE ONLY — isolated lab network.

Usage (on Raspberry Pi):
    python3 src/bot_agent.py --cc-host 192.168.1.2 --injection 30
    python3 src/bot_agent.py --cc-host 192.168.1.2 --injection 50 --bots 5
"""
from __future__ import annotations

import argparse
import random
import socket
import threading
import time
import signal
import sys
from datetime import datetime

# ── config ──────────────────────────────────────────────────────────────────
CC_PORT       = 6667
HEARTBEAT_INT = 30      # seconds between pings (Neris pattern)
RECONNECT_INT = 10      # seconds before reconnect attempt

_stop = threading.Event()

def log(tag: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag.upper().center(8)}] {msg}", flush=True)


# ── single bot session ───────────────────────────────────────────────────────
def run_bot(bot_id: int, cc_host: str, cc_port: int):
    """Simulates one Neris bot connecting to C&C and sending heartbeats."""
    nick = f"bot{bot_id:03d}"
    channel = "#update"     # Neris bots join this kind of channel

    while not _stop.is_set():
        try:
            log(nick, f"Connecting to C&C at {cc_host}:{cc_port} ...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((cc_host, cc_port))
            sock.settimeout(None)

            # IRC registration — exactly like Neris bot behaviour
            sock.sendall(f"NICK {nick}\r\n".encode())
            sock.sendall(f"USER {nick} 0 * :BotAgent\r\n".encode())
            time.sleep(1)
            sock.sendall(f"JOIN {channel}\r\n".encode())
            log(nick, f"Connected and joined {channel} ✓")

            # Heartbeat loop
            last_ping = time.time()
            buf = ""
            while not _stop.is_set():
                # Send PING heartbeat every HEARTBEAT_INT seconds
                if time.time() - last_ping >= HEARTBEAT_INT:
                    sock.sendall(f"PING :cc-server\r\n".encode())
                    log(nick, f"→ PING (heartbeat)")
                    last_ping = time.time()

                # Small data burst to simulate C&C command exchange
                # Neris sends small fixed-size payloads (100–400 bytes)
                payload_size = random.randint(100, 400)
                dummy_payload = "A" * payload_size
                try:
                    sock.sendall(
                        f"PRIVMSG {channel} :{dummy_payload[:50]}\r\n".encode()
                    )
                except Exception:
                    break

                # Listen for server responses (PING from server → reply PONG)
                sock.setblocking(False)
                try:
                    data = sock.recv(512).decode(errors="ignore")
                    if "PING" in data:
                        sock.sendall(b"PONG :cc-server\r\n")
                        log(nick, "← PING from server → PONG sent")
                except BlockingIOError:
                    pass
                finally:
                    sock.setblocking(True)

                time.sleep(HEARTBEAT_INT + random.uniform(-5, 5))

        except ConnectionRefusedError:
            log(nick, f"C&C not reachable — retrying in {RECONNECT_INT}s")
        except Exception as e:
            log(nick, f"Error: {e} — reconnecting in {RECONNECT_INT}s")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        if not _stop.is_set():
            time.sleep(RECONNECT_INT)


# ── benign traffic generator ─────────────────────────────────────────────────
def run_benign(target_host: str):
    """
    Generates benign-looking HTTP/DNS traffic alongside botnet traffic.
    This simulates a real mixed network (not pure attack traffic).
    """
    benign_targets = [
        ("8.8.8.8", 53),       # DNS (if internet available)
        (target_host, 80),     # HTTP to monitor PC
        (target_host, 443),    # HTTPS simulation
    ]
    while not _stop.is_set():
        try:
            host, port = random.choice(benign_targets)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((host, port))
            # Simulate HTTP GET request (benign)
            req = f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
            sock.sendall(req.encode())
            try:
                sock.recv(1024)
            except Exception:
                pass
            sock.close()
        except Exception:
            pass  # benign traffic failures are expected and ignored
        time.sleep(random.uniform(5, 20))  # random benign browsing pace


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    global HEARTBEAT_INT  # declare before any use of HEARTBEAT_INT in this scope

    parser = argparse.ArgumentParser(description="BotnetLens Simulated Bot Agent")
    parser.add_argument("--cc-host",   required=True,
                        help="IP address of C&C server (Raspberry Pi IP)")
    parser.add_argument("--cc-port",   type=int, default=CC_PORT)
    parser.add_argument("--bots",      type=int, default=1,
                        help="Number of simultaneous bot threads")
    parser.add_argument("--injection", type=int, default=10,
                        choices=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help="Injection percentage — controls bot vs benign thread ratio")
    parser.add_argument("--heartbeat", type=int, default=HEARTBEAT_INT,
                        help="Seconds between C&C heartbeats")
    args = parser.parse_args(argv)

    HEARTBEAT_INT = args.heartbeat

    def shutdown(sig, frame):
        log("AGENT", "Stopping all bots ...")
        _stop.set()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    # Calculate threads from injection percentage
    total_threads = 10
    n_bot_threads    = round(total_threads * args.injection / 100)
    n_benign_threads = total_threads - n_bot_threads

    log("AGENT", f"BotnetLens Bot Agent starting")
    log("AGENT", f"C&C server: {args.cc_host}:{args.cc_port}")
    log("AGENT", f"Injection level: {args.injection}%")
    log("AGENT", f"Bot threads: {n_bot_threads}  |  Benign threads: {n_benign_threads}")
    log("AGENT", "FOR ACADEMIC USE ONLY — isolated lab network")

    threads = []

    # Start bot threads
    for i in range(n_bot_threads):
        t = threading.Thread(
            target=run_bot,
            args=(i + 1, args.cc_host, args.cc_port),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)   # stagger startup slightly (realistic)

    # Start benign threads
    for _ in range(n_benign_threads):
        t = threading.Thread(
            target=run_benign,
            args=(args.cc_host,),
            daemon=True,
        )
        t.start()
        threads.append(t)

    log("AGENT", f"All {len(threads)} threads running. Press Ctrl+C to stop.")

    # Keep alive
    while not _stop.is_set():
        time.sleep(5)
        active = sum(1 for t in threads if t.is_alive())
        log("AGENT", f"Threads alive: {active}/{len(threads)} | "
                     f"Injection: {args.injection}% | Heartbeat: {HEARTBEAT_INT}s")


if __name__ == "__main__":
    main()

