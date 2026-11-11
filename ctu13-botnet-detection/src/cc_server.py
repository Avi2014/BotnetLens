"""
src/cc_server.py
Simulated Neris-style IRC C&C Server — runs on Raspberry Pi 4.

Mimics the IRC Command & Control channel used by the Neris botnet:
  - Listens on TCP port 6667 (standard IRC port used by Neris)
  - Accepts bot registrations (NICK/USER)
  - Sends periodic PING commands to bots (heartbeat)
  - Logs every bot connection and command

⚠️  FOR ACADEMIC USE ONLY — isolated lab network, no internet connection.

Usage (on Raspberry Pi):
    python3 src/cc_server.py
    python3 src/cc_server.py --port 6667 --ping-interval 30
"""
from __future__ import annotations

import argparse
import socket
import threading
import time
import signal
import sys
from datetime import datetime

# ── config ──────────────────────────────────────────────────────────────────
HOST          = "0.0.0.0"   # listen on all interfaces
PORT          = 6667         # Neris uses standard IRC port
PING_INTERVAL = 30           # seconds between PING commands (Neris pattern)
LOG_FMT       = "[{ts}] [{tag}] {msg}"

bots: dict[str, dict] = {}   # nick → {conn, addr, joined}
lock  = threading.Lock()
_stop = threading.Event()

# ── logging ──────────────────────────────────────────────────────────────────
def log(tag: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(LOG_FMT.format(ts=ts, tag=tag.upper().center(8), msg=msg), flush=True)


# ── IRC-style send helper ────────────────────────────────────────────────────
def send(conn: socket.socket, msg: str):
    try:
        conn.sendall((msg + "\r\n").encode())
    except Exception:
        pass


# ── bot session handler ──────────────────────────────────────────────────────
def handle_bot(conn: socket.socket, addr: tuple):
    ip, port = addr
    nick = None
    log("CONNECT", f"Bot connected from {ip}:{port}")

    try:
        conn.settimeout(120)
        buf = ""
        while not _stop.is_set():
            try:
                data = conn.recv(512).decode(errors="ignore")
            except socket.timeout:
                # Send PING to keep connection alive (Neris heartbeat pattern)
                if nick:
                    send(conn, f"PING :cc-server")
                    log("PING", f"→ {nick} ({ip})")
                continue
            except Exception:
                break

            if not data:
                break

            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                # Parse IRC commands
                if line.startswith("NICK "):
                    nick = line.split()[1]
                    log("NICK", f"{ip} registered as '{nick}'")
                    send(conn, f":cc-server 001 {nick} :Welcome to the botnet C&C")
                    send(conn, f":cc-server 376 {nick} :End of /MOTD command")
                    with lock:
                        bots[nick] = {"conn": conn, "addr": ip, "joined": datetime.now()}

                elif line.startswith("USER "):
                    pass  # Accept silently

                elif line.startswith("JOIN "):
                    channel = line.split()[1] if len(line.split()) > 1 else "#botnet"
                    log("JOIN", f"{nick} joined {channel}")
                    send(conn, f":{nick}!bot@{ip} JOIN {channel}")
                    # Send C&C command after join
                    send(conn, f":cc-server PRIVMSG {channel} :START_RECON")

                elif line.startswith("PONG"):
                    log("PONG", f"← {nick} alive ✓")

                elif line.startswith("PRIVMSG"):
                    log("MSG", f"← {nick}: {line}")

                elif line.startswith("QUIT"):
                    log("QUIT", f"{nick} disconnected")
                    break

                else:
                    pass  # silently ignore unknown commands

    except Exception as e:
        log("ERROR", f"{ip}: {e}")
    finally:
        if nick:
            with lock:
                bots.pop(nick, None)
            log("DISC", f"Bot '{nick}' ({ip}) disconnected | Active bots: {len(bots)}")
        conn.close()


# ── periodic status printer ──────────────────────────────────────────────────
def status_printer():
    while not _stop.is_set():
        time.sleep(10)
        with lock:
            log("STATUS", f"Active bots: {len(bots)} | {list(bots.keys())}")


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="BotnetLens Simulated C&C Server")
    parser.add_argument("--port",          type=int, default=PORT)
    parser.add_argument("--ping-interval", type=int, default=PING_INTERVAL)
    args = parser.parse_args(argv)

    def shutdown(sig, frame):
        log("SERVER", "Shutting down C&C server ...")
        _stop.set()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    log("SERVER", f"BotnetLens Simulated C&C Server starting on {HOST}:{args.port}")
    log("SERVER", "FOR ACADEMIC USE ONLY — isolated lab network")

    threading.Thread(target=status_printer, daemon=True).start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, args.port))
    server.listen(50)
    log("SERVER", f"Listening on port {args.port} ...")

    while not _stop.is_set():
        try:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_bot, args=(conn, addr), daemon=True)
            t.start()
        except Exception:
            break

    server.close()


if __name__ == "__main__":
    main()

