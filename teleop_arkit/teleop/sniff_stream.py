"""Stream sniffer — capture what ZIG SIM (or any phone app) actually sends.

We use this once to see ZIG SIM's real ARKit payload format before writing the
parser (arkit_receiver.py), so we match the wire format exactly instead of
guessing. No deps beyond the stdlib.

Usage (run on the Ubuntu machine; point ZIG SIM at this host's IP + port):
    python3 -m teleop_arkit.teleop.sniff_stream                 # UDP on :50000
    python3 -m teleop_arkit.teleop.sniff_stream --port 50000
    python3 -m teleop_arkit.teleop.sniff_stream --proto tcp     # if ZIG SIM uses TCP

In ZIG SIM: choose protocol UDP (or TCP), data format JSON, set the destination
to this machine's IP and the same port, enable ARKit, and Start.
"""

from __future__ import annotations

import argparse
import json
import socket


def _show(data: bytes, src) -> None:
    print(f"\n--- {len(data)} bytes from {src} ---")
    text = None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    if text is not None:
        try:  # JSON? pretty-print + summarize top-level keys.
            obj = json.loads(text)
            print("JSON payload:")
            print(json.dumps(obj, indent=2)[:4000])
            if isinstance(obj, dict):
                print("top-level keys:", list(obj.keys()))
        except json.JSONDecodeError:
            # Not JSON — could be OSC text or CSV. Show it raw.
            print("text payload:", text[:2000])
            if text[:1] == "/" or "/ZIG" in text:
                print("(looks like OSC — addresses start with '/')")
    else:
        # Binary (likely OSC bundle). Show hex + the ASCII bits (OSC addresses).
        print("binary payload (hex, first 256B):", data[:256].hex())
        ascii_bits = "".join(chr(b) if 32 <= b < 127 else "." for b in data[:256])
        print("ascii:", ascii_bits)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proto", choices=["udp", "tcp"], default="udp")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 = all interfaces).")
    p.add_argument("--port", type=int, default=50000)
    p.add_argument("--count", type=int, default=5, help="Packets/reads to print, then exit (0 = forever).")
    args = p.parse_args()

    print(f"[sniff] listening {args.proto.upper()} on {args.host}:{args.port}  "
          f"(point ZIG SIM here; Ctrl-C to stop)")

    n = 0
    if args.proto == "udp":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host, args.port))
        while args.count == 0 or n < args.count:
            data, src = s.recvfrom(65535)
            _show(data, src)
            n += 1
    else:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)
        print("[sniff] waiting for a TCP connection...")
        conn, src = srv.accept()
        print(f"[sniff] connected: {src}")
        with conn:
            while args.count == 0 or n < args.count:
                data = conn.recv(65535)
                if not data:
                    print("[sniff] connection closed")
                    break
                _show(data, src)
                n += 1

    print("\n[sniff] done — paste the payload above so we can write the parser.")


if __name__ == "__main__":
    main()
