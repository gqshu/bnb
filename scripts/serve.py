"""Run the binaural-beat stream service on its fixed port.

    uv run scripts/serve.py                 # then open http://127.0.0.1:8000
    uv run scripts/serve.py --host 0.0.0.0  # also reachable from the LAN

One stream, one port (bnb.server.PORT). Ctrl-C to stop. The bind address defaults to
loopback (bnb.server.HOST) — see there for why, and before opening it up.
"""

from bnb.server import main

if __name__ == "__main__":
    main()
