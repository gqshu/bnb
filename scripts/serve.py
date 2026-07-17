"""Run the binaural-beat stream service on its fixed port.

    uv run scripts/serve.py            # then open http://127.0.0.1:8000

One stream, one port (bnb.server.PORT). Ctrl-C to stop.
"""

from bnb.server import main

if __name__ == "__main__":
    main()
