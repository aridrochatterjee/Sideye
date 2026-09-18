#!/usr/bin/env python3
"""
Entry point: `python run.py` (or `python3 run.py`) starts SideEye.

Optional: pass a port number, e.g. `python run.py 9000`. Defaults to 8765.
Once it's running, open http://127.0.0.1:8765 in a browser.
"""

import sys

from backend.server import run

if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Ignoring invalid port '{sys.argv[1]}', using {port}.")
    run(port=port)
