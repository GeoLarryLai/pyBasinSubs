# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyBasinSubs browser app (``pybasinsubs-app``; aliases ``pybasinsubs-ui`` and ``pybasinsubs-tracer``).

A small FastAPI server on 127.0.0.1 and one plain HTML/JS page, used in any browser (same design as the
pyPetrograph app). Three jobs: 1-D column (backstripping, geohistory, maturity), 2-D section (CSV import,
restoration, Monte Carlo envelopes) and the fault editor. All science goes through ``pybasinsubs.app_core`` and
``pybasinsubs.fault_tracer``, the same calls the notebooks use.

    pybasinsubs-app [--port N] [--no-browser] [--page column|section|faults]
    pybasinsubs-tracer                      # = pybasinsubs-app --page faults
"""
from __future__ import annotations

import argparse
import os
import sys

PAGES = ("column", "section", "faults")


def main(argv=None, default_page: str = "") -> int:
    from .server import DEFAULT_PORT, HOST
    ap = argparse.ArgumentParser(prog="pybasinsubs-app", description="pyBasinSubs in the browser (local only)")
    ap.add_argument("--port", type=int, default=None, help=f"port (default: first free from {DEFAULT_PORT})")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--page", choices=PAGES, default=default_page or None, help="page to open (default: start page)")
    a = ap.parse_args(argv)
    if os.environ.get("PYBASINSUBS_DRY_RUN"):
        print(f"http://{HOST}:{a.port or DEFAULT_PORT}/" + (f"#{a.page}" if a.page else ""))
        return 0
    missing = [m for m in ("fastapi", "uvicorn") if __import__("importlib.util").util.find_spec(m) is None]
    if missing:
        print(f"pybasinsubs-app needs {', '.join(missing)}: pip install fastapi uvicorn", file=sys.stderr)
        return 1
    import matplotlib
    matplotlib.use("Agg")
    from .routes import create_app
    from .server import serve
    serve(create_app(), "pybasinsubs", port=a.port, open_browser=not a.no_browser, page=a.page or "")
    return 0


def tracer(argv=None) -> int:
    """``pybasinsubs-tracer``: the app opened on the fault editor."""
    return main(argv, default_page="faults")
