# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Console entry points.

pybasinsubs-ui [--streamlit] [extra args]      backstripping / geohistory app (Panel by default)
pybasinsubs-tracer [--streamlit] [extra args] fault-surface editor + restoration (Panel by default)
"""
from __future__ import annotations
import os
import subprocess
import sys

_APPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps")


def _cmd(engine, script, extra):
    path = os.path.join(_APPS, script)
    if engine == "panel":
        return [sys.executable, "-m", "panel", "serve", path, "--show", *extra]
    return [sys.executable, "-m", "streamlit", "run", path, *extra]


def _launch(argv, flag, default, other, scripts):
    argv = list(sys.argv[1:] if argv is None else argv)
    engine = other if flag in argv else default
    if flag in argv:
        argv.remove(flag)
    if "-h" in argv or "--help" in argv:
        print(__doc__); return 0
    cmd = _cmd(engine, scripts[engine], argv)
    if os.environ.get("PYBASINSUBS_DRY_RUN"):
        print(" ".join(cmd)); return 0
    return subprocess.call(cmd)


def ui(argv=None):
    return _launch(argv, "--streamlit", "panel", "streamlit",
                   {"panel": "panel_app.py", "streamlit": "streamlit_app.py"})


def tracer(argv=None):
    return _launch(argv, "--streamlit", "panel", "streamlit",
                   {"panel": "tracer_panel.py", "streamlit": "tracer_streamlit.py"})


if __name__ == "__main__":  # pragma: no cover
    sys.exit(ui())
