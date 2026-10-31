# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Locations of the bundled example data and of the user cache."""
import os
from importlib.resources import files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root in a source checkout


def data_path(*parts):
    """Path of a file shipped in ``pybasinsubs/example_data`` (env PYBASINSUBS_DATA overrides).

    Works from a source checkout and from an installed wheel (importlib.resources).
    """
    base = os.environ.get("PYBASINSUBS_DATA") or str(files("pybasinsubs") / "example_data")
    return os.path.join(base, *parts)


def cache_dir():
    """User cache for downloaded data (env PYBASINSUBS_CACHE, else ~/.cache/pybasinsubs)."""
    d = os.environ.get("PYBASINSUBS_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "pybasinsubs")
    os.makedirs(d, exist_ok=True)
    return d
