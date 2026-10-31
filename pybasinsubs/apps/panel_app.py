# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Standalone page:  pybasinsubs-ui   (or: panel serve pybasinsubs/apps/panel_app.py --show)"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # works without pip install
import matplotlib
matplotlib.use("Agg")
from pybasinsubs.ui import app

app().servable(title="pyBasinSubs")
