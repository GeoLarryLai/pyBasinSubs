# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Panel fault-surface editor on the digitized VG1 section.
Launch:  pybasinsubs-tracer   (or: panel serve pybasinsubs/apps/tracer_panel.py --show)"""
from pybasinsubs.fault_tracer import vg1_editor

vg1_editor().panel().servable(title="pyBasinSubs fault-surface editor")
