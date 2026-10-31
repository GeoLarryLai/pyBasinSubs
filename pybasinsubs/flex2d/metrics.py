# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Flatness / artefact metrics of restored sections."""
from __future__ import annotations
import numpy as np


def restoration_metrics(r, i, x, faults, win=1500.0, xlim=(2e3, 80e3), ref=None):
    """Flatness and artefact metrics of a restored horizon at step i: std, P95-P5 relief, max relief,
    residual fault offsets (mean |dz| across +-win) and roughness (rms of 2nd difference, m).

    ref: reference horizon, the unit index (or unit name) whose top is measured; default None = the
    youngest unit remaining at step i (the restored datum, i.e. top[i][i]). Layer-area change versus the
    unfaulted restoration is computed by the caller (layer_areas)."""
    if ref is None:
        k = i
    elif isinstance(ref, str):
        k = list(r["units"]).index(ref)
    else:
        k = int(ref)
    if k < i:
        raise ValueError(f"reference unit {k} is already stripped at step {i}")
    z = r["top"][i][k]; m = (x >= xlim[0]) & (x <= xlim[1])
    off = [abs(np.interp(f.x0 + f.down * win, x, z) - np.interp(f.x0 - f.down * win, x, z)) for f in faults]
    return dict(std=float(np.nanstd(z[m])), relief_p95_p5=float(np.nanpercentile(z[m], 95) - np.nanpercentile(z[m], 5)),
                max_relief=float(np.nanmax(z[m]) - np.nanmin(z[m])), mean_fault_offset=float(np.mean(off)) if off else 0.0,
                roughness=float(np.sqrt(np.nanmean(np.diff(z[m], 2) ** 2))))


def layer_areas(r, i):
    """Cross-sectional area (m^2) of every unit present at step i."""
    dx = r["x"][1] - r["x"][0]
    return np.nansum(np.maximum(r["base"][i] - r["top"][i], 0.0), axis=1) * dx
