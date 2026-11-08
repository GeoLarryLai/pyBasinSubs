# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fault interpretation for 2-D sections: auto-detection, import/export and stick fitting.

Everything returns ``flex2d.Fault`` objects that feed straight into ``flex2d.restore(sec, faults, method=...)``.
Each Fault carries ``stick`` (n x 2 array of fault-plane points x, z in m) and, for auto-detected faults,
a ``confidence`` in 0-1.

Conventions: x = distance along the section (m), z = depth (m, positive down). ``down=+1`` means the
fault dips towards +x (hangingwall on the +x side).

Detection
---------
* ``detect_faults_from_horizons`` - throw discontinuities (steep, offset segments) on every horizon,
  linked across horizons into fault sticks, then fitted with a plane (dip) and optionally a listric curve.
* ``detect_faults_from_image`` - Canny edges + probabilistic Hough lines on a section image (scikit-image,
  optional), steep segments clustered into sticks, optionally confirmed by horizon offsets.

Import / export (formats documented in ``read_fault_sticks``)
--------------------------------------------------------------
CSV fault sticks (fault_id, x, z [, y, stick]), simple fault-stick ASCII layouts of Petrel (Charisma),
Kingdom and OpendTect exports, and GeoJSON LineStrings.

Package layout (``from pybasinsubs import faults`` and every former ``faults.<name>`` keep working)
-----------------------------------------------------------------------------------------------
  sticks      fit_plane, stick_to_fault, sticks_to_faults, horizon_cutoffs, faults_from_cutoffs
  detect      horizon / image detection, compare_faults
  formats     read / write fault sticks (CSV, Petrel, Kingdom, OpendTect, GeoJSON)
  curves      curved (listric-type) fault traces, faults_from_curves, perturb_curved_faults
  user_store  stored user polylines (JSON / CSV), user_sticks_to_faults
  (here)      auto_faults, load_faults, fault_polylines (entry points)
"""

# ruff: noqa: F401  (re-exports keep the pre-split faults namespace)
from __future__ import annotations

from pathlib import Path

import numpy as np

from .sticks import (fit_plane, stick_to_fault, sticks_to_faults, horizon_cutoffs, faults_from_cutoffs)
from .detect import (horizon_discontinuities, detect_faults_from_horizons, render_section_image, detect_faults_from_image, compare_faults)
from .formats import (project_to_line, LAYOUTS, read_fault_sticks, _read_geojson, write_fault_sticks_csv, write_fault_sticks_geojson)
from .curves import (curve_dip, curve_x, prm_x, fit_fault_curve, _curve_samples, _tip_on_curve, _finish_curved, curved_fault_from_stick, faults_from_curves, perturb_curved_faults, curve_with_dip, curve_shift_dip)
from .user_store import (USER_FAULTS_FILE, USER_FAULTS_CSV, FAULTS_JSON_VERSION, user_faults_path, stored_user_faults, write_user_faults_json, read_user_faults_json, write_user_faults, read_user_faults, user_sticks_to_faults)


AUTO_SOURCES = ("auto", "curved_midramp", "curved_crest", "straight")


def auto_faults(method="curved_midramp", x=None, tops=None, base=None, tip_unit=6, Hd=10e3, trace_anchor="midramp"):
    """The automatic fault estimates, all kept selectable: 'straight' = planar picks from horizon offsets
    (detect_faults_from_horizons, face-midpoint / straight cutoff planes); 'curved_midramp' = curved faults,
    trace in the middle of the ramp (default); 'curved_crest' = curved, anchored at the footwall crest; 'auto' =
    curved with ``trace_anchor``."""
    raw, _ = detect_faults_from_horizons(x, tops)
    if method == "straight":
        return raw
    anchor = {"curved_midramp": "midramp", "curved_crest": "footwall_crest"}.get(method, trace_anchor)
    return faults_from_curves(raw, x, tops, base, tip_unit=tip_unit, Hd=Hd, trace_anchor=anchor)


def load_faults(source="auto", x=None, tops=None, base=None, tip_unit=6, Hd=10e3, trace_anchor="midramp",
                fallback=True):
    """Fault set by source: 'auto' = auto-detected curves (faults_from_curves, ``trace_anchor``, default mid-ramp);
    'curved_midramp' / 'curved_crest' / 'straight' = the auto methods (auto_faults); 'user' = the
    stored user file (vg1_user_faults.json, else .csv), falling back to 'auto' if it does not exist and ``fallback``; any other
    string / Path = a fault-stick file (user JSON, user CSV with order/tip_unit, or anything read_fault_sticks reads).
    Returns (faults, used_source)."""
    from pathlib import Path
    if source == "user":
        p = stored_user_faults()
        if p is None:
            if not fallback:
                raise FileNotFoundError(user_faults_path())
            source = "auto"
        else:
            source = p
    if source in AUTO_SOURCES:
        return auto_faults(source, x, tops, base, tip_unit, Hd, trace_anchor), source
    p = Path(source)
    try:
        sticks, tips = read_user_faults(p)
    except Exception:
        sticks, tips = read_fault_sticks(p), None
    return user_sticks_to_faults(sticks, x, tops, tips, tip_unit=tip_unit, Hd=Hd), str(p)


def fault_polylines(faults, z_bottom=None, n=8, x=None, ref_top=None):
    """Editable polylines (dict label -> (n, 2) x, z) of any fault set: the curve (subsampled to ~n nodes) between
    the tip and ``z_bottom`` (default: deepest stick point / 8 km), or the straight plane from tip to z_bottom
    (through the trace x0 on ``ref_top`` sampled at ``x`` when given, i.e. the oldest top)."""
    out = {}
    for i, f in enumerate(faults):
        k = str(f.label or f"F{i + 1}")
        z0 = f.z_tip if f.z_tip is not None else (float(np.min(f.stick[:, 1])) if f.stick is not None else 0.0)
        zb = z_bottom if z_bottom is not None else (float(np.max(f.stick[:, 1])) if f.stick is not None else 8e3)
        zb = max(zb, z0 + 500.0)
        zz = np.linspace(z0, zb, n)
        if f.curve is None and x is not None and ref_top is not None:
            zr = float(np.interp(f.x0, x, ref_top))
            xs = f.x0 + f.down * (zz - zr) / np.tan(np.radians(f.dip))
        else:
            xs = [float(f.x_at(z)) for z in zz]
        out[k] = np.c_[xs, zz]
    return out
