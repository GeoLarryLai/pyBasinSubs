# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fault planes from fault sticks and from pre-rift block cutoffs (plane fits, Fault construction)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import flex2d

# ---------------------------------------------------------------- stick fitting


def fit_plane(stick):
    """Least-squares plane through stick points: x = a + b z. Returns (dip_deg, down, rms_x_m, a, b)."""
    st = np.asarray(stick, float)
    if len(st) < 2 or np.ptp(st[:, 1]) < 1e-6:
        return np.nan, 1, np.inf, float(st[0, 0]) if len(st) else np.nan, 0.0
    b, a = np.polyfit(st[:, 1], st[:, 0], 1)
    rms = float(np.sqrt(np.mean((a + b * st[:, 1] - st[:, 0]) ** 2)))
    dip = float(np.degrees(np.arctan2(1.0, abs(b))))
    return dip, (1 if b >= 0 else -1), rms, float(a), float(b)


def stick_to_fault(stick, x, tops, label="", source="", win=1500.0, down=None, confidence=None, ref=-1):
    """Turn fault-plane points into a flex2d.Fault: planar dip from the stick, trace x0 where the fitted plane
    meets the reference horizon (default: oldest top), throws of every horizon read across x0."""
    x = np.asarray(x, float); tops = np.asarray(tops, float); st = np.asarray(stick, float)
    dip, d_fit, _, a, b = fit_plane(st)
    down = d_fit if down is None else down
    z = tops[ref]; x0 = float(np.median(st[:, 0]))
    for _ in range(30):                                   # intersection of plane with reference horizon
        x0 = float(np.clip(a + b * np.interp(x0, x, z), x[0], x[-1]))
    on = np.abs(st[:, 1] - np.interp(st[:, 0], x, z)) < 50.0   # a stick point lying on the reference horizon wins
    if on.any():
        j = np.argmin(np.abs(st[on, 0] - x0))
        if abs(st[on][j, 0] - x0) < 1000.0:              # only if consistent with the fitted plane
            x0 = float(st[on][j, 0])
    f = flex2d.fault_at(x, tops, x0, down=down, win=win, dip=float(np.clip(np.nan_to_num(dip, nan=45.0), 1, 89)),
                        label=label, source=source)
    f.stick = st[np.argsort(st[:, 1])]
    f.confidence = confidence
    return f


def sticks_to_faults(sticks: dict, x, tops, source="imported", win=1500.0):
    """dict fault_id -> (n, 2) array (x, z)  ->  list of Fault sorted by x0."""
    fs = [stick_to_fault(st, x, tops, label=str(k), source=source, win=win) for k, st in sticks.items() if len(st) >= 2]
    return sorted(fs, key=lambda f: f.x0)

# ---------------------------------------------------------------- planes from pre-rift block cutoffs
def horizon_cutoffs(x, z, x_guess, down, x_far=None, search=3000.0, s_min=0.12, smooth=750.0, min_throw=100.0):
    """Footwall and hangingwall cutoffs of one horizon at a fault: the steep face (down-throw slope
    ``down * dz/dx > s_min`` after a ``smooth`` m running mean) nearest ``x_guess`` (searched +-``search`` m),
    from the footwall crest (start of the face) to the hangingwall trough (end of the face, not beyond
    ``x_far``, e.g. the next fault). Returns (x_fw, z_fw, x_hw, z_hw) or None if the face is < min_throw."""
    x = np.asarray(x, float); z = np.asarray(z, float); dx = x[1] - x[0]
    n = max(int(smooth / dx), 1)
    zs = np.convolve(np.pad(z, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), "valid")
    g = down * np.gradient(zs, dx)
    lo, hi = sorted((x_guess - down * search, x_guess + down * search))
    near = np.where((x >= lo) & (x <= hi) & (g > s_min))[0]
    if not len(near):
        return None
    j = near[np.argmax(g[near])]
    i0 = j; i1 = j
    step = 1 if down > 0 else -1
    while 0 <= i0 - step < len(x) and g[i0 - step] > s_min:
        i0 -= step
    while 0 <= i1 + step < len(x) and g[i1 + step] > s_min and (x_far is None or down * (x[i1 + step] - x_far) < 0):
        i1 += step
    # crest / trough on the raw horizon inside the face +- n nodes
    a, b = sorted((i0, i1)); a = max(a - n, 0); b = min(b + n, len(x) - 1)
    seg = np.arange(a, b + 1)
    fw = seg[np.argmin(z[seg] + 1e-6 * down * x[seg])]; hw = seg[np.argmax(z[seg] - 1e-6 * down * x[seg])]
    if down * (x[hw] - x[fw]) <= 0 or z[hw] - z[fw] < min_throw:
        return None
    return float(x[fw]), float(z[fw]), float(x[hw]), float(z[hw])


def faults_from_cutoffs(faults, x, tops, base=None, horizons=None, tip_unit=None, win=1500.0, ref=-1,
                        min_throw=100.0, **cut_kw):
    """Re-place faults on the observed pre-/syn-rift block edges.

    For each fault (position and polarity from detection, import or tracing) the footwall and hangingwall
    cutoffs of every horizon in ``horizons`` (default: the units below ``tip_unit``, i.e. the syn-rift and
    pre-rift tops, plus ``base``) are found with :func:`horizon_cutoffs`. A plane x = a + b z is fitted to the
    cutoffs by least squares weighted by each face's throw; its dip, standard error and the trace on the
    reference horizon replace the input. With ``tip_unit`` (e.g. 6 = top Kimmeridgian, base of the post-rift)
    the fault gets an upper tip where the plane meets that horizon: it does not cut or slip younger units.
    Returns a new list of Fault objects (x order); ``Fault.cutoffs`` holds the cutoff table and fit
    (dip_se, rms_x)."""
    x = np.asarray(x, float); tops = np.asarray(tops, float)
    fs = sorted(faults, key=lambda f: f.x0)
    if horizons is None:
        horizons = list(range(len(tops) if tip_unit is None else tip_unit, len(tops)))
    H = [(f"top{k}", tops[k]) for k in horizons] + ([("base", np.asarray(base, float))] if base is not None else [])
    out = []
    for i, f in enumerate(fs):
        nxt = [g.x0 for g in fs if f.down * (g.x0 - f.x0) > 0]
        x_far = (min(nxt, key=lambda v: abs(v - f.x0)) - f.down * 1000.0) if nxt else None
        rows = []
        for name, z in H:
            c = horizon_cutoffs(x, z, f.x0, f.down, x_far=x_far, min_throw=min_throw, **cut_kw)
            if c is not None:
                rows.append(dict(horizon=name, x_fw=c[0], z_fw=c[1], x_hw=c[2], z_hw=c[3], throw=c[3] - c[1]))
        if len(rows) < 1:
            out.append(f); continue
        tab = pd.DataFrame(rows)
        P = np.r_[tab[["x_fw", "z_fw"]].to_numpy(), tab[["x_hw", "z_hw"]].to_numpy()]
        w = np.r_[tab.throw.to_numpy(float), tab.throw.to_numpy(float)]
        A = np.c_[np.ones(len(P)), P[:, 1]]; W = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A * W, P[:, 0] * W[:, 0], rcond=None)
        a, b = coef
        if b * f.down <= 0:                                  # plane leaning the wrong way: keep the input fault
            out.append(f); continue
        res = P[:, 0] - (a + b * P[:, 1]); dof = max(len(P) - 2, 1)
        s2 = np.sum(w * res ** 2) / np.sum(w) * len(P) / dof
        zc = np.average(P[:, 1], weights=w)
        se_b = np.sqrt(s2 / max(np.sum(w / w.mean() * (P[:, 1] - zc) ** 2), 1e-9))
        dip = float(np.degrees(np.arctan2(1.0, abs(b))))
        dip_se = float(np.degrees(abs(b) * se_b / (1 + b * b)))
        stick = np.c_[a + b * P[:, 1], P[:, 1]]               # cutoffs projected on the fitted plane
        g = stick_to_fault(stick, x, tops, label=f.label, source=(f.source + "; " if f.source else "") +
                           "plane fitted to pre-rift block cutoffs", win=win, down=f.down, confidence=f.confidence, ref=ref)
        g.dip = dip
        if tip_unit is not None:
            zt = tops[tip_unit]; xt = float(np.median(stick[:, 0]))
            for _ in range(30):
                xt = float(np.clip(a + b * np.interp(xt, x, zt), x[0], x[-1]))
            # the plane must stay below the tip horizon: if it rises above it further down-dip (e.g. where all
            # horizons merge on a margin scarp), the tip moves to the last crossing, so no post-rift unit is cut
            zmax = float(P[:, 1].max()); zz = np.linspace(np.interp(xt, x, zt), zmax, 400); xx = a + b * zz
            above = zz < np.interp(xx, x, zt) - 1.0
            if above.any():
                k = int(np.where(above)[0].max()) + 1
                if k < len(zz):
                    xt = float(xx[k])
            g.tip_unit = int(tip_unit); g.z_tip = float(np.interp(xt, x, zt))
            g.throw = np.where(np.arange(len(g.throw)) <= tip_unit, 0.0, g.throw)
        g.cutoffs = dict(table=tab, a=float(a), b=float(b), dip_se=dip_se, rms_x=float(np.sqrt(np.average(res ** 2, weights=w))))
        out.append(g)
    return out
