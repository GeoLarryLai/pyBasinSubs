# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fault detection from horizon offsets and from a section image, and comparison with a reference set."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import flex2d
from .sticks import fit_plane, stick_to_fault

# ---------------------------------------------------------------- detection from horizon offsets


def horizon_discontinuities(x, z, slope_min=0.35, min_throw=150.0):
    """Steep offset segments on one horizon. Returns list of (x_mid, z_mid, throw, down):
    x_mid/z_mid = middle of the fault face (lies on the fault plane for a cut horizon), throw > 0 (m),
    down = +1 if the horizon steps down towards +x."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    g = np.gradient(z, x); m = np.abs(g) > slope_min
    out = []
    i = 0; n = len(x)
    while i < n:
        if not m[i]:
            i += 1; continue
        j = i
        while j + 1 < n and m[j + 1] and np.sign(g[j + 1]) == np.sign(g[i]):
            j += 1
        i0, i1 = max(i - 1, 0), min(j + 1, n - 1)
        thr = z[i1] - z[i0]
        if abs(thr) >= min_throw:
            zm = 0.5 * (z[i0] + z[i1]); seg = slice(i0, i1 + 1)
            o = np.argsort(z[seg]); xm = float(np.interp(zm, z[seg][o], x[seg][o]))
            out.append((xm, float(zm), float(abs(thr)), 1 if thr > 0 else -1))
        i = j + 1
    return out


def detect_faults_from_horizons(x, tops, horizons=None, slope_min=0.35, min_throw=150.0, link_dx=2500.0,
                                min_hits=2, dip_guess=50.0, dip_bounds=(25.0, 85.0), win=1500.0, ref=-1, merge_dx=1500.0):
    """Auto-detect faults from horizon offsets.

    1. On each horizon find steep offset segments (|dz/dx| > slope_min, throw > min_throw); the face midpoint
       is a point on the fault plane.
    2. Link points across horizons, deepest horizon first: a point on the next shallower horizon joins a stick
       if it has the same polarity and lies within link_dx of the position predicted by dip_guess.
    3. Keep sticks with >= min_hits horizons; fit a plane (dip) and compute the trace x0 on the reference
       horizon and the throws (flex2d.fault_at).

    confidence = sqrt(hits / n_horizons) * exp(-rms_x / 500 m), halved if the polarity of the throws disagrees
    with the dip direction or the dip falls outside dip_bounds.
    4. Merge sticks of the same polarity whose traces are closer than merge_dx (one fault picked twice).
    Returns (faults, sticks_table) where sticks_table lists every linked point."""
    x = np.asarray(x, float); tops = np.asarray(tops, float)
    H = list(range(len(tops))) if horizons is None else list(horizons)
    sticks = []                                           # each: dict(points=[(k,x,z,thr,down)], down)
    for k in sorted(H, key=lambda k: -np.nanmean(tops[k])):   # deepest first
        cand = horizon_discontinuities(x, tops[k], slope_min, min_throw)
        used = set()
        for st in sticks:
            kk, xl, zl, _, dn = st["points"][-1]
            best = None
            for c, (xm, zm, thr, d) in enumerate(cand):
                if c in used or d != dn or zm >= zl:
                    continue
                xp = xl - dn * (zl - zm) / np.tan(np.radians(dip_guess))
                e = abs(xm - xp)
                if e < link_dx and (best is None or e < best[0]):
                    best = (e, c)
            if best is not None:
                used.add(best[1]); st["points"].append((k,) + cand[best[1]])
        for c, p in enumerate(cand):
            if c not in used:
                sticks.append({"points": [(k,) + p], "down": p[3]})
    faults, rows = [], []
    nf = 0
    for st in sticks:
        P = st["points"]
        if len(P) < min_hits:
            continue
        pts = np.array([(p[1], p[2]) for p in P])
        dip, d_fit, rms, _, _ = fit_plane(pts)
        conf = np.sqrt(len(P) / len(H)) * np.exp(-rms / 500.0)
        if d_fit != st["down"] or not (dip_bounds[0] <= dip <= dip_bounds[1]):
            conf *= 0.5
        nf += 1
        f = stick_to_fault(pts, x, tops, label=f"auto-{nf}", source="auto-detected from horizon offsets",
                           win=win, down=st["down"], confidence=float(conf), ref=ref)
        f.dip = float(np.clip(f.dip, *dip_bounds))
        faults.append(f)
        rows += [dict(fault_id=f.label, horizon=p[0], x=p[1], z=p[2], throw=p[3], down=p[4]) for p in P]
    faults = sorted(faults, key=lambda f: f.x0)
    # merge sticks of the same polarity whose traces are closer than merge_dx (split picks of one fault)
    merged = []
    for f in faults:
        g = merged[-1] if merged else None
        if g is not None and g.down == f.down and abs(f.x0 - g.x0) < merge_dx:
            pts = np.vstack([g.stick, f.stick]); conf = min(1.0, max(g.confidence, f.confidence) + 0.1)
            m = stick_to_fault(pts, x, tops, label=g.label, source=g.source, win=win, down=g.down, confidence=conf, ref=ref)
            m.dip = float(np.clip(m.dip, *dip_bounds)); merged[-1] = m
            rows = [dict(r, fault_id=g.label) if r["fault_id"] == f.label else r for r in rows]
        else:
            merged.append(f)
    for i, f in enumerate(merged):                       # relabel left to right
        rows = [dict(r, fault_id=f"auto-{i + 1}") if r["fault_id"] == f.label else r for r in rows]
        f.label = f"auto-{i + 1}"
    return merged, pd.DataFrame(rows)


# ---------------------------------------------------------------- detection from an image


def render_section_image(x, tops, base, path=None, colors=None, extent_z=None, px=(1600, 400), dpi=100):
    """Rasterise a depth section (filled layers, no axes) for testing image detection.
    Returns (rgb array, extent) with extent = (xmin, xmax, zmax, zmin) in m (z down)."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    x = np.asarray(x, float); tops = np.asarray(tops, float)
    zmin = 0.0 if extent_z is None else extent_z[0]; zmax = float(np.nanmax(base)) if extent_z is None else extent_z[1]
    fig = Figure(figsize=(px[0] / dpi, px[1] / dpi), dpi=dpi); canvas = FigureCanvasAgg(fig); ax = fig.add_axes([0, 0, 1, 1])
    from ..material_colors import VG1_COLORS
    cols = colors if colors is not None else (VG1_COLORS if len(tops) == len(VG1_COLORS) else
                                             [matplotlib.cm.tab10(i % 10) for i in range(len(tops))])
    ax.fill_between(x, zmin, tops[0], color="white", lw=0)
    for k in range(len(tops)):
        b = tops[k + 1] if k + 1 < len(tops) else base
        ax.fill_between(x, tops[k], b, color=cols[k], lw=0)
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(zmax, zmin); ax.axis("off")
    canvas.draw(); img = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    if path:
        fig.savefig(path, dpi=dpi)
    return img, (float(x[0]), float(x[-1]), zmax, zmin)


def detect_faults_from_image(img, extent, x=None, tops=None, sigma=2.0, line_length=15, line_gap=3,
                             hough_threshold=10, dip_bounds=(25.0, 85.0), cluster_dx=2000.0, min_segments=2,
                             min_throw=150.0, win=1500.0, seed=0, ref=-1):
    """Auto-detect fault sticks on a section image (requires scikit-image).

    img: path or array (RGB or grey); extent = (xmin, xmax, zmax, zmin) in m, as matplotlib imshow (z down).
    Canny edges -> probabilistic Hough line segments -> keep segments whose TRUE-SCALE dip (pixels converted
    to m, so vertical exaggeration is removed) lies in dip_bounds -> cluster by mid-x within cluster_dx
    and same dip direction -> one stick per cluster (segment end points).
    If x and tops are given, a cluster is kept only where the oldest horizon is offset by >= min_throw across it
    (horizon-offset alignment), and the Fault gets throws via flex2d.fault_at.
    confidence = min(1, n_segments / 6) * (0.5 + 0.5 * offset_confirmed)."""
    from skimage import io, feature, transform
    im = io.imread(img) if isinstance(img, (str, Path)) else np.asarray(img)
    if im.ndim == 3:
        im = im[..., :3]
    ny, nx = im.shape[:2]; x0e, x1e, zb, zt = extent
    edges = np.zeros(im.shape[:2], bool)                 # edges of every colour channel (similar greys stay separable)
    for ch in ([im] if im.ndim == 2 else [im[..., c] for c in range(im.shape[2])]):
        edges |= feature.canny(ch.astype(float) / max(float(ch.max()), 1.0), sigma=sigma)
    segs = transform.probabilistic_hough_line(edges, threshold=hough_threshold, line_length=line_length,
                                              line_gap=line_gap, rng=seed)
    W = []
    for (c0, r0), (c1, r1) in segs:
        X = x0e + (np.array([c0, c1]) + 0.5) / nx * (x1e - x0e)
        Z = zt + (np.array([r0, r1]) + 0.5) / ny * (zb - zt)
        dxm, dzm = X[1] - X[0], Z[1] - Z[0]
        if abs(dxm) < 1e-9:
            continue
        dip = np.degrees(np.arctan(abs(dzm / dxm)))
        if dip_bounds[0] <= dip <= dip_bounds[1]:
            W.append((X.mean(), 1 if dzm / dxm > 0 else -1, X, Z))
    W.sort(key=lambda w: w[0])
    clusters = []
    for w in W:
        for c in clusters:
            if c["down"] == w[1] and abs(np.mean(c["xm"]) - w[0]) < cluster_dx:
                c["xm"].append(w[0]); c["pts"] += list(zip(w[2], w[3])); break
        else:
            clusters.append({"down": w[1], "xm": [w[0]], "pts": list(zip(w[2], w[3]))})
    faults, n = [], 0
    for c in clusters:
        if len(c["xm"]) < min_segments:
            continue
        pts = np.array(c["pts"]); conf = min(1.0, len(c["xm"]) / 6.0)
        if x is not None and tops is not None:
            xx = np.asarray(x, float); zr = np.asarray(tops, float)[ref]; xc = float(np.mean(c["xm"]))
            thr = (np.interp(xc + c["down"] * win, xx, zr) - np.interp(xc - c["down"] * win, xx, zr))
            if thr < min_throw:
                continue
            n += 1
            f = stick_to_fault(pts, xx, tops, label=f"img-{n}", source="auto-detected from image (Canny+Hough)",
                               win=win, down=c["down"], confidence=float(conf), ref=ref)
        else:
            n += 1
            dip, d_fit, _, a, b = fit_plane(pts)
            f = flex2d.Fault(x0=float(np.median(pts[:, 0])), dip=float(np.clip(np.nan_to_num(dip, nan=45), 1, 89)),
                             down=c["down"], throw=np.zeros(0), label=f"img-{n}",
                             source="auto-detected from image (Canny+Hough)", stick=pts[np.argsort(pts[:, 1])],
                             confidence=float(conf * 0.5))
        f.dip = float(np.clip(f.dip, *dip_bounds))
        faults.append(f)
    return sorted(faults, key=lambda f: f.x0)


def compare_faults(detected, reference, max_dx=3000.0):
    """Match detected to reference faults by nearest x0 (within max_dx). Returns a DataFrame with
    position error dx0 (m), dip error (deg) and misses/false positives."""
    rows, used = [], set()
    for r in reference:
        c = [(abs(d.x0 - r.x0), i) for i, d in enumerate(detected) if i not in used]
        c = [t for t in c if t[0] <= max_dx]
        if c:
            _, i = min(c); used.add(i); d = detected[i]
            rows.append(dict(reference=r.label or f"{r.x0/1e3:.2f} km", ref_x0_km=r.x0 / 1e3, det=d.label,
                             det_x0_km=d.x0 / 1e3, dx0_m=d.x0 - r.x0, ref_dip=r.dip, det_dip=d.dip,
                             ddip_deg=d.dip - r.dip, confidence=d.confidence))
        else:
            rows.append(dict(reference=r.label, ref_x0_km=r.x0 / 1e3, det="MISSED"))
    for i, d in enumerate(detected):
        if i not in used:
            rows.append(dict(reference="(none)", det=d.label, det_x0_km=d.x0 / 1e3, det_dip=d.dip,
                             confidence=d.confidence))
    return pd.DataFrame(rows)
