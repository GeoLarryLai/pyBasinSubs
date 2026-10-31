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
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import flex2d

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
    cols = colors or [matplotlib.cm.tab10(i % 10) for i in range(len(tops))]
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


# ---------------------------------------------------------------- import / export


def project_to_line(xy, line):
    """Distance along a section line ((x0, y0), (x1, y1)) of map points xy (n x 2)."""
    (ax, ay), (bx, by) = line
    u = np.array([bx - ax, by - ay], float); u /= np.hypot(*u)
    return (np.asarray(xy, float) - [ax, ay]) @ u


LAYOUTS = {
    # tokens per data line (whitespace separated); 'skip' ignored. Vendor exports vary - adjust `columns`.
    "petrel_charisma": ["skip", "skip", "skip", "x", "y", "z", "fault_id", "stick"],  # INLINE- il xl x y z name stick
    "kingdom": ["x", "y", "z", "fault_id", "stick"],
    "opendtect": ["x", "y", "z", "stick"],
    "xyz_stick": ["x", "y", "z", "stick"],
}


def read_fault_sticks(path, fmt="auto", columns=None, line=None, fault_id=None, z_sign=1.0):
    """Read fault sticks and return dict fault_id -> (n, 2) array (x_along_section_m, z_depth_m).

    Formats (fmt):
    * 'csv'      - header with fault_id, x, z (m; z depth positive down); optional y (map northing -> projected
                   with `line`), optional stick (ignored for the 2-D section), optional x_km / z_km columns.
    * 'geojson'  - FeatureCollection of LineString / MultiLineString; fault id from properties
                   'fault_id' | 'name' | 'id'; coordinates [x, z] (section) or [x, y, z] (map + depth, needs `line`).
    * 'petrel_charisma' - Petrel 'Charisma fault sticks' style:  INLINE- <il> <xl> <x> <y> <z> <name> <stick>
    * 'kingdom'  - whitespace columns  x y z fault_name segment   (assumed simple layout)
    * 'opendtect'- FaultStickSet ASCII  x y z stick_index         ('#' comments; fault id from `fault_id` or file name)
    * 'auto'     - by extension (.csv, .geojson/.json) or first data line ('INLINE' -> charisma).
    Pass `columns=[...]` (names from x, y, z, fault_id, stick, skip) for other column orders.
    If y is present and `line` is None, x is used directly as distance along the section.
    Depth must be in metres (convert TWT first); use z_sign=-1 for elevation (negative down)."""
    path = Path(path); text = path.read_text()
    if fmt == "auto":
        ext = path.suffix.lower()
        first = next((l for l in text.splitlines() if l.strip() and not l.lstrip().startswith(("#", "!"))), "")
        if ext == ".csv":
            fmt = "csv"
        elif ext in (".geojson", ".json"):
            fmt = "geojson"
        elif first.upper().startswith("INLINE"):
            fmt = "petrel_charisma"
        else:
            fmt = "opendtect" if len(first.split()) == 4 else "kingdom"
    if fmt == "geojson":
        return _read_geojson(json.loads(text), line, z_sign)
    if fmt == "csv":
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "x_km" in df and "x" not in df:
            df["x"] = df["x_km"] * 1e3
        if "z_km" in df and "z" not in df:
            df["z"] = df["z_km"] * 1e3
        if "fault_id" not in df:
            df["fault_id"] = fault_id or path.stem
    else:
        cols = columns or LAYOUTS[fmt]
        rows = []
        for l in text.splitlines():
            t = l.split()
            if not t or t[0].startswith(("#", "!")) or t[0].upper() in ("BEGIN", "END", "VERSION", "FORMAT"):
                continue
            try:
                rows.append({c: (float(v) if c in ("x", "y", "z") else v) for c, v in zip(cols, t) if c != "skip"})
            except ValueError:
                continue                               # header / text line
        df = pd.DataFrame(rows)
        if "fault_id" not in df:
            df["fault_id"] = fault_id or path.stem
    df["z"] = df["z"].astype(float) * z_sign
    if "y" in df and line is not None:
        df["x"] = project_to_line(df[["x", "y"]].to_numpy(float), line)
    out = {}
    for k, g in df.groupby("fault_id", sort=False):
        out[str(k)] = g[["x", "z"]].to_numpy(float)
    return out


def _read_geojson(gj, line, z_sign):
    out = {}
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    for n, ft in enumerate(feats):
        g = ft["geometry"]; p = ft.get("properties") or {}
        fid = str(p.get("fault_id", p.get("name", p.get("id", f"F{n + 1}"))))
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for part in parts:
            c = np.asarray(part, float)
            if c.shape[1] == 3:
                xs = project_to_line(c[:, :2], line) if line is not None else c[:, 0]
                pts = np.c_[xs, c[:, 2] * z_sign]
            else:
                pts = np.c_[c[:, 0], c[:, 1] * z_sign]
            out[fid] = np.vstack([out[fid], pts]) if fid in out else pts
    return out


def write_fault_sticks_csv(faults_or_sticks, path):
    """Write Fault objects (their sticks) or a dict id -> (n, 2) array to CSV fault_id,x,z."""
    rows = []
    if isinstance(faults_or_sticks, dict):
        items = faults_or_sticks.items()
    else:
        items = [(f.label or f"F{i + 1}", f.stick if f.stick is not None else
                  np.array([[f.x0, 0.0], [f.x0 + f.down * 3000 / np.tan(np.radians(f.dip)), 3000.0]]))
                 for i, f in enumerate(faults_or_sticks)]
    for k, st in items:
        rows += [dict(fault_id=k, x=round(float(a), 1), z=round(float(b), 1)) for a, b in np.asarray(st)]
    pd.DataFrame(rows).drop_duplicates().to_csv(path, index=False)
    return path


def write_fault_sticks_geojson(sticks: dict, path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"fault_id": k},
         "geometry": {"type": "LineString", "coordinates": np.asarray(v).tolist()}} for k, v in sticks.items()]}
    Path(path).write_text(json.dumps(fc, indent=1))
    return path


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


# ---------------------------------------------------------------- curved (listric-type) fault traces
# dip(z) = dip_inf + (dip0 - dip_inf) exp(-(z - z_r) / lam)   for z >= z_r (dip0 above z_r)
# x(z)   = x_r + down * integral_{z_r}^{z} cot(dip(z')) dz'
# The dip decreases monotonically with depth (steep near the top, flattening toward the detachment), so the
# trace is monotone in depth and concave toward the hangingwall. Fitted by weighted least squares in x.
def curve_dip(z, z_r, dip0, lam, dip_inf=5.0):
    """Dip (deg) of the curved fault model at depth z."""
    z = np.asarray(z, float)
    return dip_inf + (dip0 - dip_inf) * np.exp(-np.maximum(z - z_r, 0.0) / lam)


def curve_x(z, x_r, z_r, dip0, lam, down=1, dip_inf=5.0, z_max=20e3):
    """x of the curved fault model at depths z (any order)."""
    z = np.asarray(z, float)
    lo = min(float(np.min(z)), z_r); hi = max(float(np.max(z)), z_r, lo + 1.0)
    zg = np.linspace(lo, min(hi, z_max), 1500)
    cot = 1.0 / np.tan(np.radians(curve_dip(zg, z_r, dip0, lam, dip_inf)))
    X = np.r_[0.0, np.cumsum(0.5 * (cot[1:] + cot[:-1]) * np.diff(zg))]
    X = X - np.interp(z_r, zg, X)
    return x_r + down * np.interp(z, zg, X)


def prm_x(z, prm):
    """x of a fitted curve (parameter dict) at depths z. With a footwall-crest anchor and ``dip_up``, the part above
    the anchor is the straight up-dip segment at the fitted syn-rift face dip ``dip_up`` (>= dip0)."""
    z = np.asarray(z, float)
    x = curve_x(z, prm["x_r"], prm["z_r"], prm["dip0"], prm["lam"], prm["down"], prm["dip_inf"])
    if prm.get("anchor") is not None and prm.get("dip_up") is not None:
        xa, za = prm["anchor"]
        x = np.where(z < za, xa + prm["down"] * (z - za) / np.tan(np.radians(prm["dip_up"])), x)
    return x


def fit_fault_curve(points, weights, down, dip_inf=5.0, dip0_bounds=(10.0, 70.0), lam_bounds=(500.0, 60e3),
                    lam_prior=(6000.0, 0.5), prior_weight=1.0, anchor=None):
    """Fit the curved fault model to constraint points (x, z) with weights. Returns dict(x_r, z_r, dip0, lam,
    dip_inf, cov (3x3 for x_r, dip0, lam), rms_x, n). z_r = depth of the shallowest point.

    ``lam_prior`` = (lam0, sd of ln lam): a soft log-normal prior on the flattening depth scale that keeps short,
    shallow constraint sets (a few km of fault) from bending the curve into an unphysical kink; its strength is
    ``prior_weight`` x the rms weighted residual of a straight-line fit (so it is in metres of x misfit).

    ``anchor`` = (x_a, z_a): hard constraint, the curve passes exactly through this point (x_r is then solved from
    dip0 and lam, and only those two are fitted; cov keeps the 3x3 layout with a zero x_r row/column)."""
    from scipy.optimize import least_squares
    P = np.asarray(points, float); w = np.sqrt(np.asarray(weights, float) / np.mean(weights))
    z_r = float(P[:, 1].min()) if anchor is None else float(min(P[:, 1].min(), anchor[1]))
    b, a = np.polyfit(P[:, 1], P[:, 0], 1, w=w)
    d_lin = float(np.clip(np.degrees(np.arctan2(1.0, abs(b))), dip0_bounds[0] + 1, dip0_bounds[1] - 1))
    p0 = [a + b * z_r, min(d_lin + 10.0, dip0_bounds[1] - 1), 4000.0]
    lin_rms = float(np.sqrt(np.average((a + b * P[:, 1] - P[:, 0]) ** 2, weights=w ** 2))) if len(P) > 2 else 300.0
    pw = prior_weight * max(lin_rms, 100.0) * np.sqrt(len(P))

    def xr_of(d0, lam):                      # x_r that puts the curve through the anchor
        return anchor[0] - float(curve_x(anchor[1], 0.0, z_r, d0, lam, down, dip_inf))

    def fun(p):
        if anchor is not None:
            p = [xr_of(p[0], p[1]), p[0], p[1]]
        r = (curve_x(P[:, 1], p[0], z_r, p[1], p[2], down, dip_inf) - P[:, 0]) * w
        return np.r_[r, pw * np.log(p[2] / lam_prior[0]) / lam_prior[1]] if lam_prior else r
    lb = [P[:, 0].min() - 20e3, max(dip0_bounds[0], dip_inf + 0.5), lam_bounds[0]]
    ub = [P[:, 0].max() + 20e3, dip0_bounds[1], lam_bounds[1]]
    if anchor is not None:
        r = least_squares(fun, p0[1:], bounds=(lb[1:], ub[1:]), x_scale=[10.0, 3000.0])
        J2 = r.jac[:len(P)]
        r.x = np.r_[xr_of(*r.x), r.x]; r.jac = np.c_[np.zeros((r.jac.shape[0], 1)), r.jac]
    else:
        r = least_squares(fun, p0, bounds=(lb, ub), x_scale=[1000.0, 10.0, 3000.0])
    res = r.fun[:len(P)] / w; dof = max(len(P) - 3, 1)
    s2 = float(np.sum(r.fun[:len(P)] ** 2) / dof)
    try:
        if anchor is not None:
            cov = np.zeros((3, 3)); cov[1:, 1:] = np.linalg.inv(J2.T @ J2) * s2
        else:
            cov = np.linalg.inv(r.jac.T @ r.jac) * s2
    except np.linalg.LinAlgError:
        cov = np.diag([0.0 if anchor is not None else 500.0 ** 2, 5.0 ** 2, 2000.0 ** 2])
    return dict(anchor=None if anchor is None else (float(anchor[0]), float(anchor[1])), x_r=float(r.x[0]), z_r=z_r, dip0=float(r.x[1]), lam=float(r.x[2]), dip_inf=dip_inf, down=int(down),
                cov=cov, rms_x=float(np.sqrt(np.average(res ** 2, weights=w ** 2))), n=len(P))


def _curve_samples(prm, z_top, Hd, n=120):
    zs = np.linspace(z_top, Hd, n)
    if prm.get("anchor") is not None and z_top < prm["anchor"][1] < Hd:     # sample the anchor exactly
        zs = np.unique(np.r_[zs, prm["anchor"][1]])
    return np.c_[prm_x(zs, prm), zs]


def _tip_on_curve(prm, x, zt_h, z_hint, z_min=0.0):
    """Depth where the curve meets the tip horizon zt_h(x), moved down-dip until the curve stays below it."""
    zz = np.linspace(z_min, max(z_hint, z_min + 10.0) + 3000.0, 1200)
    xx = prm_x(zz, prm)
    above = zz < np.interp(xx, x, zt_h) - 1.0
    if not above.any():
        return float(zz[0])
    k = int(np.where(above)[0].max()) + 1
    return float(zz[min(k, len(zz) - 1)])


def _finish_curved(g, prm, x, tops, tip_unit, Hd, win, ref=-1, seg=None):
    """Set curve, tip, trace x0, throws and effective planar dip of a curved Fault g in place."""
    tops = np.asarray(tops, float)
    zt = _tip_on_curve(prm, x, tops[tip_unit], prm["z_r"]) if tip_unit is not None else prm["z_r"]
    g.curve = _curve_samples(prm, zt, Hd); g.curve_params = dict(prm)
    zr = tops[ref]; x0 = float(g.curve[0, 0])
    if prm.get("anchor") is not None:                          # trace = the anchor (footwall crest of the oldest top)
        x0 = float(prm["anchor"][0])
    else:
        for _ in range(40):                                    # trace on the reference (oldest) top
            x0 = float(np.clip(g.x_at(np.interp(x0, x, zr)), x[0], x[-1]))
    g.x0 = x0
    zu = tops[:, :] ; thr = np.array([np.interp(x0 + g.down * win, x, t) - np.interp(x0 - g.down * win, x, t) for t in zu])
    g.throw = np.maximum(thr, 0.0)
    if tip_unit is not None:
        g.tip_unit = int(tip_unit); g.z_tip = float(zt)
        g.throw = np.where(np.arange(len(g.throw)) <= tip_unit, 0.0, g.throw)
    # block-bounding segment for the domino method: upper tip -> deepest hangingwall cutoff (base of the pre-rift
    # block), else the trace on the oldest top
    z1, z2 = (zt, float(prm.get("z_deep", np.interp(x0, x, zr)))) if seg is None else seg
    if z2 - z1 < 200.0:
        z2 = z1 + 1500.0
    g.dip = float(np.degrees(np.arctan2(z2 - z1, abs(float(g.x_at(z2)) - float(g.x_at(z1))))))
    g.stick = g.curve[:: max(len(g.curve) // 30, 1)]
    return g


def curved_fault_from_stick(f, x, tops, Hd=10e3, tip_unit=None, win=1500.0, ref=-1):
    """User-supplied (imported / traced) curved stick passed through unchanged: Fault.curve = the stick (sorted by
    depth), extended below its deepest point at its last segment's dip down to Hd; tip = shallowest point."""
    import copy
    g = copy.deepcopy(f); st = np.asarray(f.stick, float); st = st[np.argsort(st[:, 1])]
    if st[-1, 1] < Hd and len(st) >= 2:
        g.curve = st
        st = np.r_[st, [[float(g.x_at(Hd)), Hd]]]
    g.curve = st; g.curve_params = None
    if g.z_tip is None:
        g.z_tip = float(st[0, 1])
    if tip_unit is not None:
        g.tip_unit = int(tip_unit)
        g.throw = np.where(np.arange(len(g.throw)) <= tip_unit, 0.0, g.throw)
    zr = np.asarray(tops, float)[ref]; z1 = float(st[0, 1]); z2 = float(np.interp(g.x0, x, zr))
    if z2 - z1 > 200:
        g.dip = float(np.degrees(np.arctan2(z2 - z1, abs(float(g.x_at(z2)) - float(g.x_at(z1))))))
    return g


def faults_from_curves(faults, x, tops, base=None, tip_unit=None, Hd=10e3, win=1500.0, face_weight=1.0,
                       dip_inf=8.0, keep_user_sticks=True, ref=-1, curve_kw=None, trace_anchor="midramp", **cut_kw):
    """Curved fault traces fitted through ALL constraints, ordered by depth (see module notes above):

    1. the fault-face midpoints of the syn-rift horizons below the tip (``Fault.stick`` of auto-detected faults:
       the steep shallow part),
    2. the footwall and hangingwall cutoffs of the syn-rift and pre-rift tops and of ``base``
       (``faults_from_cutoffs``: the yellow block edges at depth),
    3. the model dip(z) = dip_inf + (dip0 - dip_inf) exp(-(z - z_r)/lam), which flattens toward the detachment Hd.

    Weights: cutoffs by their throw; face midpoints by ``face_weight`` x the median cutoff throw. The curve starts at
    the upper tip (top of ``tip_unit``) and ends at Hd. Imported / traced sticks (source starting 'imported' or
    'traced') pass through unchanged when ``keep_user_sticks``. Fault.dip becomes the effective planar dip of the
    block-bounding segment (tip to the deepest hangingwall cutoff), used by the domino method.

    trace_anchor: 'midramp' (default) = no anchor, x0 where the free curve meets the oldest top (in the middle of
    the drawn fault ramp); 'footwall_crest' = hard constraint, the curve passes exactly through the footwall-crest
    cutoff of the oldest top (``ref``, top Triassic on VG1), which is also the trace x0, with the syn-rift face dip
    above it."""
    if trace_anchor not in ("footwall_crest", "midramp"):
        raise ValueError("trace_anchor must be 'footwall_crest' or 'midramp'")
    x = np.asarray(x, float); tops = np.asarray(tops, float)
    fs = sorted(faults, key=lambda f: f.x0)
    user = [keep_user_sticks and f.stick is not None and str(f.source).startswith(("imported", "traced")) for f in fs]
    cut = faults_from_cutoffs(fs, x, tops, base, tip_unit=tip_unit, win=win, ref=ref, **cut_kw)
    out = []
    for f, g, u in zip(fs, cut, user):
        if u:
            out.append(curved_fault_from_stick(f, x, tops, Hd, tip_unit, win, ref)); continue
        if g.cutoffs is None:
            out.append(g); continue
        tab = g.cutoffs["table"]
        # window-picked cutoffs straddle the fault (footwall left of it, hangingwall right of it), so each horizon
        # constrains the fault at the midpoint of its fw / hw cutoff pair; the deepest hangingwall cutoff (base of
        # the pre-rift, the bottom corner of the yellow block) is used as it is: the fault at depth
        fw = tab[["x_fw", "z_fw"]].to_numpy(float); hw = tab[["x_hw", "z_hw"]].to_numpy(float)
        thr = tab.throw.to_numpy(float)
        P = [0.5 * (fw + hw)]; W = [thr.copy()]
        kb = int(np.argmax(hw[:, 1]))
        P.append(hw[kb:kb + 1]); W.append(thr[kb:kb + 1])
        face = np.zeros((0, 2))
        if f.stick is not None:
            st = np.asarray(f.stick, float)
            below = st[:, 1] > np.interp(st[:, 0], x, tops[tip_unit]) + 20.0 if tip_unit is not None else np.ones(len(st), bool)
            face = st[below]
            if len(face):
                P.append(face); W.append(np.full(len(face), face_weight * float(np.median(tab.throw))))
        P = np.vstack(P); W = np.concatenate(W)
        anc = None
        if trace_anchor == "footwall_crest":
            hn = f"top{len(tops) - 1 if ref == -1 else ref}"
            row = tab[tab.horizon == hn]
            if len(row):
                anc = (float(row.x_fw.iloc[0]), float(row.z_fw.iloc[0]))
        if anc is None:
            prm = fit_fault_curve(P, W, f.down, dip_inf=dip_inf, **(curve_kw or {}))
        else:
            # below the anchor: depth-flattening curve through the crest, fitted to the deeper constraints (yellow
            # block edges); above it: straight segment at the syn-rift face dip up to the tip (140 Ma)
            deep = P[:, 1] > anc[1] + 1.0
            if deep.sum() < 2:
                deep = np.ones(len(P), bool)
            prm = fit_fault_curve(P[deep], W[deep], f.down, dip_inf=dip_inf, anchor=anc, **(curve_kw or {}))
            up = face[face[:, 1] < anc[1] - 1.0] if len(face) else face
            if len(up) >= 1:                   # face dip through the anchor (least squares in x)
                dz = up[:, 1] - anc[1]; dx = f.down * (up[:, 0] - anc[0])
                c = float(np.sum(dz * dx) / np.sum(dz * dz)) if np.sum(dz * dz) > 0 else np.inf
                d_up = float(np.degrees(np.arctan2(1.0, c))) if c > 0 else f.dip
            else:
                d_up = f.dip                   # face-midpoint plane dip of the detected stick
            prm["dip_up"] = float(np.clip(d_up, prm["dip0"], 75.0))
        prm["points"] = P; prm["weights"] = W; prm["n_face"] = len(face)
        zbase_hw = float(tab.z_hw.max())
        prm["z_deep"] = zbase_hw
        h = _finish_curved(g, prm, x, tops, tip_unit, Hd, win, ref)
        h.source = (f.source + "; " if f.source else "") + "curve fitted to face midpoints + block cutoffs" + \
            ("; anchored at the footwall crest of the oldest top" if anc else "")
        h.cutoffs = dict(g.cutoffs, curve_rms_x=prm["rms_x"], z_deep=zbase_hw)
        out.append(h)
    return out


def perturb_curved_faults(faults, rng, x, tops, Hd=10e3, scale=1.0, win=1500.0):
    """One random draw of the fitted curves (x_r, dip0, lam from their fit covariance x scale); other faults are
    returned unchanged. Used by flex2d.monte_carlo_section(vary_faults=True)."""
    import copy
    out = []
    for f in faults:
        p = getattr(f, "curve_params", None)
        if not p or "cov" not in p:
            out.append(f); continue
        mu = np.array([p["x_r"], p["dip0"], p["lam"]])
        for _try in range(25):            # anchored draws whose curve would cut the tip horizon below the crest: redraw
            d = rng.multivariate_normal(np.zeros(3), np.asarray(p["cov"]) * scale ** 2)
            q = dict(p); q["x_r"], q["dip0"], q["lam"] = mu + d
            q["dip0"] = float(np.clip(q["dip0"], p["dip_inf"] + 1, 80)); q["lam"] = float(np.clip(q["lam"], 500, 60e3))
            if p.get("dip_up") is not None:
                q["dip_up"] = float(np.clip(p["dip_up"] + d[1], q["dip0"], 80.0))
            if p.get("anchor") is not None:          # anchored curves keep passing through the footwall crest
                xa, za = p["anchor"]
                q["x_r"] = xa - float(curve_x(za, 0.0, q["z_r"], q["dip0"], q["lam"], q["down"], q["dip_inf"]))
            g = _finish_curved(copy.deepcopy(f), q, x, tops, f.tip_unit, Hd, win)
            if q.get("anchor") is None or g.z_tip <= q["anchor"][1] + 1.0:
                break
        else:
            g = copy.deepcopy(f)
        out.append(g)
    return out


def curve_with_dip(f, dip, x, tops, Hd=10e3, win=1500.0):
    """Copy of a curved fault whose effective planar dip is changed to ``dip`` by shifting dip0 (shape kept,
    passing through the same shallowest constraint point). Planar faults: Fault.dip replaced."""
    import copy
    import dataclasses
    p = getattr(f, "curve_params", None)
    if not p:
        return dataclasses.replace(f, dip=float(dip))
    q = dict(p); g = copy.deepcopy(f)
    for _ in range(6):                                     # secant on dip0 to hit the requested effective dip
        q["dip0"] = float(np.clip(q["dip0"] + (dip - g.dip), p["dip_inf"] + 1, 85))
        g = _finish_curved(copy.deepcopy(f), q, x, tops, f.tip_unit, Hd, win)
        if abs(g.dip - dip) < 0.2:
            break
    return g


def curve_shift_dip(f, dip):
    """Copy of a fault with effective dip ``dip``. Curved faults (fitted, with curve_params): dip0 is shifted by the
    change and the curve regenerated through the same upper tip point, so the trace keeps its shape (steep top,
    flattening with depth); trace x0 and throws are kept. Plain faults: Fault.dip replaced."""
    import copy
    import dataclasses
    p = getattr(f, "curve_params", None)
    if f.curve is None or not p:
        return dataclasses.replace(f, dip=float(dip))
    g = copy.deepcopy(f); q = dict(p)
    q["dip0"] = float(np.clip(p["dip0"] + (dip - f.dip), p["dip_inf"] + 1.0, 85.0))
    zt, xt = float(f.curve[0, 1]), float(f.curve[0, 0])
    if p.get("anchor") is not None:              # rotate about the footwall-crest anchor (stays the trace)
        xt, zt = p["anchor"]
    q["x_r"] = 0.0
    q["x_r"] = xt - float(curve_x(zt, 0.0, q["z_r"], q["dip0"], q["lam"], q["down"], q["dip_inf"]))
    zs = f.curve[:, 1]
    if p.get("anchor") is not None:
        zs = np.unique(np.r_[zs, p["anchor"][1]])
    if q.get("dip_up") is not None:
        q["dip_up"] = float(np.clip(p["dip_up"] + (dip - f.dip), q["dip0"], 80.0))
    g.curve = np.c_[prm_x(zs, q), zs]
    g.curve_params = q; g.dip = float(dip); g.stick = g.curve[:: max(len(g.curve) // 30, 1)]
    return g


# ---------------------------------------------------------------- user-picked fault surfaces (stored polylines)
USER_FAULTS_FILE = "vg1_user_faults.json"
USER_FAULTS_CSV = "vg1_user_faults.csv"
FAULTS_JSON_VERSION = 1


def user_faults_path(name=USER_FAULTS_FILE):
    """Path of the persistent user fault file in the package data (pybasinsubs/example_data/faults/<name>)."""
    from pathlib import Path
    return Path(__file__).resolve().parent / "example_data" / "faults" / name


def stored_user_faults():
    """The stored user fault file to use: vg1_user_faults.json if present, else vg1_user_faults.csv, else None."""
    for n in (USER_FAULTS_FILE, USER_FAULTS_CSV):
        if user_faults_path(n).exists():
            return user_faults_path(n)
    return None


def write_user_faults_json(sticks: dict, path, tips=None, origins=None, names=None, section_id="VG1",
                           vertical_exaggeration=None, created=None):
    """Write fault polylines as JSON:
    {version, section_id, units: 'm', vertical_exaggeration, faults: [{id, name, tip_unit, method_origin,
    nodes: [[x, z], ...]}], metadata: {created, modified, pybasinsubs_version}}  (z = depth, positive down).
    ``created`` is kept from an existing file at ``path`` when not given."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from . import __version__
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if created is None and path.exists():
        try:
            created = json.loads(path.read_text()).get("metadata", {}).get("created")
        except (ValueError, OSError):
            created = None
    doc = dict(version=FAULTS_JSON_VERSION, section_id=section_id, units="m",
               vertical_exaggeration=None if vertical_exaggeration is None else float(vertical_exaggeration),
               faults=[dict(id=str(k), name=str((names or {}).get(k) or k),
                            tip_unit=None if (tips or {}).get(k) in (None, "") else int(tips[k]),
                            method_origin=(origins or {}).get(k) or "user",
                            nodes=[[float(a), float(b)] for a, b in np.asarray(v, float)]) for k, v in sticks.items()],
               metadata=dict(created=created or now, modified=now, pybasinsubs_version=__version__))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1)); tmp.replace(path)
    return path


def read_user_faults_json(path):
    """Read a fault JSON file (write_user_faults_json). Returns the document with nodes as (n, 2) arrays."""
    import json
    from pathlib import Path
    doc = json.loads(Path(path).read_text())
    if "faults" not in doc:
        raise ValueError(f"{path}: not a pybasinsubs fault JSON (no 'faults')")
    for f in doc["faults"]:
        f["nodes"] = np.asarray(f["nodes"], float).reshape(-1, 2)
    return doc


def write_user_faults(sticks: dict, path=None, tips=None, **json_kw):
    """Write fault polylines; a ``.json`` path (default user file) -> write_user_faults_json (``json_kw``: origins,
    names, section_id, vertical_exaggeration), otherwise CSV: fault_id, x (m), z (m, depth), order (node index), tip_unit (optional, per
    fault; empty = default). ``sticks``: dict fault_id -> (n, 2) array. Returns the path."""
    import pandas as pd
    from pathlib import Path
    path = Path(path) if path is not None else user_faults_path()
    if path.suffix.lower() == ".json":
        return write_user_faults_json(sticks, path, tips, **json_kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, st in sticks.items():
        st = np.asarray(st, float)
        t = (tips or {}).get(k)
        for j, (xx, zz) in enumerate(st):
            rows.append(dict(fault_id=str(k), x=float(xx), z=float(zz), order=j,
                             tip_unit="" if t is None or t == "" else int(t)))
    df = pd.DataFrame(rows, columns=["fault_id", "x", "z", "order", "tip_unit"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False); tmp.replace(path)                      # atomic replace
    return path


def read_user_faults(path=None):
    """Read a user fault file: JSON (read_user_faults_json) or CSV (fault_id, x, z[, order][, tip_unit]). Returns
    (sticks dict, tips dict); nodes are kept in ``order`` (file order if absent), fault order as first seen.
    Default: the stored user file (stored_user_faults())."""
    import pandas as pd
    from pathlib import Path
    path = Path(path) if path is not None else (stored_user_faults() or user_faults_path())
    if path.suffix.lower() == ".json":
        doc = read_user_faults_json(path)
        return ({f["id"]: f["nodes"] for f in doc["faults"]}, {f["id"]: f.get("tip_unit") for f in doc["faults"]})
    df = pd.read_csv(path, dtype={"fault_id": str})
    if "order" not in df:
        df["order"] = df.groupby("fault_id").cumcount()
    sticks, tips = {}, {}
    for k in pd.unique(df.fault_id):
        g = df[df.fault_id == k].sort_values("order")
        sticks[str(k)] = g[["x", "z"]].to_numpy(float)
        t = g["tip_unit"].dropna() if "tip_unit" in g else []
        tips[str(k)] = int(t.iloc[0]) if len(t) else None
    return sticks, tips


def user_sticks_to_faults(sticks: dict, x, tops, tips=None, tip_unit=6, Hd=10e3, win=1500.0, ref=-1):
    """User polylines -> curved Faults used unchanged by all restorations (curve = the polyline, extended to Hd at
    its last dip; trace x0 where it meets the oldest top; throws measured there; per-fault tip unit, default
    ``tip_unit``)."""
    out = []
    for k, st in sticks.items():
        st = np.asarray(st, float)
        if len(st) < 2:
            continue
        st = st[np.argsort(st[:, 1])]
        f = stick_to_fault(st, x, tops, label=str(k), source="traced (user file)", win=win, ref=ref)
        f.stick = st
        t = (tips or {}).get(k)
        tu = tip_unit if t is None else int(t)
        g = curved_fault_from_stick(f, x, tops, Hd, tu, win, ref)
        # trace x0 on the oldest top along the polyline itself
        zr = np.asarray(tops, float)[ref]; x0 = float(g.curve[0, 0])
        for _ in range(40):
            x0 = float(np.clip(g.x_at(np.interp(x0, x, zr)), x[0], x[-1]))
        g.x0 = x0
        g.throw = np.array([max(np.interp(x0 + g.down * win, x, t_) - np.interp(x0 - g.down * win, x, t_), 0.0)
                            for t_ in np.asarray(tops, float)])
        if tu is not None:
            g.throw[:tu + 1] = 0.0
        out.append(g)
    return sorted(out, key=lambda f: f.x0)


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
