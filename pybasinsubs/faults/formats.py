# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fault-stick import / export: CSV, Petrel (Charisma), Kingdom and OpendTect ASCII, GeoJSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
