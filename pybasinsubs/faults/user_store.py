# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""User-picked fault surfaces (stored polylines): JSON / CSV files and their conversion to Faults."""
from __future__ import annotations

import numpy as np
from .sticks import stick_to_fault
from .curves import curved_fault_from_stick

# ---------------------------------------------------------------- user-picked fault surfaces (stored polylines)
USER_FAULTS_FILE = "vg1_user_faults.json"
USER_FAULTS_CSV = "vg1_user_faults.csv"
FAULTS_JSON_VERSION = 1


def user_faults_path(name=USER_FAULTS_FILE):
    """Path of the persistent user fault file in the package data (pybasinsubs/example_data/faults/<name>)."""
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "example_data" / "faults" / name


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
    from .. import __version__
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
