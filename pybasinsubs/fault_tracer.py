# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fault-surface editor logic (no GUI): the editable polylines, undo / redo, save / load and restoration.

    pybasinsubs-tracer                               # the browser app (pybasinsubs-app) on the VG1 section
    from pybasinsubs.fault_tracer import FaultTracer, vg1_example
    sec, tops, base = vg1_example()
    tr = FaultTracer(sec, tops, base=base)           # loads the stored user faults if the file exists
    tr.store.add([(1e4, 2e3), (1.2e4, 5e3)])         # edit polylines (undo / redo, autosave)
    tr.faults(); tr.restore("cantilever")            # curved Faults from the polylines / a restoration

The browser page (``pybasinsubs.webapp``, "Fault editor") drives :class:`FaultStore` / :class:`FaultTracer`:
click to place nodes, Finish, select, drag or insert / delete nodes, delete / rename a fault, set its tip horizon,
Undo / Redo, Reset to (curved mid-ramp | curved footwall crest | straight | last saved | empty), Save / Load as JSON
(default, pybasinsubs/example_data/faults/vg1_user_faults.json) or CSV (vg1_user_faults.csv), and Run restoration
(method / Te / H_d) with the restored oldest horizon and its flatness metric. Every edit is autosaved to
vg1_user_faults.autosave.json.
JSON: {version, section_id, units 'm', vertical_exaggeration, faults: [{id, name, tip_unit, method_origin,
nodes: [[x, z], ...]}], metadata: {created, modified, pybasinsubs_version}}; CSV columns: fault_id, x, z, order,
tip_unit (m, depth positive down).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import flex2d, faults as F
from ._paths import data_path
from .material_colors import section_material_colors

RESETS = {"curved mid-ramp (auto, default)": "curved_midramp", "curved footwall crest (auto)": "curved_crest",
          "straight (auto planar picks)": "straight", "last saved file": "saved", "empty": "empty"}
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


class FaultStore:
    """Editable fault polylines (no GUI): dict fault_id -> (n, 2) x, z (m) + tip unit per fault, undo / redo,
    resets to the auto methods, save / load of the persistent CSV with autosave."""

    def __init__(self, x, tops, base=None, path=None, tip_unit=6, Hd=10e3, z_bottom=8e3, autosave=True):
        self.x = np.asarray(x, float); self.tops = np.asarray(tops, float); self.base = base
        self.path = Path(path) if path is not None else F.user_faults_path()
        self.tip_unit = tip_unit; self.Hd = Hd; self.z_bottom = z_bottom; self.autosave = autosave
        self.sticks: dict = {}; self.tips: dict = {}; self.origins: dict = {}
        self.vertical_exaggeration = None; self.section_id = "VG1"
        self._undo: list = []; self._redo: list = []

    @property
    def autosave_path(self):
        return self.path.with_name(self.path.stem + ".autosave" + (self.path.suffix or ".json"))

    def stored_path(self):
        """The file 'last saved' / load use: self.path, else the other format (.json <-> .csv) next to it."""
        if self.path.exists():
            return self.path
        alt = self.path.with_suffix(".csv" if self.path.suffix.lower() == ".json" else ".json")
        return alt if alt.exists() else None

    def _json_kw(self):
        return dict(origins=self.origins, section_id=self.section_id, vertical_exaggeration=self.vertical_exaggeration)

    # ---- history
    def _snap(self):
        return ({k: np.array(v, float) for k, v in self.sticks.items()}, dict(self.tips), dict(self.origins))

    def _commit(self, before):
        self._undo.append(before); self._redo.clear()
        del self._undo[:-200]
        if self.autosave:
            try:
                F.write_user_faults(self.sticks, self.autosave_path, self.tips, **self._json_kw())
            except OSError:
                pass

    def _apply(self, sticks, tips, origins=None):
        old = self.origins
        self.sticks = {str(k): np.asarray(v, float) for k, v in sticks.items()}
        self.tips = {k: tips.get(k) for k in self.sticks}
        origins = old if origins is None else origins
        self.origins = {k: origins.get(k, "user") for k in self.sticks}

    def edit(self, sticks, tips=None, origins=None):
        """Replace all polylines (one undo step). ``origins``: fault_id -> method_origin (kept / 'user' if absent)."""
        b = self._snap(); self._apply(sticks, self.tips if tips is None else tips, origins); self._commit(b)

    def undo(self):
        if self._undo:
            self._redo.append(self._snap()); self._apply(*self._undo.pop()); return True
        return False

    def redo(self):
        if self._redo:
            self._undo.append(self._snap()); self._apply(*self._redo.pop()); return True
        return False

    # ---- operations
    def new_id(self):
        i = len(self.sticks) + 1
        while f"F{i}" in self.sticks:
            i += 1
        return f"F{i}"

    def add(self, pts, fid=None, tip=None):
        fid = fid or self.new_id(); b = self._snap()
        self.sticks[fid] = np.asarray(pts, float); self.tips[fid] = tip; self.origins[fid] = "user"
        self._commit(b); return fid

    def delete(self, fid):
        b = self._snap(); self.sticks.pop(fid, None); self.tips.pop(fid, None); self.origins.pop(fid, None); self._commit(b)

    def rename(self, fid, new):
        new = str(new).strip()
        if not new or new == fid or new in self.sticks or fid not in self.sticks:
            return False
        b = self._snap()
        self.sticks = {(new if k == fid else k): v for k, v in self.sticks.items()}
        self.tips = {(new if k == fid else k): v for k, v in self.tips.items()}
        self.origins = {(new if k == fid else k): v for k, v in self.origins.items()}
        self._commit(b); return True

    def set_tip(self, fid, tip):
        b = self._snap(); self.tips[fid] = None if tip is None else int(tip); self._commit(b)

    def move_node(self, fid, j, xz):
        b = self._snap(); st = self.sticks[fid].copy(); st[j] = xz; self.sticks[fid] = st; self._commit(b)

    def insert_node(self, fid, j=None):
        """Insert the midpoint after node j (default: in the longest segment). Returns the new node index."""
        st = self.sticks[fid]
        if len(st) < 2:
            return None
        if j is None or j >= len(st) - 1:
            j = int(np.argmax(np.hypot(*np.diff(st, axis=0).T))) if j is None else len(st) - 2
        b = self._snap(); self.sticks[fid] = np.insert(st, j + 1, (st[j] + st[j + 1]) / 2, axis=0)
        self._commit(b); return j + 1

    def delete_node(self, fid, j):
        st = self.sticks[fid]
        if len(st) <= 2:
            return False
        b = self._snap(); self.sticks[fid] = np.delete(st, j, axis=0); self._commit(b); return True

    def reset(self, kind="curved_midramp"):
        """kind: 'curved_midramp' | 'curved_crest' | 'straight' (auto methods), 'saved' (last saved file) | 'empty'."""
        if kind == "empty":
            self.edit({}, {}); return
        if kind == "saved":
            p = self.stored_path()
            if p is None:
                self.edit({}, {}); return
            self.load(p); return
        fs = F.auto_faults(kind, self.x, self.tops, self.base, tip_unit=self.tip_unit, Hd=self.Hd)
        s = F.fault_polylines(fs, self.z_bottom, n=12, x=self.x, ref_top=self.tops[-1])
        self.edit(s, {k: None for k in s}, {k: kind for k in s})

    def save(self, path=None, fmt=None):
        """Save to ``path`` (default self.path); ``fmt`` 'json' | 'csv' swaps the suffix of the default path."""
        p = Path(path) if path is not None else self.path
        if fmt is not None:
            p = p.with_suffix("." + fmt.lower())
        return F.write_user_faults(self.sticks, p, self.tips, **self._json_kw())

    def load(self, path=None, fmt=None):
        """Load a JSON or CSV fault file (default: the stored file; ``fmt`` picks the .json / .csv sibling)."""
        p = Path(path) if path is not None else (self.path.with_suffix("." + fmt.lower()) if fmt else self.stored_path())
        if p is None or not p.exists():
            raise FileNotFoundError(p or self.path)
        if p.suffix.lower() == ".json":
            doc = F.read_user_faults_json(p)
            s = {f["id"]: f["nodes"] for f in doc["faults"]}
            t = {f["id"]: f.get("tip_unit") for f in doc["faults"]}
            o = {f["id"]: f.get("method_origin") or "user" for f in doc["faults"]}
            if doc.get("vertical_exaggeration") is not None:
                self.vertical_exaggeration = doc["vertical_exaggeration"]
            self.section_id = doc.get("section_id", self.section_id)
        else:
            s, t = F.read_user_faults(p); o = {k: "imported" for k in s}
        self.edit(s, t, o); return p

    def faults(self, win=1500.0):
        return F.user_sticks_to_faults(self.sticks, self.x, self.tops, self.tips, tip_unit=self.tip_unit, Hd=self.Hd,
                                       win=win)


class FaultTracer:
    """Fault-surface editor over a section. ``start``: 'auto' = the stored user file if it exists, else the
    curved mid-ramp auto faults; or any FaultStore.reset kind; ``faults`` (list of Fault) overrides it."""

    def __init__(self, sec, tops, image=None, extent=None, faults=None, colors=None, title="Fault-surface editor",
                 base=None, path=None, tip_unit=6, Hd=10e3, start="auto", units=None):
        self.sec = sec; self.x = np.asarray(sec.x, float); self.tops = np.asarray(tops, float)
        self.base = base; self.image = image; self.extent = extent; self.colors = colors; self.title = title
        self.units = units or list(getattr(sec, "units", [])) or [f"unit {i}" for i in range(len(tops))]
        self.store = FaultStore(self.x, self.tops, base, path, tip_unit, Hd, autosave=False)
        if faults is not None:
            self.store.edit(F.fault_polylines(faults, None, x=self.x, ref_top=self.tops[-1]))
        elif start == "auto":
            self.store.reset("saved" if self.store.stored_path() is not None else "curved_midramp")
        else:
            self.store.reset(start)
        self.store._undo.clear(); self.store.autosave = True
        self.result = None

    # ------------------------------------------------------------ data access
    def sticks(self):
        return {k: v.copy() for k, v in self.store.sticks.items()}

    def set_sticks(self, sticks: dict, tips=None):
        self.store.edit(sticks, tips)

    def faults(self, win=1500.0):
        return self.store.faults(win)

    def save(self, path=None, fmt=None):
        return self.store.save(path, fmt)

    def restore(self, method="cantilever", **kw):
        kw.setdefault("Te", 1500.0); kw.setdefault("Hd", self.store.Hd)
        if method == "cantilever":
            kw.setdefault("beta", 1.15)
        self.result = flex2d.restore(self.sec, self.faults(), method=method, **kw)
        return self.result

    def to_csv(self, path):
        return F.write_user_faults(self.store.sticks, Path(path).with_suffix(".csv"), self.store.tips)

    def to_json(self, path):
        return F.write_user_faults(self.store.sticks, Path(path).with_suffix(".json"), self.store.tips, **self.store._json_kw())

    def _file_bytes(self, fmt="json"):
        with tempfile.NamedTemporaryFile(suffix="." + fmt, delete=False) as fh:
            pth = fh.name
        (self.to_json if fmt == "json" else self.to_csv)(pth)
        return Path(pth).read_bytes()


    def run_restoration(self, method="cantilever", Te=1500.0, Hd=None):
        """Restore with the current polylines and draw the restored oldest horizon. Returns (figure, metrics, message)."""
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Hd = self.store.Hd if Hd is None else Hd
        fs = self.faults(); r = self.restore(method, Te=float(Te), Hd=float(Hd))
        nu = len(r["ages"]); i = nu - 1
        fig, ax = plt.subplots(figsize=(15, 3.4))
        cols = self.colors if self.colors is not None else section_material_colors(self.units)
        for k in range(i, nu):
            ax.fill_between(r["x"] / 1e3, r["top"][i][k], r["base"][i][k], color=cols[k % len(cols)], alpha=.8, lw=0,
                            label=str(self.units[k]))
        ax.plot(r["x"] / 1e3, r["top"][i][i], "k", lw=1)
        for f in fs:
            ax.axvline(f.x0 / 1e3, color="k", lw=.5, ls="--")
        ax.set_ylim(max(np.nanmax(r["base"][i][i]), 1000) + 300, min(-1000., float(np.nanmin(r["top"][i][i])) - 1000.)); ax.set_xlabel("distance (km)")
        ax.set_ylabel("depth (m)"); ax.set_title(f"{method}, Te {Te:g} m, H_d {Hd / 1e3:g} km: restored {r['ages'][i]:g} Ma")
        ax.legend(loc="lower right", fontsize=8)
        m = flex2d.restoration_metrics(r, i, self.x, fs)
        msg = (f"{method} (Te {Te:g} m, H_d {Hd / 1e3:g} km), {len(fs)} faults - restored {r['ages'][i]:g} Ma top: "
               f"std {m['std']:.0f} m, P95-P5 relief {m['relief_p95_p5']:.0f} m (flatter = better)")
        return fig, m, msg



def vg1_example():
    """Section + tops of the digitized Roberts et al. (1998) Fig. 4a line (for demos)."""
    import pandas as pd
    root = Path(data_path("north_sea", "roberts1998_fig4a_vg1_digitized.csv"))
    d = pd.read_csv(root)
    x = np.arange(1000.0, 81001.0, 250.0); I = lambda c: np.interp(x, d.x_km * 1e3, d[c])
    tops = np.array([I(f"top{k}_m") for k in range(1, 11)]); base = I("base_m")
    units = ["Pliocene-Recent", "Miocene", "Eocene-Oligocene", "Palaeocene", "Upper Cretaceous", "Lower Cretaceous",
             "Kimmeridgian-Ryazanian", "Callovian-Oxfordian", "Bajocian-Bathonian (Brent)", "Triassic-Lower Jurassic"]
    at = [0, 5.3, 23, 56, 66, 100.5, 140, 157.3, 166.1, 170.3]
    lith = [{"shale": 1}] * 4 + [{"chalk": 1}, {"shale": 1}, {"shale": 1}, {"shale": 1}, {"sandstone": 1}, {"shaly_sandstone": 1}]
    sec = flex2d.section_from_horizons(x, tops, base, units, at, at[1:] + [252.0], lith, tops[0])
    return sec, tops, base


def vg1_editor(**kw):
    """The editor on VG1 with the coloured section image (as launched by ``pybasinsubs-tracer``)."""
    sec, tops, base = vg1_example()
    kw.setdefault("colors", section_material_colors(sec.units))
    img, ext = F.render_section_image(sec.x, tops, base, colors=kw["colors"], px=(1600, 500))
    kw.setdefault("title", "VG1 (digitized Roberts et al. 1998 Fig. 4a) - fault-surface editor")
    return FaultTracer(sec, tops, image=img, extent=ext, base=base, **kw)
