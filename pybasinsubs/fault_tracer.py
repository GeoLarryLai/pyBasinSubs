# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Interactive fault-surface editor (optional dependencies: panel + bokeh).

    pybasinsubs-tracer                               # browser app on the VG1 section (Panel)
    from pybasinsubs.fault_tracer import FaultTracer, vg1_example
    sec, tops, base = vg1_example()
    tr = FaultTracer(sec, tops, base=base)           # loads the stored user faults if the file exists
    tr.panel()                                       # in Jupyter: the editor inline
    tr.faults(); tr.restore("cantilever")            # curved Faults from the polylines / a restoration

Editing (Bokeh toolbar, right of the plot):
* Draw (PolyDraw, pencil): click nodes of a new fault, double-click the last node to finish. Drag a whole fault to move it.
* Edit nodes (PolyEdit): double-click a fault to show its nodes; drag a node; with a node selected, click empty
  space to append a node after it; select a node and press Backspace to delete it.
* Tap (pointer): click a fault to select it (also via the "fault" menu).
Buttons: Insert node (midpoint after the selected node, or in the longest segment), Delete node, Delete fault,
Rename, Tip horizon, Undo / Redo, Reset to (curved mid-ramp | curved footwall crest | straight | last saved | empty),
Save / Load as JSON (default, pybasinsubs/example_data/faults/vg1_user_faults.json) or CSV (vg1_user_faults.csv);
every edit is also autosaved to vg1_user_faults.autosave.json,
vertical exaggeration, background auto faults, Run restoration (method / Te / H_d) with the restored oldest
horizon and its flatness metric.
JSON: {version, section_id, units 'm', vertical_exaggeration, faults: [{id, name, tip_unit, method_origin,
nodes: [[x, z], ...]}], metadata: {created, modified, pybasinsubs_version}}; CSV columns: fault_id, x, z, order,
tip_unit (m, depth positive down).
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np

from . import flex2d, faults as F
from ._paths import data_path

RESETS = {"curved mid-ramp (auto, default)": "curved_midramp", "curved footwall crest (auto)": "curved_crest",
          "straight (auto planar picks)": "straight", "last saved file": "saved", "empty": "empty"}
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def _rgba_uint32(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[2] == 3:
        img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], 2)
    return np.ascontiguousarray(img).view(np.uint32)[..., 0]


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
        self.result = None; self._app = None; self.src = None

    # ------------------------------------------------------------ data access
    def sticks(self):
        return {k: v.copy() for k, v in self.store.sticks.items()}

    def set_sticks(self, sticks: dict, tips=None):
        self.store.edit(sticks, tips); self._push()

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

    # ------------------------------------------------------------ bokeh sync
    def _cds_data(self):
        ks = list(self.store.sticks)
        return dict(xs=[list(self.store.sticks[k][:, 0] / 1e3) for k in ks], ys=[list(self.store.sticks[k][:, 1]) for k in ks],
                    fault_id=ks, color=[PALETTE[i % len(PALETTE)] for i in range(len(ks))])

    def _push(self):
        if self.src is None:
            return
        self._busy = True
        try:
            self.src.data = self._cds_data()
            if hasattr(self, "_w_fault"):
                ks = list(self.store.sticks); self._w_fault.options = ks
                if self._w_fault.value not in ks:
                    self._w_fault.value = ks[0] if ks else None
                self._sync_widgets()
        finally:
            self._busy = False

    def _pull(self, attr, old, new):
        """CDS changed by a Bokeh tool (draw / drag / node edit) -> store (one undo step)."""
        if getattr(self, "_busy", False):
            return
        d = self.src.data; sticks = {}; tips = {}; ids = list(d.get("fault_id", []))
        ids += [""] * (len(d["xs"]) - len(ids))
        for xs, ys, k in zip(d["xs"], d["ys"], ids):
            if len(xs) < 2:
                continue
            k = str(k) if k not in ("", None) and str(k) not in sticks else None
            if k is None:
                k = "F1"; i = 1
                while k in sticks or k in self.store.sticks and k not in ids:
                    i += 1; k = f"F{i}"
            sticks[k] = np.c_[np.asarray(xs, float) * 1e3, np.asarray(ys, float)]
            tips[k] = self.store.tips.get(k)
        changed = set(sticks) != set(self.store.sticks) or any(
            sticks[k].shape != self.store.sticks[k].shape or not np.allclose(sticks[k], self.store.sticks[k]) for k in sticks)
        if changed:
            self.store.edit(sticks, tips)
            if set(sticks) != set(ids) or len(ids) != len(sticks):
                self._push()
            elif hasattr(self, "_w_fault"):
                self._busy = True; self._w_fault.options = list(sticks); self._busy = False
            self._status(f"edited ({len(sticks)} faults, autosaved)")

    def _status(self, s):
        if hasattr(self, "_info"):
            self._info.object = s

    def _sync_widgets(self):
        k = self._w_fault.value
        if k is None:
            return
        self._w_name.value = k
        t = self.store.tips.get(k)
        self._w_tip.value = "default" if t is None else self.units[t]
        ks = list(self.store.sticks)
        if k in ks:
            self.src.selected.indices = [ks.index(k)]

    # ------------------------------------------------------------ UI
    def panel(self):
        import panel as pn
        from bokeh.models import ColumnDataSource, PolyDrawTool, PolyEditTool, TapTool
        from bokeh.plotting import figure
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pn.extension()
        x = self.x / 1e3; zmax = float(np.nanmax(self.tops if self.base is None else self.base)) + 500
        W = 1200
        p = figure(width=W, height=420, title=self.title, x_axis_label="distance (km)", y_axis_label="depth (m)",
                   y_range=(zmax, -200), x_range=(x[0], x[-1]), tools="pan,wheel_zoom,box_zoom,reset,save")
        if self.image is not None:
            x0, x1, zb, zt = self.extent
            p.image_rgba(image=[_rgba_uint32(self.image)], x=x0 / 1e3, y=zt, dw=(x1 - x0) / 1e3, dh=zb - zt, global_alpha=0.75)
        else:
            lay = np.r_[self.tops, [self.base]] if self.base is not None else self.tops
            for k in range(len(lay) - 1):
                p.varea(x, lay[k], lay[k + 1], fill_color=f"C{k % 10}" if False else PALETTE[(k + 3) % 10], fill_alpha=0.35)
        cols = self.colors or ["#333333"] * len(self.tops)
        for k, t in enumerate(self.tops):
            p.line(x, t, color=cols[k % len(cols)], line_width=1)
        bg = ColumnDataSource(dict(xs=[], ys=[]))
        p.multi_line("xs", "ys", source=bg, line_width=1.5, color="black", line_dash="dashed", alpha=0.6)
        self.src = ColumnDataSource(self._cds_data())
        r = p.multi_line("xs", "ys", source=self.src, line_width=3, color="color", selection_line_width=5,
                         nonselection_line_alpha=0.6)
        vsrc = ColumnDataSource(dict(x=[], y=[]))
        vr = p.scatter("x", "y", source=vsrc, size=10, color="black")
        draw = PolyDrawTool(renderers=[r], empty_value=""); edit = PolyEditTool(renderers=[r], vertex_renderer=vr)
        tap = TapTool(renderers=[r])
        p.add_tools(draw, edit, tap); p.toolbar.active_drag = draw
        self.src.on_change("data", self._pull)
        self._vsrc = vsrc

        w = pn.widgets
        ks = list(self.store.sticks)
        self._w_fault = w.Select(name="fault", options=ks, value=ks[0] if ks else None, width=110)
        self._w_name = w.TextInput(name="name", value=ks[0] if ks else "", width=110)
        self._w_tip = w.Select(name="tip horizon (fault dies out at the top of)", options=["default"] + self.units,
                               value="default", width=260)
        b_ren = w.Button(name="Rename", width=80); b_del = w.Button(name="Delete fault", button_type="danger", width=110)
        b_ins = w.Button(name="Insert node", width=100); b_dn = w.Button(name="Delete node", width=100)
        b_undo = w.Button(name="↶ Undo", width=80); b_redo = w.Button(name="↷ Redo", width=80)
        w_reset = w.Select(name="reset to", options=list(RESETS), width=240)
        b_reset = w.Button(name="Reset", button_type="warning", width=80)
        fmt = w.RadioButtonGroup(name="format", options=["JSON", "CSV"], value="JSON", width=110)
        b_save = w.Button(name="💾 Save", button_type="primary", width=90)
        b_load = w.Button(name="📂 Load", width=90)
        dl = w.FileDownload(callback=lambda: io.BytesIO(self._file_bytes(fmt.value.lower())), filename="user_faults.json",
                            label="Download", width=110)
        fmt.param.watch(lambda e: setattr(dl, "filename", "user_faults." + e.new.lower()), "value")
        up = w.FileInput(accept=".csv,.geojson,.json,.txt,.dat", width=220)
        ve = w.FloatSlider(name="vertical exaggeration", start=1, end=10, step=0.5,
                            value=float(self.store.vertical_exaggeration or 4), width=220)
        w_bg = w.Select(name="background auto faults", options=["none", "curved_midramp", "curved_crest", "straight"],
                        value="curved_midramp", width=180)
        method = w.Select(name="method", options=["cantilever", "listric", "domino"], value="cantilever", width=120)
        te = w.Select(name="Te (m)", options=[0, 500, 1000, 1500, 3000, 5000, 10000], value=1500, width=100)
        hd = w.Select(name="H_d (km)", options=[6, 8, 10, 12, 15, 20], value=int(self.store.Hd / 1e3), width=100)
        b_run = w.Button(name="▶ Run restoration", button_type="success", width=160)
        self._info = pn.pane.Markdown(f"Loaded **{len(ks)}** faults "
                                      f"({'user file ' + self.store.stored_path().name if self.store.stored_path() else 'auto curved mid-ramp'}).",
                                      width=W)
        out = pn.pane.Matplotlib(None, dpi=80, tight=True, width=W)

        def set_ve(_=None):
            self.store.vertical_exaggeration = float(ve.value)
            rng_m = (x[-1] - x[0]) * 1e3; h = int(np.clip((W - 80) * (zmax + 200) / rng_m * ve.value + 80, 200, 1100))
            p.height = h
        set_ve()

        def set_bg(_=None):
            if w_bg.value == "none":
                bg.data = dict(xs=[], ys=[]); return
            fs = F.auto_faults(w_bg.value, self.x, self.tops, self.base, tip_unit=self.store.tip_unit, Hd=self.store.Hd)
            s = F.fault_polylines(fs, zmax, n=30, x=self.x, ref_top=self.tops[-1])
            bg.data = dict(xs=[list(v[:, 0] / 1e3) for v in s.values()], ys=[list(v[:, 1]) for v in s.values()])
        set_bg()

        def sel_from_plot(attr, old, new):
            if getattr(self, "_busy", False) or not new:
                return
            ks = list(self.store.sticks)
            if new[0] < len(ks):
                self._w_fault.value = ks[new[0]]
        self.src.selected.on_change("indices", sel_from_plot)

        def cur():
            k = self._w_fault.value
            return k if k in self.store.sticks else None

        def sel_node():
            k = cur()
            if k is None or not vsrc.selected.indices:
                return None
            xv = vsrc.data["x"]; yv = vsrc.data["y"]; j = vsrc.selected.indices[0]
            st = self.store.sticks[k]
            if j < len(xv) and len(xv) == len(st):
                return j
            d = np.hypot(st[:, 0] / 1e3 - xv[j], (st[:, 1] - yv[j]) / 1e3) if j < len(xv) else None
            return None if d is None else int(np.argmin(d))

        def on_fault(e):
            if not getattr(self, "_busy", False):
                self._busy = True; self._sync_widgets(); self._busy = False

        def on_ren(_):
            k = cur()
            if k and self.store.rename(k, self._w_name.value):
                self._push(); self._w_fault.value = self._w_name.value.strip(); self._status(f"renamed {k} → {self._w_fault.value}")
            else:
                self._status("rename: name empty or already used")

        def on_tip(e):
            k = cur()
            if k is None or getattr(self, "_busy", False):
                return
            t = None if e.new == "default" else self.units.index(e.new)
            self.store.set_tip(k, t); self._status(f"{k}: tip horizon = {e.new}")

        def on_del(_):
            k = cur()
            if k:
                self.store.delete(k); self._push(); self._status(f"deleted {k}")

        def on_ins(_):
            k = cur()
            if k:
                j = self.store.insert_node(k, sel_node()); self._push(); self._status(f"{k}: node inserted (#{j})")

        def on_dn(_):
            k = cur(); j = sel_node()
            if k is None:
                return
            if j is None:
                self._status("select a node first (Edit-nodes tool: double-click the fault, click a node)"); return
            ok = self.store.delete_node(k, j); self._push()
            self._status(f"{k}: node #{j} deleted" if ok else "a fault needs at least 2 nodes")

        def on_undo(_):
            self._status("undone" if self.store.undo() else "nothing to undo"); self._push()

        def on_redo(_):
            self._status("redone" if self.store.redo() else "nothing to redo"); self._push()

        def on_reset(_):
            self.store.reset(RESETS[w_reset.value]); self._push(); self._status(f"reset to {w_reset.value} ({len(self.store.sticks)} faults)")

        def on_save(_):
            pth = self.store.save(fmt=fmt.value.lower()); self._status(f"saved {len(self.store.sticks)} faults to `{pth}`")

        def on_load(_):
            try:
                pth = self.store.load(fmt=fmt.value.lower())
            except FileNotFoundError as ex:
                self._status(f"no stored file `{ex}`"); return
            self._push(); self._status(f"loaded {len(self.store.sticks)} faults from `{pth}`")

        def on_up(_):
            if up.value is None:
                return
            suf = Path(up.filename).suffix or ".csv"
            with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as fh:
                fh.write(up.value); pth = fh.name
            try:
                s, t = F.read_user_faults(pth)
            except Exception:
                s, t = F.read_fault_sticks(pth), None
            self.store.edit(s, t, {k: "imported" for k in s}); self._push(); self._status(f"imported {up.filename}")

        def on_run(_):
            self._status("running restoration ..."); kw = dict(Te=float(te.value), Hd=float(hd.value) * 1e3)
            try:
                fs = self.faults(); r = self.restore(method.value, **kw)
            except Exception as ex:  # report in the app
                self._status(f"restoration failed: {type(ex).__name__}: {ex}"); return
            nu = len(r["ages"]); i = nu - 1
            fig, ax = plt.subplots(figsize=(15, 3.4))
            for k in range(i, nu):
                ax.fill_between(r["x"] / 1e3, r["top"][i][k], r["base"][i][k], color=PALETTE[(k + 3) % 10], alpha=.8, lw=0)
            ax.plot(r["x"] / 1e3, r["top"][i][i], "k", lw=1)
            for f in fs:
                ax.axvline(f.x0 / 1e3, color="k", lw=.5, ls="--")
            ax.set_ylim(max(np.nanmax(r["base"][i][i]), 1000) + 300, -300); ax.set_xlabel("distance (km)")
            ax.set_ylabel("depth (m)"); ax.set_title(f"{method.value}, Te {te.value} m, H_d {hd.value} km: restored {r['ages'][i]:g} Ma")
            out.object = fig; plt.close(fig)
            m = flex2d.restoration_metrics(r, i, self.x, fs)
            self._status(f"**{method.value}** (Te {te.value} m, H_d {hd.value} km), {len(fs)} faults - restored {r['ages'][i]:g} Ma "
                         f"top: **std {m['std']:.0f} m**, P95-P5 relief {m['relief_p95_p5']:.0f} m (flatter = better)")

        self._w_fault.param.watch(on_fault, "value"); self._w_tip.param.watch(on_tip, "value")
        for b, f in [(b_ren, on_ren), (b_del, on_del), (b_ins, on_ins), (b_dn, on_dn), (b_undo, on_undo), (b_redo, on_redo),
                     (b_reset, on_reset), (b_save, on_save), (b_load, on_load), (b_run, on_run)]:
            b.on_click(f)
        up.param.watch(on_up, "value"); ve.param.watch(set_ve, "value"); w_bg.param.watch(set_bg, "value")
        if ks:
            self._busy = True; self._sync_widgets(); self._busy = False
        help_ = pn.pane.Markdown(
            "**Toolbar:** ✏️ *Draw* - click nodes, **double-click** to finish a new fault; drag a fault to move it. "
            "*Edit nodes* - **double-click** a fault to show its nodes, **drag** a node, click a node then click empty space "
            "to append, **Backspace** deletes the selected node. *Tap* - click a fault to select it. "
            "Dashed black = auto faults (background). Every edit is autosaved; **Save** / **Load** the stored file as JSON or CSV.", width=W)
        self._app = pn.Column(
            help_, pn.Row(ve, w_bg), pn.pane.Bokeh(p),
            pn.Row(self._w_fault, self._w_name, b_ren, self._w_tip, b_ins, b_dn, b_del),
            pn.Row(b_undo, b_redo, w_reset, b_reset, fmt, b_save, b_load, dl, up),
            pn.Row(method, te, hd, b_run), self._info, out)
        return self._app


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
    img, ext = F.render_section_image(sec.x, tops, base, px=(1600, 500))
    kw.setdefault("title", "VG1 (digitized Roberts et al. 1998 Fig. 4a) - fault-surface editor")
    return FaultTracer(sec, tops, image=img, extent=ext, base=base, **kw)
