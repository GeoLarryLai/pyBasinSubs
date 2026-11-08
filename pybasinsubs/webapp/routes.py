# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""FastAPI routes of the browser app. Thin: the science is in ``pybasinsubs.app_core`` (1-D column, 2-D section)
and ``pybasinsubs.fault_tracer`` (fault editor), the same calls the notebooks use.

Every request that draws a figure runs under one lock (matplotlib is not thread-safe). Figures are returned as
base64 PNG; tables as ``{columns, data}`` (pandas "split" orient, NaN -> null). Errors are HTTP 400 with the message.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import app_core as core  # noqa: E402
from ..input_plots import stratigraphic_column, compaction_relations, interpreted_section  # noqa: E402
from .. import faults as F  # noqa: E402
from .._paths import data_path  # noqa: E402
from ..fault_tracer import PALETTE, RESETS, vg1_editor  # noqa: E402

STATIC = Path(__file__).parent / "static"
TEMPLATES = {"column": ("template", "my_column_template.csv"), "horizons": ("template", "my_section_horizons.csv"),
             "units": ("template", "my_section_units.csv"), "faults": ("template", "my_section_faults.csv")}
HIDDEN_SUMMARY = ("_p2.5", "_p97.5", "_p16", "_p84")       # same columns hidden as in the notebook table
DEFAULTS = dict(n=1000, wscale=1.0, hf=0, Te_km=1.5, Hd_km=8.0, n_mc=50)


def fig_b64(fig, *, tight=True) -> str:
    """Matplotlib figure -> base64 PNG (the figure is closed)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def table_json(df, nd=2) -> dict:
    """DataFrame -> {columns, data} (rounded, NaN -> null)."""
    d = json.loads(df.round(nd).to_json(orient="split", index=False))
    return dict(columns=d["columns"], data=d["data"])


def _text(s: Optional[str]) -> Optional[bytes]:
    return None if s is None or not str(s).strip() else str(s).encode("utf-8")


def create_app(testing: bool = False, faults_path=None):
    """The FastAPI app. ``faults_path``: where the fault editor saves / loads (default: the package user-faults file)."""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="pyBasinSubs")
    app.state.testing = testing
    lock = threading.RLock()
    editor = {"tr": None}

    def tracer():
        if editor["tr"] is None:
            editor["tr"] = vg1_editor(path=faults_path)
        return editor["tr"]

    def guard(fn):
        """Run ``fn`` under the lock; any error becomes HTTP 400 with its message."""
        with lock:
            try:
                return fn()
            except HTTPException:
                raise
            except Exception as e:        # shown in the page
                plt.close("all")
                raise HTTPException(400, f"{type(e).__name__}: {e}")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/options")
    def options():
        tr = tracer()
        from .wizard import curve_catalog
        wells = dict(viking_graben=core.wells("viking_graben"), barents_hammerfest=core.wells("barents_hammerfest"))
        datasets = []
        for source in core.SOURCES:
            if source == "My CSV":
                continue
            key = "viking_graben" if source.startswith("North") else "barents_hammerfest" if source.startswith("Barents") else None
            if key:
                datasets.extend(dict(label=f"{source.replace(' well (Sodir)', '')} {well} (Sodir)", source=source, well=well) for well in wells[key])
            else:
                datasets.append(dict(label=source, source=source, well=""))
        return dict(sources=core.SOURCES, wells=wells, datasets=datasets,
                    section_datasets=[dict(id="vg1", label="Viking Graben VG1")], porosity_curves=curve_catalog(),
                    methods=core.SECTION_METHODS, defaults=DEFAULTS, resets=[[k, v] for k, v in RESETS.items()],
                    units=tr.units, palette=PALETTE)

    @app.get("/api/template/{kind}")
    def template(kind: str):
        if kind not in TEMPLATES:
            raise HTTPException(404, f"unknown template {kind!r}")
        return Response(Path(data_path(*TEMPLATES[kind])).read_text(), media_type="text/csv")

    # ------------------------------------------------------------------ 1-D column
    @app.post("/api/column")
    def column(body: dict = Body(...)):
        def run():
            src = body.get("source", core.SOURCES[0])
            col = core.load_case(src, body.get("well"), _text(body.get("csv")))
            col = core.apply_overrides(col, float(body.get("wscale", 1.0)), body.get("eroded"))
            hf = float(body.get("hf") or 0)
            out = core.run_case(col, n=int(body.get("n", 1000)), mixing=body.get("mixing", "paper"), heat_flow_mW=hf or None)
            s = out["summary"]
            res = dict(status=f"{col.name}: {len(col)} rows, mixing = {body.get('mixing', 'paper')}, MC draws = {int(body.get('n', 1000))}",
                       stratigraphy=fig_b64(stratigraphic_column(col)), compaction=fig_b64(compaction_relations(col)),
                       geohistory=fig_b64(out["fig_geohistory"]), summary=table_json(s), summary_all=table_json(s, 4),
                       maturity=None, maturity_table=None)
            if "fig_maturity" in out:
                res["maturity"] = fig_b64(out["fig_maturity"]); res["maturity_table"] = table_json(out["maturity"])
            res["summary"] = table_json(s[[c for c in s.columns if not any(h in c for h in HIDDEN_SUMMARY)]])
            plt.close("all")
            return res
        return guard(run)

    # ------------------------------------------------------------------ 2-D section
    @app.post("/api/section")
    def section(body: dict = Body(...)):
        def run():
            hz, un, fl = _text(body.get("horizons")), _text(body.get("units")), _text(body.get("faults"))
            if (hz is None) != (un is None):
                raise ValueError("Upload both the horizons and the units CSV (or neither for the template).")
            sec, fs = core.load_section(hz, un, fl, cutoff_planes=bool(body.get("cutoff_planes")))
            o = core.run_section(sec, fs, body.get("method", "cantilever"), Te=float(body.get("Te_km", 1.5)) * 1e3,
                                 Hd=float(body.get("Hd_km", 8.0)) * 1e3, n_mc=int(body.get("n_mc", DEFAULTS["n_mc"])), seed=0, ages=[0.])
            o["fig"].suptitle(f"Present-day section · 0 Ma · {o['method']} (growth reconstruction below is explicit)", fontsize=10)
            status = f"{len(sec.units)} units, {len(sec.x)} columns, {len(fs)} faults; method = {o['method']}"
            res = dict(status=status, figure=fig_b64(interpreted_section(sec, fs)),
                       compaction=fig_b64(compaction_relations(sec)), table=table_json(o["table"], 1))
            plt.close("all")
            return res
        return guard(run)

    @app.post("/api/section/growth")
    def section_growth(body: dict = Body(...)):
        """Explicit fixed-trace export only: never called on load or by /api/section."""
        def run():
            from ..growth_spatial import reconstruct_spatial_growth, binned_spatial_rates
            from ..growth_display import growth_frame_steps, growth_frame_limits, plot_growth_frame
            hz, un, fl = _text(body.get("horizons")), _text(body.get("units")), _text(body.get("faults"))
            if (hz is None) != (un is None):
                raise ValueError("Upload both the horizons and the units CSV (or neither for the template).")
            sec, fs = core.load_section(hz, un, fl, cutoff_planes=False)
            opts = dict(Te=float(body.get("Te_km", 1.5)) * 1e3, Hd=float(body.get("Hd_km", 8.)) * 1e3,
                        make_figures=False,
                        throw_rates_m_Myr=body.get("throw_rates_m_Myr"))
            out = reconstruct_spatial_growth(sec, fs, **opts)
            sensitivity = [reconstruct_spatial_growth(sec, fs, **opts, datum_sigma_m=ds, growth_sigma_m=gs)
                           for ds, gs in [(125., 500.), (175., 700.)]]
            bins = binned_spatial_rates(out, sensitivity=sensitivity)
            limits = growth_frame_limits(out, bins=bins)
            frames = []
            for step in growth_frame_steps(out):
                frames.append(dict(**step, figure=fig_b64(plot_growth_frame(out, step["unit_index"], bins=bins, limits=limits), tight=False),
                                   figure_without_numbers=fig_b64(plot_growth_frame(out, step["unit_index"], bins=bins, limits=limits, show_fault_numbers=False), tight=False)))
            return dict(status="Fixed interpreted traces; strip + decompaction + flexural unloading + spatial non-negative vertical-shear inference. "
                               "One timestep at a time; section + short aligned dual-axis rates; fixed axes across ALL steps; 1-km bins at interpreted INPUT x. "
                               "Normal throw is a distributed vertical proxy, NOT resolved down-dip slip or curved-fault balancing. "
                               "Zero is assumed. No input traces, tips or horizons changed; no assists. "
                               "Bars: spatial / fit-sensitivity range, NOT confidence intervals.",
                        figure=frames[0]["figure"], frames=frames, frame_limits=limits, history=table_json(out.history, 6), bins=table_json(bins, 6),
                        metadata=out.metadata)
        return guard(run)

    # ------------------------------------------------------------------ fault editor
    def fstate(msg=""):
        tr = tracer(); st = tr.store
        sp = st.stored_path()
        return dict(message=msg, extent=[float(v) for v in tr.extent], units=tr.units, palette=PALETTE,
                    faults=[dict(id=k, nodes=v.tolist(), tip=st.tips.get(k), origin=st.origins.get(k, "user"))
                            for k, v in st.sticks.items()],
                    can_undo=bool(st._undo), can_redo=bool(st._redo), ve=float(st.vertical_exaggeration or 4.0),
                    stored=None if sp is None else sp.name, Hd_km=st.Hd / 1e3)

    @app.get("/api/faults/state")
    def faults_state():
        return guard(fstate)

    @app.get("/api/faults/image.png")
    def faults_image():
        def run():
            tr = tracer(); buf = io.BytesIO()
            plt.imsave(buf, np.asarray(tr.image), format="png")
            return Response(buf.getvalue(), media_type="image/png")
        return guard(run)

    @app.get("/api/faults/auto/{kind}")
    def faults_auto(kind: str):
        def run():
            if kind not in F.AUTO_SOURCES:
                raise ValueError(f"unknown auto method {kind!r}")
            tr = tracer(); st = tr.store
            fs = F.auto_faults(kind, tr.x, tr.tops, tr.base, tip_unit=st.tip_unit, Hd=st.Hd)
            s = F.fault_polylines(fs, float(tr.extent[2]), n=30, x=tr.x, ref_top=tr.tops[-1])
            return dict(faults=[dict(id=k, nodes=v.tolist()) for k, v in s.items()])
        return guard(run)

    @app.post("/api/faults/op")
    def faults_op(body: dict = Body(...)):
        def run():
            st = tracer().store; op = body.get("op"); fid = body.get("id"); msg = ""
            if op == "add":
                pts = np.asarray(body["nodes"], float)
                if len(pts) < 2:
                    raise ValueError("a fault needs at least 2 nodes")
                msg = f"added {st.add(pts)}"
            elif op == "delete":
                st.delete(fid); msg = f"deleted {fid}"
            elif op == "rename":
                msg = f"renamed {fid} -> {body['name']}" if st.rename(fid, body["name"]) else "rename: name empty or already used"
            elif op == "set_tip":
                st.set_tip(fid, body.get("tip")); msg = f"{fid}: tip horizon set"
            elif op == "move_node":
                st.move_node(fid, int(body["j"]), body["xz"]); msg = f"{fid}: node {int(body['j'])} moved"
            elif op == "insert_node":
                j = st.insert_node(fid, body.get("j")); msg = f"{fid}: node inserted (#{j})"
            elif op == "delete_node":
                msg = f"{fid}: node deleted" if st.delete_node(fid, int(body["j"])) else "a fault needs at least 2 nodes"
            elif op == "undo":
                msg = "undone" if st.undo() else "nothing to undo"
            elif op == "redo":
                msg = "redone" if st.redo() else "nothing to redo"
            elif op == "reset":
                st.reset(body.get("kind", "curved_midramp")); msg = f"reset to {body.get('kind')} ({len(st.sticks)} faults)"
            elif op == "set_ve":
                st.vertical_exaggeration = float(body["value"])
            else:
                raise ValueError(f"unknown operation {op!r}")
            return fstate(msg)
        return guard(run)

    @app.post("/api/faults/save")
    def faults_save(body: dict = Body(default={})):
        def run():
            p = tracer().store.save(fmt=str(body.get("fmt", "json")).lower())
            return fstate(f"saved {len(tracer().store.sticks)} faults to {Path(p).name}")
        return guard(run)

    @app.post("/api/faults/load")
    def faults_load(body: dict = Body(default={})):
        def run():
            try:
                p = tracer().store.load(fmt=str(body.get("fmt", "json")).lower())
            except FileNotFoundError as e:
                raise ValueError(f"no stored file {Path(str(e)).name}")
            return fstate(f"loaded {len(tracer().store.sticks)} faults from {Path(p).name}")
        return guard(run)

    @app.post("/api/faults/import")
    def faults_import(body: dict = Body(...)):
        def run():
            name = str(body.get("name", "faults.csv")); suf = Path(name).suffix or ".csv"
            with tempfile.NamedTemporaryFile(suffix=suf, delete=False, mode="w", encoding="utf-8") as fh:
                fh.write(body["text"]); pth = fh.name
            try:
                try:
                    s, t = F.read_user_faults(pth)
                except Exception:
                    s, t = F.read_fault_sticks(pth), None
            finally:
                Path(pth).unlink(missing_ok=True)
            tracer().store.edit(s, t, {k: "imported" for k in s})
            return fstate(f"imported {name} ({len(s)} faults)")
        return guard(run)

    @app.get("/api/faults/download/{fmt}")
    def faults_download(fmt: str):
        def run():
            if fmt not in ("json", "csv"):
                raise ValueError("format must be json or csv")
            return Response(tracer()._file_bytes(fmt), media_type="application/octet-stream",
                            headers={"Content-Disposition": f'attachment; filename="user_faults.{fmt}"'})
        return guard(run)

    @app.post("/api/faults/restore")
    def faults_restore(body: dict = Body(...)):
        def run():
            fig, m, msg = tracer().run_restoration(body.get("method", "cantilever"), Te=float(body.get("Te", 1500.0)),
                                                   Hd=float(body.get("Hd_km", 8.0)) * 1e3)
            return dict(figure=fig_b64(fig), message=msg, metrics={k: float(v) for k, v in m.items() if np.isscalar(v)})
        return guard(run)

    @app.post("/api/quit")
    def quit_():
        srv = getattr(app.state, "server", None)
        if srv is not None:
            srv.should_exit = True
        return dict(quitting=True)

    from .wizard import mount_wizard
    mount_wizard(app, guard, fig_b64, table_json)
    return app
