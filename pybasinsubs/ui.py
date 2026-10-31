# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Panel app (works inside Jupyter and with ``panel serve``).

In a notebook::

    from pybasinsubs.ui import app
    app()            # displays the app inline

Standalone page::

    panel serve pybasinsubs/apps/panel_app.py --show
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import panel as pn

from . import app_core as core


def app():


    """Build the Panel app (data source, mixing, MC draws, water-depth scale, eroded thickness, heat flow) and return it."""
    pn.extension("tabulator", sizing_mode="stretch_width")
    src = pn.widgets.Select(name="Data", options=core.SOURCES, value=core.SOURCES[0])
    well = pn.widgets.Select(name="Well", options=core.wells("viking_graben"), visible=False)
    upload = pn.widgets.FileInput(accept=".csv", visible=False)
    mixing = pn.widgets.RadioButtonGroup(name="Mixing", options=["paper", "mixed"], value="paper")
    n = pn.widgets.IntSlider(name="Monte Carlo draws", start=0, end=4000, step=250, value=1000)
    wscale = pn.widgets.FloatSlider(name="Water-depth scale", start=0.5, end=1.5, step=0.05, value=1.0)
    ero = pn.widgets.FloatInput(name="Eroded thickness (m, blank = data)", value=None)
    hf = pn.widgets.IntSlider(name="Heat flow for maturity (mW/m², 0 = off)", start=0, end=120, value=0)
    run = pn.widgets.Button(name="Run", button_type="primary")
    fig_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", dpi=110, tight=True)
    mat_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", dpi=100, tight=True)
    table = pn.widgets.Tabulator(pagination=None, height=300, disabled=True)
    status = pn.pane.Markdown("")

    def on_src(event):
        v = event.new
        well.visible = v.startswith("North") or v.startswith("Barents")
        if well.visible:
            well.options = core.wells("viking_graben" if v.startswith("North") else "barents_hammerfest")
            well.value = well.options[0]
        upload.visible = v == "My CSV"
    src.param.watch(on_src, "value")

    def do_run(_=None):
        try:
            col = core.load_case(src.value, well.value, upload.value)
            col = core.apply_overrides(col, wscale.value, ero.value)
            out = core.run_case(col, n=n.value, mixing=mixing.value, heat_flow_mW=hf.value or None)
            plt.close("all")
            fig_pane.object = out["fig_geohistory"]
            mat_pane.object = out.get("fig_maturity")
            cols = [c for c in out["summary"].columns if not any(s in c for s in ("_p2.5", "_p97.5", "_p16", "_p84"))]
            table.value = out["summary"][cols].round(2)
            status.object = f"**{col.name}** - {len(col)} rows, mixing = {mixing.value}, MC draws = {n.value}"
        except Exception as e:     # show errors in the app instead of failing silently
            status.object = f"**Error:** {e}"
    run.on_click(do_run)
    do_run()
    side = pn.Column("## pyBasinSubs", src, well, upload, mixing, n, wscale, ero, hf, run, width=320)
    main = pn.Column(status, pn.Tabs(("Geohistory", fig_pane), ("Table", table), ("Maturity", mat_pane)))
    return pn.Tabs(("1-D column", pn.Row(side, main)), ("2-D section", section_app(run_now=True)))


def section_app(run_now=False):
    """2-D section tab: upload horizons/units (+ fault sticks), choose method, Te, H_d, MC draws; view the restored sequence."""
    hz = pn.widgets.FileInput(accept=".csv", name="Horizons CSV")
    un = pn.widgets.FileInput(accept=".csv", name="Units CSV")
    fl = pn.widgets.FileInput(accept=".csv", name="Fault sticks CSV")
    method = pn.widgets.Select(name="Method", options=core.SECTION_METHODS, value="cantilever")
    te = pn.widgets.FloatInput(name="Te (km)", value=1.5, step=0.5, start=0.0)
    hd = pn.widgets.FloatInput(name="H_d (km)", value=8.0, step=1.0, start=2.0)
    nmc = pn.widgets.IntInput(name="MC draws (0 = off)", value=0, step=10, start=0)
    cut = pn.widgets.Checkbox(name="Re-place planes on block cutoffs", value=False)
    run = pn.widgets.Button(name="Run section", button_type="primary")
    fig = pn.pane.Matplotlib(sizing_mode="stretch_width", dpi=100, tight=True)
    table = pn.widgets.Tabulator(pagination=None, height=280, disabled=True)
    status = pn.pane.Markdown("Blank uploads = bundled Viking Graben template (example_data/template/my_section_*.csv).")

    def do_run(_=None):
        try:
            if (hz.value is None) != (un.value is None):
                status.object = "**Upload both horizons and units CSV (or neither).**"; return
            sec, fs = core.load_section(hz.value, un.value, fl.value, cutoff_planes=cut.value)
            o = core.run_section(sec, fs, method.value, Te=te.value * 1e3, Hd=hd.value * 1e3, n_mc=nmc.value)
            plt.close("all"); fig.object = o["fig"]; table.value = o["table"].round(1)
            status.object = f"{len(sec.units)} units, {len(sec.x)} columns, {len(fs)} faults; method = {o['method']}"
        except Exception as e:
            status.object = f"**Error:** {e}"
    run.on_click(do_run)
    if run_now:
        do_run()
    side = pn.Column("### 2-D section", hz, un, fl, method, te, hd, nmc, cut, run, width=320)
    return pn.Row(side, pn.Column(status, fig, table))
