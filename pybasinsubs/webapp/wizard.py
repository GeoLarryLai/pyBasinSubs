# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Local wizard snapshots: user inputs -> compaction model -> results -> exports.

Snapshots are immutable copies. Thermal uses the stored backstrip result, and
animation/export uses the stored growth result, never a hidden inversion.
"""
from __future__ import annotations

from copy import deepcopy
import io
import json
import tempfile
from pathlib import Path
from uuid import uuid4
import threading
import zipfile

import numpy as np
import pandas as pd

from .. import app_core as core, thermal
from ..column import Column
from ..input_plots import stratigraphic_column, compaction_relations, interpreted_section
from ..porosity import fit_exponential
from ..timestep import (
    timestep_steps, timestep_column, timestep_thermal_fields,
    plot_geohistory_timestep, plot_thermal_timestep,
    save_geohistory_animation, save_thermal_animation,
    maturity_timestep_data, plot_maturity_timestep, save_maturity_animation,
)


def model_rows(model):
    if isinstance(model, Column):
        names = [n for n, used in zip(model.lith_names, np.any(model.fractions() > 0, axis=0)) if used]
        t = model.lithologies.loc[names, ["phi0", "C"]].copy()
    else:
        used = np.any(model.F > 0, axis=(0, 1))
        t = pd.DataFrame(dict(phi0=model.phi0[used], C=model.C[used]), index=np.asarray(model.lith)[used])
    t.index.name = "lithology"
    return t.reset_index()


def configure_compaction(model, curves=None, points=None, fit_mode="per_lithology"):
    """Apply imported exponential parameters or conservative fits of REAL points.

    Four valid points, three distinct depths spanning >=100 m, physical decreasing
    exponential and r² >=0.5 are required. Otherwise retain the chosen curve and
    show the points. Shared mode applies one supported fit to all used lithologies.
    """
    out = deepcopy(model); table = model_rows(out); notes = []
    if fit_mode not in {"per_lithology", "shared"}:
        raise ValueError("Fit mode must be per_lithology or shared")
    if curves and str(curves).strip():
        incoming = pd.read_csv(io.StringIO(curves)).rename(columns={"name": "lithology"})
        if not {"lithology", "phi0", "C"} <= set(incoming):
            raise ValueError("Curve CSV requires lithology,phi0,C (C in metres)")
        if incoming.lithology.duplicated().any() or not set(incoming.lithology) <= set(table.lithology):
            raise ValueError("Curve lithology names must be unique and match the opened data")
        if not np.isfinite(incoming[["phi0", "C"]].to_numpy(float)).all() or not (incoming.phi0.between(0, 1, inclusive="right") & (incoming.C > 0)).all():
            raise ValueError("Curves require 0 < phi0 <= 1 and positive finite C")
        table = table.set_index("lithology");table.update(incoming.set_index("lithology"));table = table.reset_index()
        notes.append("Chosen curve parameters applied; omitted lithologies retain their opened model.")
    obs = pd.read_csv(io.StringIO(points)) if points and str(points).strip() else pd.DataFrame()
    if not obs.empty:
        if not {"depth_m", "porosity"} <= set(obs):
            raise ValueError("Points CSV requires depth_m,porosity, optional lithology; porosity is a fraction")
        valid = np.isfinite(obs.depth_m) & np.isfinite(obs.porosity) & (obs.depth_m >= 0) & obs.porosity.between(0, 1, inclusive="right")
        if not valid.all():
            raise ValueError("All points must have finite nonnegative depth and 0 < porosity <= 1")
        if fit_mode == "per_lithology" and "lithology" not in obs:
            notes.append("Unlabelled points shown; choose shared fit to apply them. Chosen curves retained.")
            groups = []
        else:
            groups = [("shared", obs)] if fit_mode == "shared" else list(obs.groupby("lithology", sort=False))
        for name, data in groups:
            chosen = np.ones(len(table), bool) if name == "shared" else (table.lithology == name).to_numpy()
            if not chosen.any():
                notes.append(f"{name}: no matching lithology; points shown but model unchanged.");continue
            reason = "at least 4 points, 3 depths spanning 100 m required"
            fit = None
            if len(data) >= 4 and data.depth_m.nunique() >= 3 and np.ptp(data.depth_m) >= 100:
                try:
                    trial = fit_exponential(data.depth_m.to_numpy(), data.porosity.to_numpy())
                    if all(np.isfinite(trial[k]) for k in ["phi0", "C", "r2"]) and 0 < trial["phi0"] <= 1 and trial["C"] > 0 and trial["r2"] >= .5:
                        fit = trial
                    else:
                        reason = "fit is nonphysical or r² below 0.5"
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    reason = "ill-conditioned fit"
            if fit is None:
                notes.append(f"{name}: {reason}; points shown, chosen curve retained.")
            else:
                table.loc[chosen, ["phi0", "C"]] = [fit["phi0"], fit["C"]]
                if isinstance(out, Column):
                    for lit in table.loc[chosen, "lithology"]:
                        out.lithologies.loc[lit, ["phi0_err", "C_err"]] = [fit["phi0_se"], fit["C_se"]]
                notes.append(f"{name}: fitted {len(data)} real points, r²={fit['r2']:.3f}.")
    else:
        notes.append("No measured porosity points supplied; chosen model curves only.")
    for row in table.itertuples():
        if isinstance(out, Column):
            out.lithologies.loc[row.lithology, ["phi0", "C"]] = [row.phi0, row.C]
        else:
            i = out.lith.index(row.lithology);out.phi0[i] = row.phi0;out.C[i] = row.C
    table["equation"] = [f"φ(z) = {v.phi0:.4f} exp(−z / {v.C:.1f} m)" for v in table.itertuples()]
    return out, obs, table, notes


def imported_curve_rows(text):
    """Validate parameter CSV, including optional plot-only reference lithologies."""
    if not text or not str(text).strip():
        return pd.DataFrame(columns=["lithology", "phi0", "C"])
    rows = pd.read_csv(io.StringIO(text)).rename(columns={"name": "lithology"})
    if not {"lithology", "phi0", "C"} <= set(rows):
        raise ValueError("Curve CSV requires lithology,phi0,C (C in metres)")
    if rows.lithology.isna().any() or rows.lithology.duplicated().any() or not rows.lithology.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("Curve lithology names must be nonempty and unique")
    if not np.isfinite(rows[["phi0", "C"]].to_numpy(float)).all() or not (rows.phi0.between(0, 1, inclusive="right") & (rows.C > 0)).all():
        raise ValueError("Curves require 0 < phi0 <= 1 and positive finite C")
    return rows[["lithology", "phi0", "C"]]


def curve_catalog(model=None, references=None):
    """Built-in comparisons plus the actual opened/applied lithologies; no model edits."""
    from ..lithology import default_lithologies
    library = default_lithologies()
    used = model_rows(model) if model is not None else pd.DataFrame(columns=["lithology", "phi0", "C"])
    aliases = {"mudstone": "Taiwan mudstone", "sandstone_tw": "Taiwan sandstone", "marine_avg": "Average marine sediment"}
    records = {}
    for name, row in library.iterrows():
        source = "Lai et al. (2022)" if "Lai et al." in row.source else "Sclater & Christie (1980)"
        records[name] = dict(name=name, label=f"{aliases.get(name, name.replace('_', ' ').capitalize())} — {source}",
                             source=row.source, phi0=float(row.phi0), C=float(row.C), used=False)
    if references is not None:
        for row in references.itertuples():
            records[row.lithology] = dict(name=row.lithology, label=f"{row.lithology} — imported reference", source="Imported curve CSV", phi0=float(row.phi0), C=float(row.C), used=False)
    for row in used.itertuples():
        name = row.lithology
        item = records.get(name, dict(name=name, label=f"{name} — opened/imported model", source="Opened/imported lithology"))
        if name in library.index and not np.allclose([row.phi0, row.C], library.loc[name, ["phi0", "C"]].to_numpy(float), rtol=1e-10, atol=1e-10):
            citation = "Lai et al. (2022)" if "Lai et al." in library.loc[name, "source"] else "Sclater & Christie (1980)"
            item["label"] = f"{name.replace('_', ' ').capitalize()} — opened/applied model (baseline {citation})"
            item["source"] = f"Opened/applied parameters; baseline {library.loc[name, 'source']}"
        item.update(phi0=float(row.phi0), C=float(row.C), used=True)
        records[name] = item
    return list(records.values())


def mount_wizard(app, guard, fig_b64, table_json):
    from fastapi import Body, HTTPException
    from fastapi.responses import Response
    cache = {}
    jobs = {}
    jobs_lock = threading.Lock()
    JOB_LIMIT = 32

    def update_job(job_id, *, percent=None, done=None, total=None, message=None,
                   status="running", error=None):
        if not job_id:
            return
        with jobs_lock:
            rec = jobs.get(job_id) or {
                "percent": 0, "done": 0, "total": 0,
                "message": "Computing properties…", "status": "running",
            }
            if done is not None:
                rec["done"] = int(done)
            if total is not None:
                rec["total"] = int(total)
            if percent is not None:
                rec["percent"] = int(max(0, min(100, round(percent))))
            elif rec.get("total"):
                rec["percent"] = int(max(0, min(100, round(100 * rec["done"] / rec["total"]))))
            if message is not None:
                rec["message"] = message
            rec["status"] = status
            if error is not None:
                rec["error"] = str(error)
            jobs[job_id] = rec
            while len(jobs) > JOB_LIMIT:
                jobs.pop(next(iter(jobs)))

    def job_label(percent, detail=""):
        pct = int(max(0, min(100, round(percent))))
        return f"Computing properties… {pct}%"

    def put(kind, **values):
        key = uuid4().hex
        if len(cache) >= 64:
            del cache[next(iter(cache))]
        cache[key] = dict(kind=kind, **values)
        return key

    def get(key, kind):
        value = cache.get(key)
        if value is None or value["kind"] != kind:
            raise ValueError("This step needs a current earlier snapshot. Reopen/recompute the earlier step.")
        return value

    @app.get("/api/wizard/progress/{job_id}")
    def wizard_progress(job_id: str):
        with jobs_lock:
            rec = jobs.get(job_id)
        if rec is None:
            return dict(percent=0, done=0, total=0, message="", status="unknown")
        return dict(rec)

    def column_frames(result, fields=None, maturity_data=None):
        frames = []
        for st in timestep_steps(result):
            fig = (plot_geohistory_timestep(result, st["index"], slider=False) if fields is None else
                   plot_thermal_timestep(result, st["index"], fields=fields, slider=False))
            frame = dict(**timestep_column(result, st["index"]), figure=fig_b64(fig, tight=False))
            if maturity_data is not None:
                frame["maturity_figure"] = fig_b64(plot_maturity_timestep(result, st["index"], data=maturity_data), tight=False)
            frames.append(frame)
        return frames

    @app.post("/api/wizard/{mode}/input")
    def open_input(mode: str, body: dict = Body(...)):
        def run():
            if mode == "column":
                col = core.load_case("My CSV" if body.get("csv") else body.get("source", core.SOURCES[0]), body.get("well"),
                                     body["csv"].encode() if body.get("csv") else None)
                if body.get("csv"):
                    col.name = body.get("name") or "Edited column"
                col = core.apply_overrides(col, float(body.get("wscale", 1)), body.get("eroded"))
                model = col;files = {"column.csv": col.units.to_csv(index=False)}
                image = fig_b64(stratigraphic_column(col))
                fs = None
            elif mode == "section":
                hz, un, fl = [body.get(k) or None for k in ("horizons", "units", "faults")]
                if (hz is None) != (un is None):
                    raise ValueError("Supply horizons AND units, or leave both blank for VG1")
                model, fs = core.load_section(*[v.encode() if v else None for v in (hz, un, fl)], cutoff_planes=bool(body.get("cutoff_planes")))
                from ..flex2d import read_section_template
                tpl = read_section_template()
                # Preserve supplied geometry verbatim; do not average laterally varying inputs on export.
                files = {"horizons.csv": hz or Path(tpl["horizons"]).read_text(), "units.csv": un or Path(tpl["units"]).read_text(),
                         "faults.csv": fl or (Path(tpl["faults"]).read_text() if hz is None else "fault_id,x,z\n")}
                image = fig_b64(interpreted_section(model, fs))
            else:
                raise ValueError("Unknown analysis")
            files["input_options.json"] = json.dumps({k: body.get(k) for k in ["source", "well", "wscale", "eroded", "cutoff_planes"]}, indent=2)
            ident = put(mode + ":input", model=model, faults=fs, files=files, figures={"input_" + mode + ".png": image})
            return dict(input_id=ident, figure=image, files=files, curves=model_rows(model).to_csv(index=False),
                        curve_options=curve_catalog(model), status="Opened inputs. Inspect/edit them before applying a compaction model.")
        return guard(run)

    @app.post("/api/wizard/{mode}/curve-options")
    def curve_options(mode: str, body: dict = Body(...)):
        def run():
            data = get(body.get("input_id"), mode + ":input")
            rows = imported_curve_rows(body.get("curves"))
            return dict(curve_options=curve_catalog(data["model"], rows[~rows.lithology.isin(model_rows(data["model"]).lithology)]),
                        selected=list(dict.fromkeys(model_rows(data["model"]).lithology.tolist() + rows.lithology.tolist())))
        return guard(run)

    @app.post("/api/wizard/{mode}/compaction")
    def compaction(mode: str, body: dict = Body(...)):
        def run():
            data = get(body.get("input_id"), mode + ":input")
            incoming = imported_curve_rows(body.get("curves"))
            used = model_rows(data["model"]).lithology.tolist()
            applied = incoming[incoming.lithology.isin(used)]
            references = incoming[~incoming.lithology.isin(used)]
            model, obs, table, notes = configure_compaction(data["model"], applied.to_csv(index=False) if len(applied) else None,
                                                          body.get("points"), body.get("fit_mode", "per_lithology"))
            options = curve_catalog(model, references)
            selected = body.get("plot_curves")
            if selected is not None and (not isinstance(selected, list) or any(not isinstance(v, str) for v in selected) or not set(selected) <= {v["name"] for v in options}):
                raise ValueError("plot_curves must list available porosity curve names")
            selected = list(dict.fromkeys(selected)) if selected is not None else used
            extras = pd.DataFrame([dict(lithology=v["name"], phi0=v["phi0"], C=v["C"]) for v in options if not v["used"]])
            files = dict(data["files"], **{"compaction.csv": table.drop(columns="equation").to_csv(index=False),
                         "porosity_plot_options.json": json.dumps(dict(selected=selected, plot_only_references=extras.to_dict(orient="records")), indent=2)})
            if len(obs):
                files["porosity_points.csv"] = obs.to_csv(index=False)
            if len(references):
                files["reference_curves.csv"] = references.to_csv(index=False)
                notes.append("Additional imported lithologies are plot-only references, not added to the decompaction model.")
            image = fig_b64(compaction_relations(model, measured=obs if len(obs) else None, selected_curves=selected, extra_curves=extras,
                                                measured_note="Real observations shown; curves reflect applied fitting/import choices."))
            ident = put(mode + ":model", model=model, faults=deepcopy(data["faults"]), files=files, notes=notes, figures=dict(data["figures"], **{"compaction.png": image}))
            return dict(model_id=ident, figure=image, table=table_json(table, 4), notes=notes, curve_options=options, plot_curves=selected)
        return guard(run)

    @app.post("/api/wizard/{mode}/run")
    def compute(mode: str, body: dict = Body(...)):
        def run():
            data = get(body.get("model_id"), mode + ":model");files = dict(data["files"])
            if mode == "column":
                n = int(body.get("n", 1000))
                if not 0 <= n <= 4000:
                    raise ValueError("Monte Carlo draws must be between 0 and 4000")
                out = core.run_case(deepcopy(data["model"]), n=n, mixing=body.get("mixing", "paper"))
                image = fig_b64(out["fig_geohistory"]);figures = dict(data["figures"], **{"geohistory.png": image})
                files["computation_options.json"] = json.dumps(dict(n=n, mixing=body.get("mixing", "paper")), indent=2)
                files["geohistory.csv"] = out["summary"].to_csv(index=False)
                frames = column_frames(out["result"])
                figures.update({f"geohistory_timestep_{i:02d}.png": st["figure"] for i, st in enumerate(frames)})
                files["column_timesteps.json"] = json.dumps([{k: v for k, v in f.items() if k != "figure"} for f in frames], indent=2)
                ident = put(mode + ":run", result=out["result"], model=data["model"], files=files, figures=figures, options=body)
                return dict(result_id=ident, figure=image, frames=frames, table=table_json(out["summary"]), status=f"Geohistory computed: {n} Monte Carlo draws.")
            options = dict(method=body.get("method", "cantilever"), Te=float(body.get("Te_km", 1.5))*1000,
                           Hd=float(body.get("Hd_km", 8))*1000, n_mc=int(body.get("n_mc", 50)), seed=0, ages=[0.])
            if not 0 <= options["n_mc"] <= 1000 or not 0 <= options["Te"] <= 50000 or not 2000 <= options["Hd"] <= 30000:
                raise ValueError("Check Te, Hd and Monte Carlo ranges")
            out = core.run_section(deepcopy(data["model"]), deepcopy(data["faults"]), **options)
            out["fig"].set_size_inches(11, 4.2)
            out["fig"].suptitle("Computed present-day section · growth reconstruction is a separate explicit action", fontsize=10)
            out["fig"].tight_layout()
            files["computation_options.json"] = json.dumps(options, indent=2)
            image = fig_b64(out["fig"]);files["restoration_diagnostics.csv"] = out["table"].to_csv(index=False)
            ident = put(mode + ":run", model=data["model"], faults=data["faults"], files=files,
                        figures=dict(data["figures"], **{"section.png": image}), options=options)
            return dict(result_id=ident, figure=image, table=table_json(out["table"]), status="Ordinary section computed. Growth has NOT run.")
        return guard(run)

    @app.post("/api/wizard/column/thermal")
    def compute_thermal(body: dict = Body(...)):
        def run():
            data = get(body.get("result_id"), "column:run");result = data["result"]
            ts = float(body.get("surface_temperature_C", 4.));q = float(body.get("heat_flow_mW_m2", 59.)) / 1000
            text = body.get("geothermal_csv") or ""
            note = f"Heat-flow model: {q*1000:g} mW/m²; surface {ts:g} °C."
            if not np.isfinite(ts) or not -30 <= ts <= 60 or not np.isfinite(q) or not 0 < q <= .3:
                raise ValueError("Use finite surface temperature −30…60 °C and heat flow >0…300 mW/m²")
            if text.strip():
                tab = pd.read_csv(io.StringIO(text))
                if {"age_Ma", "heat_flow_mW_m2"} <= set(tab):
                    tab = tab.sort_values("age_Ma")
                    if not np.isfinite(tab[["age_Ma", "heat_flow_mW_m2"]]).all().all() or tab.age_Ma.duplicated().any() or (tab.age_Ma < 0).any() or not tab.heat_flow_mW_m2.between(0, 300, inclusive="right").all():
                        raise ValueError("Heat-flow history needs unique nonnegative ages and 0 < heat flow <=300")
                    q = lambda age: float(np.interp(age, tab.age_Ma, tab.heat_flow_mW_m2)) / 1000
                    note = "Imported heat-flow history; linear interpolation, end values held outside supplied ages."
                elif {"depth_m", "temperature_C"} <= set(tab):
                    z, _, k, _ = thermal.column_profile(result)
                    if not np.isfinite(tab[["depth_m", "temperature_C"]]).all().all() or (tab.depth_m <= 0).any() or (tab.depth_m > z.max()).any():
                        raise ValueError("Temperature points need positive depths within the model conductivity profile")
                    qs = [thermal.heat_flow_from_temperature(t, d, z, k, T_s=ts) for d, t in zip(tab.depth_m, tab.temperature_C)]
                    q = float(np.mean(qs))
                    if not np.isfinite(q) or not 0 < q <= .3:
                        raise ValueError("Temperature observations imply an unsupported heat flow")
                    note = f"Heat flow {q*1000:.2f} mW/m² from imported temperature/depth observations and existing conductivity model; arithmetic mean of point estimates."
                else:
                    raise ValueError("Geothermal CSV: age_Ma,heat_flow_mW_m2 OR depth_m,temperature_C")
            bt = thermal.burial_temperatures(result, heat_flow=q, T_s=ts)
            fig, table = core.maturity_figure(result, bt)
            names = [str(u.unit) for u in result.column.units.itertuples() if u.thickness_m > 0]
            ro = pd.DataFrame({"age_Ma": fig.axes[1].lines[0].get_xdata(), **{n: line.get_ydata() for n, line in zip(names, fig.axes[1].lines)}})
            temps = pd.DataFrame(bt["T"], columns=[str(v) for v in result.column.units.unit]);temps.insert(0, "age_Ma", bt["age"])
            files = dict(data["files"], **{"thermal_present.csv": table.to_csv(index=False), "temperature_history.csv": temps.to_csv(index=False), "maturity_history.csv": ro.to_csv(index=False)})
            if text.strip():
                files["geothermal_inputs.csv"] = text
            files["thermal_conditions.json"] = json.dumps(dict(surface_temperature_C=ts, note=note), indent=2)
            image = fig_b64(fig)
            fields = timestep_thermal_fields(result, heat_flow=q, T_s=ts)
            maturity_data = maturity_timestep_data(result, burial=bt)
            frames = column_frames(result, fields, maturity_data)
            figures = dict(data["figures"], **{"thermal.png": image})
            figures.update({f"thermal_timestep_{i:02d}.png": st["figure"] for i, st in enumerate(frames)})
            figures.update({f"maturity_timestep_{i:02d}.png": st["maturity_figure"] for i, st in enumerate(frames)})
            ident = put("column:thermal", result=result, fields=fields, maturity_data=maturity_data, files=files, figures=figures)
            return dict(thermal_id=ident, figure=image, frames=frames, table=table_json(table), note=note)
        return guard(run)

    @app.post("/api/wizard/section/growth")
    def compute_growth(body: dict = Body(...)):
        def run():
            from ..growth_spatial import reconstruct_spatial_growth, binned_spatial_rates
            from ..growth_display import growth_frame_steps, growth_frame_limits, plot_growth_frame
            data = get(body.get("result_id"), "section:run");opts = data["options"]
            kw = dict(Te=opts["Te"], Hd=opts["Hd"], make_figures=False,
                      assumed_datum_m=0. if body.get("assume_zero", True) else None,
                      contact_placement=body.get("contact_placement", "auto"))
            out = reconstruct_spatial_growth(data["model"], data["faults"], **kw)
            trials = [reconstruct_spatial_growth(data["model"], data["faults"], **kw, datum_sigma_m=d, growth_sigma_m=g) for d, g in [(125.,500.),(175.,700.)]]
            bins = binned_spatial_rates(out, sensitivity=trials);limits = growth_frame_limits(out, bins=bins)
            frames = [dict(**st, figure=fig_b64(plot_growth_frame(out, st["unit_index"], bins=bins, limits=limits), tight=False),
                           figure_without_numbers=fig_b64(plot_growth_frame(out, st["unit_index"], bins=bins, limits=limits, show_fault_numbers=False), tight=False)) for st in growth_frame_steps(out)]
            files = dict(data["files"], **{"growth_history.csv": out.history.to_csv(index=False), "binned_rates.csv": bins.to_csv(index=False), "growth_metadata.json": json.dumps(out.metadata, indent=2)})
            figures = dict(data["figures"], **{f"timestep_{i:02d}.png": st["figure"] for i, st in enumerate(frames)})
            ident = put("section:growth", result=out, model=data["model"], bins=bins, files=files, figures=figures, frames=frames)
            return dict(growth_id=ident, frames=frames, history=table_json(out.history), bins=table_json(bins), metadata=out.metadata,
                        status="Vertical-shear apparent throw PROTOTYPE, not resolved down-dip slip, heave or curved-fault balancing. No tip/trace assist. Ranges are spatial / fitting sensitivity, NOT confidence intervals.")
        return guard(run)

    @app.post("/api/wizard/section/thermal")
    def compute_section_thermal(body: dict = Body(...)):
        def run():
            from ..section_thermal import section_thermal_data, plot_section_thermal_timestep
            # Explicit action on a cached growth snapshot; never rerun restoration.
            data = get(body.get("growth_id"), "section:growth")
            q = float(body.get("heat_flow_mW", 59.)) / 1000
            ts = float(body.get("T_s", 4.))
            if not 0 < q <= .3 or not -20 <= ts <= 60:
                raise ValueError("Heat flow must be 0–300 mW/m²; surface temperature -20–60°C")
            fields = section_thermal_data(data["result"], data["model"], heat_flow=q, T_s=ts)
            frames = [dict(**{k: v for k, v in f.items() if k not in {"T", "ro"}},
                           figure=fig_b64(plot_section_thermal_timestep(fields, i), tight=False))
                      for i, f in enumerate(fields["frames"])]
            files = dict(data["files"], **{
                "section_thermal_conditions.json": json.dumps(fields["conditions"], indent=2),
                "section_thermal_fractions.csv": fields["fractions"].to_csv(index=False)})
            ident = put("section:thermal", result=data["result"], bins=data["bins"],
                        files=files, figures=data["figures"], frames=data["frames"],
                        thermal_fields=fields, thermal_frames=frames)
            return dict(thermal_id=ident, frames=frames,
                        note="Independent steady columns + causal Easy%Ro, not 2-D heat conduction. "
                             "Three 0–100% line strips show one line per supported remaining unit, in its unit color. "
                             "Oldest repeated geometry has no supported thermal reconstruction.")
        return guard(run)

    @app.post("/api/wizard/column/properties")
    def compute_column_properties(body: dict = Body(...)):
        def run():
            from .. import petrophysics as petra
            from ..diagenesis import evolve_properties
            from ..property_plots import (
                plot_property_logs, property_log_limits, property_log_frame_ages,
                _rows_at_time,
            )
            if body.get("thermal_id"):
                data = get(body.get("thermal_id"), "column:thermal")
            else:
                data = get(body.get("result_id"), "column:run")
            result = data["result"]
            ts = float(body.get("surface_temperature_C", 4.))
            q = float(body.get("heat_flow_mW_m2", body.get("heat_flow_mW", 59.))) / 1000
            n_mc = int(body.get("n_mc", 40))
            dt = float(body.get("dt_Myr", 1.0))
            if not 0 <= n_mc <= 400:
                raise ValueError("Property Monte Carlo draws must be between 0 and 400")
            if not np.isfinite(ts) or not -30 <= ts <= 60 or not np.isfinite(q) or not 0 < q <= .3:
                raise ValueError("Use finite surface temperature −30…60 °C and heat flow >0…300 mW/m²")
            job_id = body.get("job_id")
            update_job(job_id, percent=0, message=job_label(0), status="running")

            def on_compute(done, total, message=""):
                pct = 80 * (0 if total <= 0 else done / total)
                update_job(job_id, percent=pct, done=done, total=total,
                           message=job_label(pct, message or "Computing properties"))

            try:
                prop = evolve_properties(result, heat_flow=q, T_s=ts, n_mc=n_mc, dt_Myr=dt,
                                        seed=0, progress=on_compute)
            except Exception as exc:
                update_job(job_id, status="error", message="Computing properties failed", error=exc)
                raise
            limits = property_log_limits(prop)
            cap = body.get("max_frames")
            ages = property_log_frame_ages(prop, max_frames=None if cap is None else int(cap))
            frames, figures = [], dict(data["figures"])
            nF = max(len(ages), 1)
            for i, age in enumerate(ages):
                image = fig_b64(plot_property_logs(prop, age=float(age), limits=limits))
                it = int(np.argmin(np.abs(np.asarray(prop.ages, float) - float(age))))
                layers = []
                for rec in _rows_at_time(prop, it):
                    layers.append(dict(
                        unit_index=rec["j"], unit=rec["name"], top_m=rec["top"],
                        base_m=rec["base"], thickness_m=rec["th"],
                        temperature_C=rec["temperature_C"],
                        porosity=rec["porosity"],
                        phi_mechanical=rec["phi_mechanical"],
                        quartz_cement=rec["quartz_cement"],
                        permeability_mD=float(petra.m2_to_md(rec["permeability_m2"])),
                        effective_stress_mpa=rec["effective_stress_mpa"],
                        pore_pressure_mpa=rec["pore_pressure_mpa"],
                    ))
                frames.append(dict(age_Ma=float(age), label=f"{float(age):g} Ma",
                                   figure=image, layers=layers))
                figures[f"property_logs_{i:02d}.png"] = image
                pct = 80 + 19 * (i + 1) / nF
                update_job(job_id, percent=pct, message=job_label(pct))
            files = dict(data["files"], **{
                "property_present.csv": prop.present.to_csv(index=False),
                "property_table.csv": prop.table.to_csv(index=False),
                "property_conditions.json": json.dumps(dict(
                    heat_flow_mW_m2=q * 1000, surface_temperature_C=ts,
                    n_mc=n_mc, dt_Myr=dt,
                    note=("Schneider et al. (1996) mechanical compaction; "
                          "Walderhaug (1996) quartz cement; "
                          "Kozeny–Carman / Yang & Aplin (2010) permeability; "
                          "Mortensen et al. (1998) / Fabricius (2007) chalk relation. "
                          "Hydrostatic Terzaghi stress. Overlay only."),
                ), indent=2),
            })
            ident = put("column:properties", result=result, prop=prop, files=files,
                        figures=figures, frames=frames,
                        fields=data.get("fields"), maturity_data=data.get("maturity_data"))
            update_job(job_id, percent=100, message=job_label(100), status="done")
            return dict(
                properties_id=ident, frames=frames, table=table_json(prop.present),
                status="Property logs ready. Axis ranges fit every time step.",
            )
        return guard(run)

    @app.post("/api/wizard/section/properties")
    def compute_section_properties(body: dict = Body(...)):
        def run():
            from ..diagenesis import evolve_section_properties
            from ..property_plots import plot_section_property, section_property_limits
            data = get(body.get("growth_id"), "section:growth")
            thermal_data = None
            if body.get("thermal_id"):
                thermal_data = get(body.get("thermal_id"), "section:thermal").get("thermal_fields")
            q = float(body.get("heat_flow_mW", 59.)) / 1000
            ts = float(body.get("T_s", 4.))
            if not 0 < q <= .3 or not -20 <= ts <= 60:
                raise ValueError("Heat flow must be 0–300 mW/m²; surface temperature -20–60°C")
            job_id = body.get("job_id")
            update_job(job_id, percent=0, message=job_label(0), status="running")

            def on_compute(done, total, message=""):
                pct = 70 * (0 if total <= 0 else done / total)
                update_job(job_id, percent=pct, done=done, total=total,
                           message=job_label(pct, message or "Computing properties"))

            try:
                secprop = evolve_section_properties(
                    data["result"], data["model"], heat_flow=q, T_s=ts,
                    thermal_data=thermal_data, progress=on_compute,
                )
            except Exception as exc:
                update_job(job_id, status="error", message="Computing properties failed", error=exc)
                raise
            ages = np.asarray(data["result"].restoration.ages, float)
            variables = ["porosity", "permeability_m2", "effective_stress_mpa"]
            frames_by_variable = {}
            figures = dict(data.get("figures", {}))
            n_plots = max(len(variables) * max(len(ages), 1), 1)
            drawn = 0
            for var in variables:
                limits = section_property_limits(data["result"], secprop, var)
                frames = []
                for i, age in enumerate(ages):
                    image = fig_b64(plot_section_property(
                        data["result"], secprop, age=float(age), variable=var,
                        limits=limits,
                    ), tight=False)
                    frames.append(dict(age_Ma=float(age), label=f"{float(age):g} Ma", figure=image))
                    figures[f"section_{var}_{i:02d}.png"] = image
                    drawn += 1
                    pct = 70 + 29 * drawn / n_plots
                    update_job(job_id, percent=pct, message=job_label(pct))
                frames_by_variable[var] = frames
            files = dict(data["files"], **{
                "section_property_conditions.json": json.dumps(dict(
                    heat_flow_mW=q * 1000, T_s=ts,
                    temperature_source=secprop.temperature_source,
                    note="Same 1-D overlay as the column step, applied independently at each x.",
                ), indent=2),
            })
            ident = put("section:properties", result=data["result"], model=data["model"],
                        secprop=secprop, bins=data.get("bins"), files=files,
                        figures=figures, frames=data.get("frames"),
                        property_frames=frames_by_variable)
            update_job(job_id, percent=100, message=job_label(100), status="done")
            return dict(
                properties_id=ident, frames_by_variable=frames_by_variable,
                variables=variables,
                status="Section properties ready. Colour limits are shared across restoration ages.",
            )
        return guard(run)

    @app.get("/api/wizard/export/{ident}")
    def export(ident: str, animation: bool = False, fault_numbers: bool = True, section_thermal: bool = False, section_properties: bool = False):
        def run():
            import base64
            item = cache.get(ident)
            allowed = {"column:run", "column:thermal", "column:properties",
                       "section:run", "section:growth", "section:thermal", "section:properties"}
            if item is None or item["kind"] not in allowed:
                raise HTTPException(400, "Compute a current result before export")
            if animation and item["kind"] not in {"section:growth", "section:thermal", "section:properties",
                                                   "column:run", "column:thermal", "column:properties"}:
                raise ValueError("Compute growth before exporting an animation")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, text in item["files"].items():
                    archive.writestr(name, text)
                for name, image in item["figures"].items():
                    if not fault_numbers and name.startswith("timestep_"):
                        image = item["frames"][int(name.split("_")[1].split(".")[0])]["figure_without_numbers"]
                    archive.writestr(name, base64.b64decode(image))
                if section_properties and item.get("secprop") is not None:
                    from ..property_plots import save_section_property_animation
                    with tempfile.TemporaryDirectory() as folder:
                        path = save_section_property_animation(
                            item["result"], item["secprop"], Path(folder)/"section_porosity.gif",
                            variable="porosity", show_fault_numbers=fault_numbers)
                        archive.writestr(path.name, path.read_bytes())
                if section_thermal and item["kind"] == "section:thermal":
                    from ..section_thermal import save_section_thermal_animation, plot_section_thermal_timestep
                    for i, frame in enumerate(item["thermal_frames"]):
                        image = frame["figure"] if fault_numbers else fig_b64(
                            plot_section_thermal_timestep(item["thermal_fields"], i, show_fault_numbers=False), tight=False)
                        archive.writestr(f"section_thermal_{i:02d}.png", base64.b64decode(image))
                    with tempfile.TemporaryDirectory() as folder:
                        path = save_section_thermal_animation(item["thermal_fields"], Path(folder)/"section_thermal.gif",
                                                              show_fault_numbers=fault_numbers)
                        archive.writestr(path.name, path.read_bytes())
                if animation:
                    with tempfile.TemporaryDirectory() as folder:
                        folder = Path(folder)
                        if item["kind"].startswith("column:"):
                            path = save_geohistory_animation(item["result"], folder/"geohistory_history.gif")
                            archive.writestr(path.name, path.read_bytes())
                            if item.get("fields") is not None:
                                path = save_thermal_animation(item["result"], folder/"thermal_history.gif", fields=item["fields"])
                                archive.writestr(path.name, path.read_bytes())
                                path = save_maturity_animation(item["result"], folder/"maturity_history.gif", data=item["maturity_data"])
                                archive.writestr(path.name, path.read_bytes())
                            if item.get("prop") is not None:
                                from ..property_plots import save_property_logs_animation
                                path = save_property_logs_animation(item["prop"], folder/"property_logs.gif")
                                archive.writestr(path.name, path.read_bytes())
                        else:
                            from ..growth_display import save_growth_animation
                            path = save_growth_animation(item["result"], folder/"restoration.gif", bins=item["bins"], fps=1, show_fault_numbers=fault_numbers)
                            archive.writestr("restoration.gif", path.read_bytes())
                            if item.get("secprop") is not None and not section_properties:
                                from ..property_plots import save_section_property_animation
                                path = save_section_property_animation(
                                    item["result"], item["secprop"], folder/"section_porosity.gif",
                                    variable="porosity", show_fault_numbers=fault_numbers)
                                archive.writestr(path.name, path.read_bytes())
                archive.writestr("export.json", json.dumps(dict(analysis=item["kind"], computed_from="opened inputs and applied compaction model", figures=list(item["figures"])), indent=2))
            return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="pybasinsubs_results.zip"'})
        return guard(run)
