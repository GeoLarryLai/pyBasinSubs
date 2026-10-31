# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Shared engine for the two UIs (Panel/ipywidgets in notebooks, Streamlit).

Both UIs call :func:`load_case` and :func:`run_case`, so they always agree.
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .column import Column, from_basinsubs, from_sodir, read_column
from .backstrip import backstrip
from .plotting import geohistory, maturity_plot
from . import thermal
from ._paths import data_path

SOURCES = ["Coastal Range N (Lai et al. 2022)", "Coastal Range S (Lai et al. 2022)",
           "North Viking Graben well (Sodir)", "Barents Hammerfest well (Sodir)", "My CSV"]


def wells(transect):


    """Well names of a Sodir transect ("viking_graben" or "barents_hammerfest")."""
    w = pd.read_csv(data_path("north_sea", f"sodir_{transect}_wells.csv"))
    return w.well.tolist()


def load_case(source, well=None, csv_bytes=None) -> Column:
    """Build a Column for one of SOURCES (Coastal Range N/S, a Sodir well, or an uploaded CSV; template if none)."""
    if source.startswith("Coastal Range N"):
        return from_basinsubs("N")
    if source.startswith("Coastal Range S"):
        return from_basinsubs("S")
    if source.startswith("North Viking"):
        return from_sodir(well or "30/4-1", "viking_graben")
    if source.startswith("Barents"):
        return from_sodir(well or "7120/9-2", "barents_hammerfest")
    if csv_bytes is None:
        return read_column(data_path("template", "my_column_template.csv"))
    u = pd.read_csv(io.BytesIO(csv_bytes) if isinstance(csv_bytes, (bytes, bytearray)) else csv_bytes)
    return Column(u, name="my column")


def apply_overrides(col: Column, water_scale=1.0, eroded_thickness=None, lith=None, exclude=()) -> Column:
    """Return a copy of the column with water depths scaled, eroded thickness replaced, lithology constants overridden ({name: {col: value}}) and units excluded."""
    c = col.copy()
    u = c.units
    u["water_depth_m"] = u["water_depth_m"] * water_scale
    if eroded_thickness is not None and u.eroded.any():


        u.loc[u.eroded == 1, "thickness_m"] = eroded_thickness
    if exclude:
        keep = ~u.unit.astype(str).isin([str(e) for e in exclude]) | (u.index == u.index[-1])
        u = u[keep].reset_index(drop=True)
    if lith:
        for name, vals in lith.items():
            for k, v in vals.items():
                c.lithologies.loc[name, k] = v
    return Column(u, c.lithologies, name=c.name, lat=c.lat, lon=c.lon, meta=c.meta)


def run_case(col: Column, n=1000, mixing="paper", heat_flow_mW=None, T_s=4.0, seed=0):
    """Backstrip (+ optional burial temperature / Easy%Ro). Returns dict of outputs."""
    res = backstrip(col, n=n, seed=seed, mixing=mixing, mc=n > 0)
    out = dict(result=res, summary=res.summary())
    fig, ax = plt.subplots(figsize=(9, 5))
    pe = col.meta.get("present_elevation_m")
    geohistory(res, ax=ax, title=col.name, present_elevation=pe / 2 if pe else None)
    out["fig_geohistory"] = fig
    if heat_flow_mW:
        bt = thermal.burial_temperatures(res, heat_flow=heat_flow_mW / 1000, T_s=T_s)
        out["fig_maturity"], out["maturity"] = maturity_figure(res, bt)
    return out


def maturity_figure(res, bt):


    """Easy%Ro history for every unit and a present-day table (T, %Ro, window)."""
    u = res.column.units
    idx = [j for j in range(len(u)) if u.thickness_m[j] > 0]
    ages = bt["age"]; o = np.argsort(-ages)
    T = bt["T"][o][:, idx]; a = ages[o]
    ro = np.full_like(T, np.nan)
    for i in range(T.shape[1]):
        m = ~np.isnan(T[:, i])
        if m.sum() >= 2:
            ro[m, i] = thermal.easy_ro(a[m], T[m, i])
    fig = maturity_plot(a, T, ro, [str(u.unit[j]) for j in idx], title=f"{res.column.name}: burial temperature and maturity")
    tab = pd.DataFrame(dict(unit=[str(u.unit[j]) for j in idx], present_T_C=T[-1], present_Ro=ro[-1],
                            window=[thermal.maturity_window(r) if not np.isnan(r) else "" for r in ro[-1]]))
    return fig, tab


# ---------------------------------------------------------------- 2-D section
SECTION_METHODS = ["backstrip (no faults)", "cantilever", "listric", "domino"]


def load_section(horizons=None, units=None, faults=None, tip_unit=None, dx=None, cutoff_planes=False):
    """Section + faults from uploaded CSV bytes (None -> bundled template files).

    faults: fault-stick CSV (fault_id, x, z). tip_unit: index or name of the unit whose TOP is the upper fault tip
    (e.g. the base of the post-rift); default = the unit whose top lies nearest the shallowest stick point.
    cutoff_planes=True re-places the planes on the footwall/hangingwall cutoffs (faults.faults_from_cutoffs)."""
    from . import flex2d
    from . import faults as F
    t = flex2d.read_section_template()
    use_tpl = horizons is None and units is None
    sec = flex2d.section_from_csv(t["horizons"] if horizons is None else horizons,
                                  t["units"] if units is None else units, dx=dx)
    if faults is None and use_tpl:
        faults = t["faults"]
    fs = []
    if faults is not None:
        src = io.BytesIO(faults) if isinstance(faults, (bytes, bytearray)) else faults
        if not isinstance(src, str):
            sticks = {}
            for k, g in pd.read_csv(src).groupby("fault_id", sort=False):
                sticks[str(k)] = g[["x", "z"]].to_numpy(float)
        else:
            sticks = F.read_fault_sticks(src)
        tops = section_tops(sec)
        fs = F.sticks_to_faults(sticks, sec.x, tops[:-1])
        if isinstance(tip_unit, str):
            tip_unit = sec.units.index(tip_unit)
        for f in fs:
            st = np.asarray(f.stick) if f.stick is not None else np.array([[f.x0, 0.0]])
            xt, zt = st[np.argmin(st[:, 1])]
            j = int(np.argmin(np.abs(x_interp(tops[:-1], sec.x, xt) - zt))) if tip_unit is None else int(tip_unit)
            f.tip_unit = min(j, len(sec.units) - 1); f.z_tip = zt
            f.throw = np.where(np.arange(len(f.throw)) <= f.tip_unit, 0.0, f.throw)
        if cutoff_planes and fs:
            fs = F.faults_from_cutoffs(fs, sec.x, tops[:-1], tops[-1], tip_unit=fs[0].tip_unit)
    return sec, fs


def section_tops(sec):
    """(nu + 1, nx) depth of every unit top and of the base."""
    return sec.W0 + np.r_[np.zeros((1, sec.h.shape[1])), np.cumsum(sec.h, 0)]


def x_interp(Z, x, x0):
    return np.array([np.interp(x0, x, z) for z in Z])


def run_section(sec, faults=(), method="cantilever", Te=1500.0, Hd=10e3, n_mc=0, seed=0, ages=None, colors=None, **kw):
    """Restore a section and draw the restored sequence (one panel per age). Returns dict(result, fig, table, mc)."""
    from . import flex2d
    from .faultfit import plot_fault_planes
    m = "backstrip" if (method.startswith("backstrip") or not faults) else method
    if m == "backstrip":
        r = flex2d.backstrip_section(sec, Te=Te); top, base, ages_all = r.top, r.base, r.ages
    else:
        r = flex2d.restore(sec, list(faults), method=m, Te=Te, Hd=Hd, **kw)
        top, base, ages_all = np.asarray(r.top), np.asarray(r.base), np.asarray(r.ages)
    mc = flex2d.monte_carlo_section(sec, list(faults) if m != "backstrip" else None, None if m == "backstrip" else m,
                                    n=n_mc, seed=seed, Te=Te, **({} if m == "backstrip" else dict(Hd=Hd))) if n_mc else None
    nt = len(ages_all)
    idx = list(range(nt)) if ages is None else [int(np.argmin(np.abs(ages_all - a))) for a in ages]
    idx = idx if len(idx) <= 6 else list(np.unique(np.linspace(0, nt - 1, 6).round().astype(int)))
    cols = colors or [plt.cm.tab20(k % 20) for k in range(len(sec.units))]
    fig, axs = plt.subplots(len(idx), 1, figsize=(11, 2.1 * len(idx)), sharex=True, squeeze=False)
    xs = sec.x / 1e3
    for a, i in zip(axs[:, 0], idx):
        for k in range(len(sec.units)):
            if not np.all(np.isnan(top[i, k])):
                a.fill_between(xs, top[i, k], base[i, k], color=cols[k], lw=0, label=sec.units[k] if i == idx[0] else None)
        if mc is not None:
            for k in range(len(sec.units)):
                lo, _, hi = mc.envelope(min(i, len(mc.ages) - 1), k)
                if not np.all(np.isnan(lo)):
                    a.fill_between(xs, lo, hi, color="k", alpha=.18, lw=0)
        if i == 0 and m != "backstrip":
            plot_fault_planes(a, list(faults), Hd, "planar", sec.x, section_tops(sec)[:-1], labels=False, color="k", lw=1)
        a.set_ylim(np.nanmax(base) * 1.05, min(0.0, np.nanmin(top)) - 100); a.set_ylabel("depth (m)")
        a.set_title(f"{ages_all[i]:g} Ma", loc="left", fontsize=9)
    axs[0, 0].legend(fontsize=6, ncol=5, loc="lower right"); axs[-1, 0].set_xlabel("distance (km)")
    fig.suptitle(f"Restored sequence: {m}, Te = {Te / 1e3:g} km" + (f", H_d = {Hd / 1e3:g} km" if m != "backstrip" else "")
                 + (f", {n_mc} MC draws (grey = 5-95 %)" if mc else ""), fontsize=10)
    fig.tight_layout()
    tab = pd.DataFrame(dict(age_Ma=ages_all, mean_seafloor_m=np.nanmean(np.asarray(r.seafloor), 1),
                            seafloor_std_m=np.nanstd(np.asarray(r.seafloor), 1)))
    if m != "backstrip":
        met = [flex2d.restoration_metrics(r, i, sec.x, list(faults), xlim=(sec.x[0], sec.x[-1])) for i in range(nt)]
        tab = pd.concat([tab, pd.DataFrame(met).drop(columns=[c for c in ("age_Ma",) if c in pd.DataFrame(met)])], axis=1)
    return dict(result=r, fig=fig, table=tab, mc=mc, method=m)
