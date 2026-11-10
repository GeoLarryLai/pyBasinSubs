# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Figures for phase-1 property evolution (opt-in; existing plots are unchanged)."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from .material_colors import VG1_MATERIAL_COLORS
from .plotting import _style
from . import petrophysics as _petra
from .diagenesis import mechanical_porosity, T_CEMENT_ONSET_C
from .constants import G, RHO_W

# North Sea group names aligned with the VG1 material palette where possible.
# One place to change property colours. Keys accept both short and stored names.
PROPERTY_CMAPS = {
    "porosity": "copper",
    "phi_mechanical": "copper",
    "permeability": "cividis",
    "permeability_m2": "cividis",
    "permeability_mD": "cividis",
    "temperature_C": "inferno",
    "effective_stress_mpa": "viridis",
    "pore_pressure_mpa": "Blues",
    "quartz_cement": "YlOrBr",
    "vertical_stress_mpa": "viridis",
}


def cmap_for(variable):
    """Return the matplotlib colormap name for a stored or short variable name."""
    return PROPERTY_CMAPS.get(variable, "viridis")


GROUP_COLORS = {
    "NORDLAND": VG1_MATERIAL_COLORS["Pliocene-Recent"],
    "HORDALAND": VG1_MATERIAL_COLORS["Miocene"],
    "ROGALAND": VG1_MATERIAL_COLORS["Palaeocene"],
    "SHETLAND": VG1_MATERIAL_COLORS["Upper Cretaceous"],
    "CROMER KNOLL": VG1_MATERIAL_COLORS["Lower Cretaceous"],
    "VIKING": VG1_MATERIAL_COLORS["Kimmeridgian-Ryazanian"],
    "BRENT": VG1_MATERIAL_COLORS["Bajocian-Bathonian (Brent)"],
    "DUNLIN": VG1_MATERIAL_COLORS["Triassic-Lower Jurassic"],
    "STATFJORD": "#8c564b",
    "HEGRE": "#c49c94",
    "ERODED CENOZOIC": "#7f7f7f",
    "SOTBAKKEN": VG1_MATERIAL_COLORS["Palaeocene"],
    "NYGRUNNEN": VG1_MATERIAL_COLORS["Upper Cretaceous"],
    "ADVENTDALEN": VG1_MATERIAL_COLORS["Kimmeridgian-Ryazanian"],
    "KAPP TOSCANA": VG1_MATERIAL_COLORS["Bajocian-Bathonian (Brent)"],
    "SASSENDALEN": VG1_MATERIAL_COLORS["Triassic-Lower Jurassic"],
    "TEMPELFJORDEN": "#e377c2",
}


def unit_color(name, index=0):
    key = str(name).replace(" GP", "").strip()
    if key in GROUP_COLORS:
        return GROUP_COLORS[key]
    cmap = plt.get_cmap("tab10")
    return cmap(index % 10)


def _unit_colors(units):
    return [unit_color(u, i) for i, u in enumerate(units)]


def _active_units(prop):
    """Units that have at least one finite porosity sample and nonzero thickness."""
    names = []
    for j, name in enumerate(prop.units):
        if not np.isfinite(prop.best["porosity"][:, j]).any():
            continue
        if j < len(prop.column.units) and prop.column.units.thickness_m[j] <= 0:
            continue
        names.append((j, name))
    return names


def plot_geohistory_porosity(prop, ax=None, title=None, cmap=None,
                             vmin=None, vmax=None, show_column=True):
    """Geohistory bands colored by evolving porosity, plus a present-day column.

    The right-hand column is what a vertical time slider would show at the
    selected age (here: present day). Colour defaults to
    :data:`PROPERTY_CMAPS` (copper for porosity).
    """
    if cmap is None:
        cmap = cmap_for("porosity")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    result = prop.backstrip
    b = result.best
    u = result.column.units
    ages_n = b["age"][0]
    sea_n = b["W"][0] - b["SL"][0]
    t = prop.ages
    if ax is None:
        if show_column:
            fig, (ax, axc) = plt.subplots(
                1, 2, figsize=(10.6, 5.4),
                gridspec_kw={"width_ratios": [4.6, 1.0]},
            )
        else:
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            axc = None
    else:
        fig = ax.figure
        axc = None
    vals = prop.best["porosity"]
    finite = vals[np.isfinite(vals)]
    if vmin is None:
        vmin = max(0.0, np.percentile(finite, 2) - 0.02) if len(finite) else 0.0
    if vmax is None:
        vmax = min(0.7, np.percentile(finite, 98) + 0.02) if len(finite) else 0.5
    norm = mpl.colors.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)

    # Paleo-water column on the geohistory (same convention as plotting.geohistory).
    order = np.argsort(-ages_n)
    ages_o = ages_n[order]
    surf = -b["SL"][0][order]
    sf = sea_n[order]
    ax.fill_between(ages_o, surf, sf, where=sf >= surf, interpolate=True,
                    color="#6fb7e8", alpha=0.25, lw=0, zorder=0)

    for j, name in _active_units(prop):
        if "top_bsl_m" in prop.best:
            top = prop.best["top_bsl_m"][:, j]
            th = prop.best["thickness_m"][:, j]
            bot = top + th
        else:
            top_bsf = np.array([_safe_interp(ages_n, b["tops"][0, :, j], tt) for tt in t])
            th = np.array([_safe_interp(ages_n, b["thick"][0, :, j], tt) for tt in t])
            sea = np.array([_safe_interp(ages_n, sea_n, tt) for tt in t])
            top = top_bsf + sea
            bot = top + th
        phi = vals[:, j]
        ok = np.isfinite(top) & np.isfinite(bot) & np.isfinite(phi)
        if ok.sum() < 2:
            continue
        # Colour each time step as a thin vertical slice of the band.
        idx = np.where(ok)[0]
        for a, b_ in zip(idx[:-1], idx[1:]):
            xa = [t[a], t[b_]]
            ax.fill_between(xa, [top[a], top[b_]], [bot[a], bot[b_]],
                            color=cm(norm(0.5 * (phi[a] + phi[b_]))), lw=0, zorder=2)
        ax.plot(t[ok], bot[ok], color="k", lw=0.35, alpha=0.55, zorder=3)

    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0.0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(max(float(u.age_base_Ma.max()), float(np.nanmax(ages_n))), 0)
    ax.invert_yaxis()
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Depth below sea level (m)")
    _style(ax)
    ax.set_title(title or f"{result.column.name}: geohistory coloured by porosity",
                 loc="left", fontweight="bold")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    if axc is not None:
        _present_column(prop, axc, cm, norm)
        cax = make_axes_locatable(axc).append_axes("right", size="18%", pad=0.12)
        cb = fig.colorbar(sm, cax=cax)
    else:
        cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("Porosity (fraction)")
    fig.tight_layout()
    return fig


def _present_column(prop, ax, cm, norm):
    """Thin present-day column coloured by present porosity (slider companion)."""
    pres = prop.present.set_index("unit")
    y = 0.0
    sea = float(prop.backstrip.best["W"][0, -1] - prop.backstrip.best["SL"][0, -1])
    # Younger units are first in the column table.
    for j, name in _active_units(prop):
        if name not in pres.index:
            continue
        # thickness at present from the last finite sample of the backstrip
        b = prop.backstrip.best
        th = b["thick"][0, -1, j]
        if not np.isfinite(th):
            # last finite node
            col = b["thick"][0, :, j]
            ok = np.where(np.isfinite(col))[0]
            if len(ok) == 0:
                continue
            th = col[ok[-1]]
        phi = float(pres.loc[name, "porosity"])
        ax.bar(0, th, bottom=sea + y, width=0.8, color=cm(norm(phi)),
               edgecolor="k", lw=0.4, align="center")
        if th > 140:
            ax.text(0, sea + y + 0.5 * th, name, ha="center", va="center",
                    fontsize=6.5, color="white" if phi < 0.25 else "#222")
        else:
            ax.text(0.48, sea + y + 0.5 * th, name, ha="left", va="center",
                    fontsize=6, color="#333")
        y += th
    ax.set_xlim(-0.7, 1.15)
    ax.set_ylim(ax.figure.axes[0].get_ylim())
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.set_title("Present day\n(slider would\npick an age)", loc="left", fontsize=9)
    _style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelleft=False)


def _safe_interp(ages, values, t):
    a = np.asarray(ages, float)
    v = np.asarray(values, float)
    m = np.isfinite(a) & np.isfinite(v)
    if m.sum() == 0:
        return np.nan
    amin, amax = float(a[m].min()), float(a[m].max())
    if t < amin - 1e-9 or t > amax + 1e-9:
        return np.nan
    if m.sum() == 1:
        return float(v[m][0])
    oo = np.argsort(a[m])
    return float(np.interp(t, a[m][oo], v[m][oo]))


def _normal_curve(phi0, phi_r, beta, rho_g, z_max=6000.0, dz=20.0):
    """Hydrostatic Schneider reference curve (virgin loading)."""
    z = np.arange(0.0, z_max + dz, dz)
    phi = np.empty_like(z)
    sig = 0.0
    p = phi0
    for i, zi in enumerate(z):
        phi[i] = p
        if i + 1 < len(z):
            rho_b = p * RHO_W + (1.0 - p) * rho_g
            sig += max(rho_b - RHO_W, 0.0) * G * dz
            p = mechanical_porosity(sig, phi0, phi_r, beta)
    return z, phi


def plot_porosity_depth_paths(prop, ax=None, title=None):
    """Porosity–depth paths of each unit on top of normal compaction curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 7.0))
    else:
        fig = ax.figure
    priors = prop.priors
    # Reference curves for the two main clastic end-members.
    styles = [("sandstone", "#6a51a3", "Sandstone normal compaction"),
              ("shale", "#238b45", "Shale / mudstone normal compaction")]
    for key, color, lab in styles:
        r = priors.loc[key]
        rho_g = 2650.0 if key == "sandstone" else 2720.0
        z, phi = _normal_curve(r.phi0, r.phi_r, r.beta_1_MPa, rho_g)
        ax.plot(phi, z, color=color, lw=1.6, ls="--", label=lab, zorder=1)

    onset_handles = False
    unload_seen = False
    for j, name in _active_units(prop):
        z = prop.best["depth_bsf_m"][:, j]
        phi = prop.best["porosity"][:, j]
        T = prop.best["temperature_C"][:, j]
        ok = np.isfinite(z) & np.isfinite(phi)
        if ok.sum() < 2:
            continue
        c = unit_color(name, j)
        ax.plot(phi[ok], z[ok], color=c, lw=2.0, label=name, zorder=3)
        ax.plot(phi[ok][-1], z[ok][-1], "o", color=c, ms=6, zorder=4)
        # Cement onset: first time T reaches 80 °C, and only where sandstone is present.
        f_sand = 0.0
        if getattr(prop, "mixed", None) is not None and j < len(prop.mixed):
            f_sand = float(prop.mixed.f_sandstone.iloc[j])
        hit = ok & np.isfinite(T) & (T >= T_CEMENT_ONSET_C)
        if hit.any() and f_sand > 0:
            i0 = np.where(hit)[0][0]
            ax.plot(phi[i0], z[i0], marker="*", ms=12, color=c, markeredgecolor="k",
                    markeredgewidth=0.4, zorder=5)
            onset_handles = True
        # Unloading: depth shallows while porosity stays nearly constant.
        dz = np.diff(z[ok])
        if np.nanmin(dz) < -50:
            i_up = np.where(ok)[0][1:][dz < -50]
            ax.plot(phi[i_up], z[i_up], marker="v", ms=6, color=c,
                    markeredgecolor="k", lw=0, zorder=5)
            unload_seen = True

    ax.set_xlabel("Porosity (fraction)")
    ax.set_ylabel("Mid-point depth below seafloor (m)")
    ax.set_xlim(0, 0.70)
    zmax = np.nanmax(prop.best["depth_bsf_m"])
    ax.set_ylim(max(6000, zmax * 1.05) if np.isfinite(zmax) else 6000, 0)
    extra = []
    if onset_handles:
        extra.append(Line2D([], [], marker="*", ls="None", color="k", ms=10,
                            label=f"Quartz-cement onset ({T_CEMENT_ONSET_C:.0f} °C)"))
    if unload_seen:
        extra.append(Line2D([], [], marker="v", ls="None", color="k", ms=6,
                            label="Unloading (porosity frozen at σ′_max)"))
    else:
        extra.append(Line2D([], [], ls="None", marker=None,
                            label="No unloading path in this well"))
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + extra, lab + [e.get_label() for e in extra],
              fontsize=7.5, loc="lower right", frameon=False)
    _style(ax)
    ax.set_title(title or f"{prop.column.name}: porosity–depth paths",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_property_timeseries(prop, title=None):
    """Four panels: porosity, log permeability, effective stress, pore pressure."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True)
    panels = [
        (axes[0, 0], "porosity", "Porosity (fraction)", False, 1.0),
        (axes[0, 1], "permeability_m2", "Permeability (mD)", True, 1.0),
        (axes[1, 0], "effective_stress_mpa", "Effective stress (MPa)", False, 1.0),
        (axes[1, 1], "pore_pressure_mpa", "Pore pressure (MPa)", False, 1.0),
    ]
    t = prop.ages
    for ax, var, ylab, logk, _ in panels:
        for j, name in _active_units(prop):
            y = prop.best[var][:, j]
            if logk:
                y = _petra.m2_to_md(y)
            ok = np.isfinite(y) & np.isfinite(t)
            if ok.sum() < 2:
                continue
            c = unit_color(name, j)
            ax.plot(t[ok], y[ok], color=c, lw=1.8, label=name)
            if prop.p10 is not None and var in prop.p10:
                lo = prop.p10[var][:, j]
                hi = prop.p90[var][:, j]
                if logk:
                    lo = _petra.m2_to_md(lo)
                    hi = _petra.m2_to_md(hi)
                band = ok & np.isfinite(lo) & np.isfinite(hi)
                if band.any():
                    ax.fill_between(t[band], lo[band], hi[band], color=c, alpha=0.18, lw=0)
        if logk:
            ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_xlim(float(np.nanmax(t)), 0)
        _style(ax)
    axes[0, 0].legend(fontsize=7, ncol=2, loc="best", frameon=False)
    axes[1, 0].set_xlabel("Age (Ma)")
    axes[1, 1].set_xlabel("Age (Ma)")
    band_note = (f"Shaded bands are P10–P90 from {prop.n_mc} Monte Carlo draws of the "
                 "lithology priors"
                 + (" and backstrip burial depths." if prop.mc_used else "."))
    if not prop.mc_used:
        band_note = "Deterministic run only (no P10–P90 bands)."
    fig.suptitle(title or f"{prop.column.name}: property evolution through time",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, -0.01, band_note, fontsize=8, style="italic")
    fig.tight_layout()
    return fig


def plot_porosity_partition(prop, ax=None, title=None):
    """Present-day porosity-loss breakdown: mechanical vs quartz cement vs remaining φ."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
    else:
        fig = ax.figure
    pres = prop.present.copy()
    # Drop basement-like rows.
    pres = pres[pres.unit.isin([n for _, n in _active_units(prop)])]
    if pres.empty:
        ax.set_axis_off()
        return fig
    labels = pres.unit.tolist()
    x = np.arange(len(pres))
    mech = np.clip(pres.mech_loss.to_numpy(float), 0, None)
    cem = np.clip(pres.quartz_cement.to_numpy(float), 0, None)
    rem = np.clip(pres.porosity.to_numpy(float), 0, None)
    ax.bar(x, mech, color="#8c510a", label="Mechanical compaction (φ₀ − φ_mech)")
    ax.bar(x, cem, bottom=mech, color="#d8b365", label="Quartz cement (Vq)")
    ax.bar(x, rem, bottom=mech + cem, color="#01665e", label="Remaining porosity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Volume fraction of bulk rock")
    ax.set_ylim(0, max(0.55, float(np.nanmax(mech + cem + rem)) + 0.05))
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _style(ax)
    ax.set_title(title or f"{prop.column.name}: present-day porosity partition",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_calibration_layout(prop, measured=None, title=None):
    """Predicted vs measured layout. Bundled 30/4-1 data have no measured φ or Pp.

    ``measured`` is an optional DataFrame with columns variable, measured, source,
    and optionally unit or depth_bsf_m. Without it, only the temperature panel
    can show a real point (corrected BHT at TD, if the column metadata has one).
    The other panels are clearly labelled as a layout example.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    specs = [
        (axes[0], "porosity", "Porosity (fraction)", 0, 0.55),
        (axes[1], "pore_pressure_mpa", "Pore pressure (MPa)", 0, None),
        (axes[2], "temperature_C", "Temperature (°C)", 0, None),
    ]
    pres = prop.present
    # Predicted present-day ranges for axis limits.
    tmax = np.nanmax(pres.temperature_C) if len(pres) else 200
    pmax = np.nanmax(pres.pore_pressure_mpa) if len(pres) else 80
    specs[1] = (axes[1], "pore_pressure_mpa", "Pore pressure (MPa)", 0, max(20, pmax * 1.15))
    specs[2] = (axes[2], "temperature_C", "Temperature (°C)", 0, max(50, tmax * 1.15))

    meas = measured if measured is not None else _bht_calibration_points(prop)
    for ax, var, lab, lo, hi in specs:
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1 : 1")
        pts = meas[meas.variable == var] if meas is not None and len(meas) else meas
        if pts is not None and len(pts):
            ax.scatter(pts.measured, pts.predicted, s=45, c="#c0392b", zorder=3)
            for _, r in pts.iterrows():
                ax.annotate(str(r.source), (r.measured, r.predicted),
                            fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.set_title(lab, loc="left", fontsize=10)
        else:
            ax.text(0.5, 0.52, "Layout example only", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#555", fontweight="bold")
            ax.text(0.5, 0.40, "No measured " + lab.split(" (")[0].lower()
                    + "\nis bundled for this well.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#666")
            ax.set_title(lab + "  ·  no data", loc="left", fontsize=10)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        _style(ax)
    fig.suptitle(title or (f"{prop.column.name}: predicted vs measured "
                           "(layout example — no measured porosity or pressure is bundled)"),
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def _bht_calibration_points(prop):
    """One real temperature point from the Sodir BHT, if metadata exist."""
    meta = prop.column.meta or {}
    if "bht_degC" not in meta or "td_bsf_m" not in meta:
        return pd_empty()
    from .thermal import bht_correct, column_profile, steady_geotherm
    z = float(meta["td_bsf_m"])
    Tc, _, assumed = bht_correct(meta["bht_degC"], z, None, T_s=4.0)
    Tc = float(Tc[0])
    zg, phi, k, _ = column_profile(prop.backstrip)
    Tpred = float(np.interp(z, zg, steady_geotherm(zg, k, prop.heat_flow, 4.0)))
    note = "12 h assumed; on 1:1 by construction (q from this BHT)" if assumed[0] else "on 1:1 by construction (q from this BHT)"
    src = f"corrected BHT at TD (Waples & Ramly 2001; {note})"
    import pandas as pd
    return pd.DataFrame([dict(variable="temperature_C", measured=Tc, predicted=Tpred,
                              source=src, unit="TD")])


def pd_empty():
    import pandas as pd
    return pd.DataFrame(columns=["variable", "measured", "predicted", "source", "unit"])


def _nearest_time_index(prop, age):
    t = np.asarray(prop.ages, float)
    return int(np.argmin(np.abs(t - float(age))))


def _rows_at_time(prop, it):
    """Present units at time index it, sorted shallow to deep (BSF)."""
    rows = []
    for j, name in _active_units(prop):
        th = prop.best["thickness_m"][it, j]
        top = prop.best["top_bsf_m"][it, j]
        if not (np.isfinite(th) and th > 1e-6 and np.isfinite(top)):
            continue
        rec = dict(j=j, name=name, top=float(top), th=float(th), base=float(top + th))
        for var in ("porosity", "phi_mechanical", "quartz_cement", "permeability_m2",
                    "effective_stress_mpa", "pore_pressure_mpa", "vertical_stress_mpa",
                    "temperature_C"):
            rec[var] = float(prop.best[var][it, j])
            if prop.p10 is not None and var in prop.p10:
                rec[var + "_p10"] = float(prop.p10[var][it, j])
                rec[var + "_p90"] = float(prop.p90[var][it, j])
        rows.append(rec)
    rows.sort(key=lambda r: r["top"])
    return rows


def _step_xy(rows, key):
    """Depth edges and a stepwise (x, y) pair for a per-unit property."""
    if not rows:
        return np.array([]), np.array([]), np.array([])
    edges = np.array([rows[0]["top"]] + [r["base"] for r in rows], float)
    vals = np.array([r[key] for r in rows], float)
    y = np.repeat(edges, 2)[1:-1]
    x = np.repeat(vals, 2)
    return x, y, edges


def _fill_band(ax, rows, key, color, alpha=0.22):
    lo_key, hi_key = key + "_p10", key + "_p90"
    if not rows or lo_key not in rows[0]:
        return
    lo, y, _ = _step_xy(rows, lo_key)
    hi, _, _ = _step_xy(rows, hi_key)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    if ok.sum() >= 2:
        ax.fill_betweenx(y[ok], lo[ok], hi[ok], color=color, alpha=alpha, lw=0, zorder=1)


def _label_depths(y, zmax):
    y = np.asarray(y, float).copy()
    gap = max(zmax / max(22, len(y) + 1), 40.0)
    for k in range(len(y) - 2, -1, -1):
        y[k] = min(y[k], y[k + 1] - gap)
    return y


def _ro_at_index(prop, j, it):
    from .thermal import easy_ro
    T = prop.best["temperature_C"][:it + 1, j]
    ages = prop.ages[:it + 1]
    ok = np.isfinite(T) & np.isfinite(ages)
    if ok.sum() < 2:
        return np.nan
    return float(easy_ro(ages[ok], T[ok])[-1])


def _draw_strat_column(ax, prop, rows, zmax):
    """ICS-coloured present (or snapshot) column matching input_plots style."""
    from matplotlib.patches import Rectangle
    from .plotting import ics_color

    u = prop.column.units
    mids = np.array([0.5 * (r["top"] + r["base"]) for r in rows])
    lab_y = _label_depths(mids, zmax)
    for rec, ytxt in zip(rows, lab_y):
        mid_age = 0.5 * (float(u.age_top_Ma[rec["j"]]) + float(u.age_base_Ma[rec["j"]]))
        color = ics_color(mid_age)[0]
        ax.add_patch(Rectangle((0.0, rec["top"]), 0.42, rec["th"],
                               facecolor=color, edgecolor="white", linewidth=1.0, zorder=2))
        ax.annotate(
            rec["name"], (0.42, rec["top"] + 0.5 * rec["th"]),
            xytext=(0.52, ytxt), ha="left", va="center", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": ".55", "lw": 0.5},
        )
        ax.axhline(rec["top"], color=".8", lw=0.6, zorder=1)
    if rows:
        ax.axhline(rows[-1]["base"], color=".8", lw=0.6, zorder=1)
    ax.set_xlim(0, 1.55)
    ax.set_xticks([])
    ax.set_ylabel("Depth below seafloor (m)")
    ax.set_title("Stratigraphic column", loc="left", fontsize=10)
    _style(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _guide_lines(ax, rows):
    for rec in rows:
        ax.axhline(rec["top"], color=".8", lw=0.6, zorder=0)
    if rows:
        ax.axhline(rows[-1]["base"], color=".8", lw=0.6, zorder=0)


PROPERTY_LOG_FIGSIZE = (13.4, 7.2)
PROPERTY_LOG_WIDTH_RATIOS = [1.15, 1.35, 1.2, 1.4, 1.2]


def property_log_limits(prop):
    """Fixed axis limits that fit every time step of a property result.

    Depth, porosity, permeability, stress, temperature and Easy%Ro ranges are
    taken from the full history so later frames never jump the axes. Call once
    and pass the dict to :func:`plot_property_logs` and
    :func:`save_property_logs_animation`.

    Returns
    -------
    dict
        ``ylim`` (deep, shallow), ``porosity``, ``permeability_mD``,
        ``stress_mpa``, ``temperature_C`` and ``easy_ro`` as ``(low, high)``.
    """
    zmax = 0.0
    for it in range(len(prop.ages)):
        rows = _rows_at_time(prop, it)
        if rows:
            zmax = max(zmax, max(r["base"] for r in rows))
    zmax = max(float(zmax) * 1.04, 100.0)

    def _finite(name):
        v = np.asarray(prop.best[name], float)
        return v[np.isfinite(v)]

    phi = _finite("porosity")
    phi_m = _finite("phi_mechanical")
    both = np.r_[phi, phi_m] if (len(phi) or len(phi_m)) else np.array([])
    phi_hi = max(0.65, float(np.nanmax(both)) * 1.04) if len(both) else 0.65

    k = _petra.m2_to_md(_finite("permeability_m2"))
    k = k[np.isfinite(k) & (k > 0)]
    if len(k):
        kmin = 10 ** np.floor(np.log10(np.nanmin(k)) - 0.15)
        kmax = 10 ** np.ceil(np.log10(np.nanmax(k)) + 0.15)
        kmin = float(np.clip(kmin, 1e-9, 1e-4))
        kmax = float(max(kmax, 1.0))
    else:
        kmin, kmax = 1e-7, 1e2

    sv = _finite("vertical_stress_mpa")
    pp = _finite("pore_pressure_mpa")
    se = _finite("effective_stress_mpa")
    stress_stack = np.r_[sv, pp, se] if (len(sv) or len(pp) or len(se)) else np.array([])
    stress_hi = max(10.0, float(np.nanmax(stress_stack)) * 1.05) if len(stress_stack) else 10.0

    T = _finite("temperature_C")
    T_hi = max(50.0, float(np.nanmax(T)) * 1.05) if len(T) else 50.0

    ro_hi = 2.0
    last = len(prop.ages) - 1
    ros = [_ro_at_index(prop, j, last) for j, _name in _active_units(prop)]
    ros = [r for r in ros if np.isfinite(r)]
    if ros:
        ro_hi = max(2.0, float(np.max(ros)) * 1.15)

    return {
        "ylim": (zmax, 0.0),
        "porosity": (0.0, phi_hi),
        "permeability_mD": (kmin, kmax),
        "stress_mpa": (0.0, stress_hi),
        "temperature_C": (0.0, T_hi),
        "easy_ro": (0.0, ro_hi),
    }


def property_log_frame_ages(prop, max_frames=None):
    """Ages (Ma, oldest first) at backstrip / stratigraphic time steps.

    Defaults to the same constraint-node ages as
    :func:`pybasinsubs.timestep.timestep_steps` (unit-boundary ages, including
    an empty deposition-start surface when that node exists). This is one
    frame per model time step, not the finer integration grid used internally
    by :func:`~pybasinsubs.diagenesis.evolve_properties`.

    ``max_frames`` is an optional cap; it is applied only when there are more
    constraint nodes than the cap (unusual for a well column).
    """
    from .timestep import timestep_steps

    if getattr(prop, "backstrip", None) is not None:
        ages = np.array([float(s["age_Ma"]) for s in timestep_steps(prop.backstrip)], float)
    else:
        ages = np.asarray(prop.ages, float).copy()
    n = len(ages)
    if n == 0:
        return ages
    if max_frames is None:
        return ages
    max_frames = int(max_frames)
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if n <= max_frames:
        return ages
    idx = np.unique(np.round(np.linspace(0, n - 1, max_frames)).astype(int))
    idx[0] = 0
    idx[-1] = n - 1
    return ages[idx]


def plot_property_logs(prop, age=0.0, title=None, show_ro=True, ylim=None,
                       limits=None, _figure=None):
    """Thin stratigraphic column plus log-style property tracks at one age.

    Tracks share the depth axis: total and mechanical porosity with quartz-cement
    shading, log permeability, effective stress with pore pressure and overburden,
    and temperature (optional causal Easy%Ro on a twin axis). P10–P90 bands are
    drawn when Monte Carlo results are present.

    Axis ranges default to :func:`property_log_limits` so a present-day call
    already uses ranges that fit the whole history. Pass the same ``limits``
    dict to every frame to keep the layout fixed. Figure size is
    ``PROPERTY_LOG_FIGSIZE`` (13.4 × 7.2 in).
    """
    if limits is None:
        limits = property_log_limits(prop)
    it = _nearest_time_index(prop, age)
    rows = _rows_at_time(prop, it)
    if _figure is None:
        fig, axes = plt.subplots(
            1, 5, figsize=PROPERTY_LOG_FIGSIZE, sharey=True,
            gridspec_kw={"width_ratios": PROPERTY_LOG_WIDTH_RATIOS},
        )
    else:
        fig = _figure
        for extra in list(fig.axes):
            if extra.get_label() != "property_age":
                fig.delaxes(extra)
        for extra in list(fig.texts):
            extra.remove()
        fig._suptitle = None
        for legend in list(fig.legends):
            legend.remove()
        axes = fig.subplots(
            1, 5, sharey=True,
            gridspec_kw={"width_ratios": PROPERTY_LOG_WIDTH_RATIOS},
        )
    axc, axp, axk, axs, axt = axes
    zmax, zmin = limits["ylim"]
    if ylim is not None:
        zmax, zmin = ylim

    _draw_strat_column(axc, prop, rows, zmax)
    axc.set_ylim(zmax, zmin)

    # Porosity: remaining (solid), mechanical (dashed), cement shade between them.
    _guide_lines(axp, rows)
    xp, yp, _ = _step_xy(rows, "porosity")
    xm, ym, _ = _step_xy(rows, "phi_mechanical")
    if len(xp):
        ok = np.isfinite(xp) & np.isfinite(xm)
        if ok.any():
            axp.fill_betweenx(yp[ok], xp[ok], xm[ok], color="#d8b365",
                              alpha=0.55, lw=0, zorder=2, label="Quartz cement")
        _fill_band(axp, rows, "porosity", "#8c510a")
        axp.plot(xm, ym, color="#8c510a", ls="--", lw=1.4, label="Mechanical only", zorder=3)
        axp.plot(xp, yp, color="#01665e", lw=1.8, label="Total porosity", zorder=4)
    axp.set_xlabel("Porosity (fraction)")
    axp.set_xlim(*limits["porosity"])
    axp.set_title("Porosity", loc="left", fontsize=10)
    handles, labels = axp.get_legend_handles_labels()
    if handles:
        axp.legend(handles, labels, fontsize=7, frameon=False, loc="lower right")
    _style(axp)

    # Permeability (mD), log scale.
    _guide_lines(axk, rows)
    for rec in rows:
        rec["permeability_mD"] = float(_petra.m2_to_md(rec["permeability_m2"]))
        if "permeability_m2_p10" in rec:
            rec["permeability_mD_p10"] = float(_petra.m2_to_md(rec["permeability_m2_p10"]))
            rec["permeability_mD_p90"] = float(_petra.m2_to_md(rec["permeability_m2_p90"]))
    xk, yk, _ = _step_xy(rows, "permeability_mD")
    _fill_band(axk, rows, "permeability_mD", "#4a306d")
    if len(xk):
        axk.plot(np.clip(xk, 1e-8, None), yk, color="#4a306d", lw=1.8)
    axk.set_xscale("log")
    axk.set_xlabel("Permeability (mD)")
    axk.set_xlim(*limits["permeability_mD"])
    axk.set_title("Permeability", loc="left", fontsize=10)
    _style(axk)

    # Stress: overburden, pore pressure, effective stress.
    _guide_lines(axs, rows)
    styles = [
        ("vertical_stress_mpa", "#333333", "-", 1.8, "Overburden σᵥ"),
        ("pore_pressure_mpa", "#2b6cb0", "-", 1.8, "Pore pressure (hydrostatic)"),
        ("effective_stress_mpa", "#b45309", "--", 1.6, "Effective stress σ′"),
    ]
    for key, color, ls, lw, lab in styles:
        xs, ys, _ = _step_xy(rows, key)
        if key == "effective_stress_mpa":
            _fill_band(axs, rows, key, color)
        if len(xs):
            axs.plot(xs, ys, color=color, ls=ls, lw=lw, label=lab)
    axs.set_xlabel("Stress / pressure (MPa)")
    axs.set_xlim(*limits["stress_mpa"])
    axs.set_title("Stress", loc="left", fontsize=10)
    handles, labels = axs.get_legend_handles_labels()
    if handles:
        axs.legend(handles, labels, fontsize=7, frameon=False, loc="upper right")
    _style(axs)

    # Temperature, with optional causal Easy%Ro.
    _guide_lines(axt, rows)
    xt, yt, _ = _step_xy(rows, "temperature_C")
    _fill_band(axt, rows, "temperature_C", "#b2182b")
    if len(xt):
        axt.plot(xt, yt, color="#b2182b", lw=1.8, label="Temperature")
    axt.axvline(T_CEMENT_ONSET_C, color="#b2182b", ls=":", lw=0.9, alpha=0.8,
                label=f"{T_CEMENT_ONSET_C:.0f} °C onset")
    axt.set_xlabel("Temperature (°C)")
    axt.set_xlim(*limits["temperature_C"])
    axt.set_title("Temperature", loc="left", fontsize=10)
    if show_ro and rows:
        axr = axt.twiny()
        ro_rows = [dict(r, easy_ro=_ro_at_index(prop, r["j"], it)) for r in rows]
        xr, yr, _ = _step_xy(ro_rows, "easy_ro")
        if len(xr) and np.isfinite(xr).any():
            axr.plot(xr, yr, color="#7a4e0c", ls="--", lw=1.3, label="Easy%Ro")
            axr.set_xlabel("Easy%Ro (Sweeney & Burnham 1990)", color="#7a4e0c", fontsize=8)
            axr.set_xlim(*limits["easy_ro"])
            axr.tick_params(axis="x", colors="#7a4e0c", labelsize=8)
            axr.spines["top"].set_color("#7a4e0c")
        handles, labels = axt.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        axt.legend(handles + h2, labels + l2, fontsize=7, frameon=False, loc="upper left")
    else:
        axt.legend(fontsize=7, frameon=False, loc="upper left")
    _style(axt)
    axt.spines["top"].set_visible(True)

    age_used = float(prop.ages[it])
    fig.suptitle(
        title or f"{prop.column.name}: property logs at {age_used:g} Ma",
        x=0.01, ha="left", fontweight="bold", fontsize=12,
    )
    note = ("Blocky tracks are the per-unit model (one value per unit). "
            "Dashed porosity is mechanical-only; the gold shade is quartz cement.")
    if prop.mc_used:
        note += f" Shaded bands are P10–P90 from {prop.n_mc} Monte Carlo draws."
    fig.text(0.01, 0.01, note, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def plot_property_logs_snapshots(prop, ages=(140.0, 66.0, 0.0), title=None,
                                 limits=None):
    """Compact multi-age preview (column + porosity + permeability).

    The notebook and app use :func:`plot_property_logs` for every time step.
    This helper remains for a smaller side-by-side comparison and shares the
    depth / porosity / permeability limits of :func:`property_log_limits`.
    """
    if limits is None:
        limits = property_log_limits(prop)
    zmax = limits["ylim"][0]
    n = len(ages)
    fig, axes = plt.subplots(
        n, 3, figsize=(10.8, 3.15 * n), sharex=False,
        gridspec_kw={"width_ratios": [1.05, 1.25, 1.15]},
    )
    if n == 1:
        axes = np.array([axes])
    for i, age in enumerate(ages):
        it = _nearest_time_index(prop, age)
        rows = _rows_at_time(prop, it)
        axc, axp, axk = axes[i]
        axc.set_ylim(zmax, 0)
        axp.set_ylim(zmax, 0)
        axk.set_ylim(zmax, 0)
        _draw_strat_column(axc, prop, rows, zmax)
        axc.set_title(f"{float(prop.ages[it]):g} Ma", loc="left", fontsize=10, fontweight="bold")
        if i < n - 1:
            axc.set_ylabel("Depth BSF (m)")

        _guide_lines(axp, rows)
        xp, yp, _ = _step_xy(rows, "porosity")
        xm, ym, _ = _step_xy(rows, "phi_mechanical")
        if len(xp):
            ok = np.isfinite(xp) & np.isfinite(xm)
            if ok.any():
                axp.fill_betweenx(yp[ok], xp[ok], xm[ok], color="#d8b365", alpha=0.55, lw=0)
            axp.plot(xm, ym, color="#8c510a", ls="--", lw=1.2)
            axp.plot(xp, yp, color="#01665e", lw=1.6)
        axp.set_xlim(*limits["porosity"])
        _style(axp)
        if i == n - 1:
            axp.set_xlabel("Porosity (fraction)")

        for rec in rows:
            rec["permeability_mD"] = float(_petra.m2_to_md(rec["permeability_m2"]))
        _guide_lines(axk, rows)
        xk, yk, _ = _step_xy(rows, "permeability_mD")
        if len(xk):
            axk.plot(np.clip(xk, 1e-8, None), yk, color="#4a306d", lw=1.6)
        axk.set_xscale("log")
        axk.set_xlim(*limits["permeability_mD"])
        _style(axk)
        if i == n - 1:
            axk.set_xlabel("Permeability (mD)")
        for ax in (axp, axk):
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")
    fig.suptitle(
        title or f"{prop.column.name}: property logs through time",
        x=0.01, ha="left", fontweight="bold",
    )
    fig.text(0.01, 0.01,
             "Same depth scale (present-day total). Gold = quartz cement between "
             "mechanical (dashed) and total (solid) porosity.",
             fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


def _section_stage_index(out, age):
    ages = np.asarray(out.restoration.ages, float)
    return int(np.argmin(np.abs(ages - float(age))))


def _section_limits(out, stages):
    r = out.restoration
    deep, shallow = [], []
    for s in stages:
        deep.append(float(np.nanmax(r.base[s])) + 250)
        shallow.append(min(-1000.0, float(np.nanmin(r.top[s])) - 400))
    return (r.x[0] / 1e3, r.x[-1] / 1e3), (max(deep), min(shallow))


def _draw_section_property(ax, out, secprop, stage, variable, norm, cmap,
                           show_fault_numbers=True):
    """Colour restored material polygons by a per-unit, per-x property."""
    from .growth_display import display_material_polygons, restored_fault_traces

    r = out.restoration
    vals = secprop.at_stage(stage, variable)
    for p in display_material_polygons(out, stage):
        unit = p["unit_index"]
        xx = np.asarray(p["x_m"], float)
        top = np.asarray(p["top_m"], float)
        base = np.asarray(p["base_m"], float)
        v = np.interp(xx, r.x, vals[unit], left=np.nan, right=np.nan)
        _fill_unit_strip(ax, xx / 1e3, top, base, v, norm, cmap)
        ax.plot(xx / 1e3, top, color="k", lw=0.35, alpha=0.55, zorder=3)
    traces = restored_fault_traces(out, stage)
    if len(traces):
        for _, t in traces.groupby(["fault", "segment"], sort=False):
            ax.plot(t.x_m / 1e3, t.restored_z_m, color="0.14", lw=1.0, zorder=5)
        if show_fault_numbers:
            for number, fault in enumerate(out.faults, 1):
                t = traces[traces.fault == fault.label]
                if len(t):
                    tip = t.loc[t.restored_z_m.idxmax()]
                    ax.annotate(
                        f"#{number}", (tip.x_m / 1e3, tip.restored_z_m),
                        xytext=(3, 3), textcoords="offset points", fontsize=7,
                        color="0.12", clip_on=True,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
                        zorder=7,
                    )
    datum = out.metadata.get("assumed_datum_m")
    if datum is not None:
        ax.axhline(datum, color="navy", ls=":", lw=0.75)


def _property_norm(secprop, variable, stages):
    """Shared colour limits from the requested restoration stages."""
    samples = []
    display_var = variable
    scale = 1.0
    log = False
    if variable in {"permeability_m2", "permeability", "permeability_mD"}:
        display_var = "permeability_m2"
        log = True
        scale = 1.0 / 9.869233e-16  # to mD
    for s in stages:
        v = secprop.at_stage(s, display_var) * scale
        samples.append(v[np.isfinite(v)])
    data = np.concatenate([s for s in samples if len(s)]) if samples else np.array([0.0, 1.0])
    if log:
        finite = data[np.isfinite(data) & (data > 0)]
        if len(finite):
            lo = 10 ** np.floor(np.log10(np.nanmin(finite)) - 0.1)
            hi = 10 ** np.ceil(np.log10(np.nanmax(finite)) + 0.1)
            lo = float(max(lo, 1e-8))
            hi = float(max(hi, lo * 10.0))
        else:
            lo, hi = 1e-6, 1e2
        return mpl.colors.LogNorm(lo, hi), display_var, scale, True
    if variable == "porosity":
        phi_hi = 0.60
        if len(data):
            phi_hi = max(0.60, float(np.nanmax(data)) * 1.02)
        return mpl.colors.Normalize(0.0, phi_hi), display_var, scale, False
    if "stress" in variable or "pressure" in variable:
        vmax = max(10.0, float(np.nanmax(data)) * 1.02) if len(data) else 10.0
        return mpl.colors.Normalize(0.0, vmax), display_var, scale, False
    lo, hi = np.percentile(data, [2, 98]) if len(data) else (0.0, 1.0)
    return mpl.colors.Normalize(lo, hi), display_var, scale, False


def plot_section_properties(out, secprop, ages=(0.0, 66.0, 140.0),
                            variables=("porosity", "permeability_m2", "effective_stress_mpa"),
                            title=None):
    """Restored section coloured by overlay properties at a few restoration ages."""
    labels = {
        "porosity": "Porosity (fraction)",
        "permeability_m2": "Permeability (mD)",
        "effective_stress_mpa": "Effective stress (MPa)",
        "temperature_C": "Temperature (°C)",
        "quartz_cement": "Quartz cement (fraction)",
        "pore_pressure_mpa": "Pore pressure (MPa)",
    }
    stages = [_section_stage_index(out, a) for a in ages]
    xlim, ylim = _section_limits(out, stages)
    nR, nC = len(stages), len(variables)
    fig, axes = plt.subplots(
        nR + 1, nC, figsize=(4.4 * nC + 1.2, 3.15 * nR + 1.15), squeeze=False,
        gridspec_kw={"height_ratios": [0.10] + [1.0] * nR, "hspace": 0.22, "wspace": 0.12},
    )
    norms = [_property_norm(secprop, var, stages) for var in variables]
    for j, var in enumerate(variables):
        norm, _, _, _ = norms[j]
        cmap = cmap_for(var)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        cb = fig.colorbar(sm, cax=axes[0, j], orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.tick_params(labelsize=7)
        cb.set_label(labels.get(var, var), fontsize=9)
    for i, stage in enumerate(stages):
        age_used = float(out.restoration.ages[stage])
        for j, var in enumerate(variables):
            ax = axes[i + 1, j]
            norm, display_var, scale, _ = norms[j]
            cmap = cmap_for(var)
            if scale != 1.0:
                class _Scaled:
                    def __init__(self, inner, sc):
                        self.inner, self.sc = inner, sc
                    def at_stage(self, s, name):
                        return self.inner.at_stage(s, name) * self.sc
                src = _Scaled(secprop, scale)
            else:
                src = secprop
            _draw_section_property(ax, out, src, stage, display_var, norm, cmap,
                                   show_fault_numbers=(j == 0))
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            _style(ax)
            if j == 0:
                ax.set_ylabel(f"{age_used:g} Ma\ndepth (m)", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if i == nR - 1:
                ax.set_xlabel("interpreted input distance (km)", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
    fig.suptitle(
        title or "VG1 section: phase-1 porosity, permeability and effective stress",
        x=0.01, ha="left", fontweight="bold", fontsize=12,
    )
    fig.text(
        0.01, 0.01,
        "Same 1-D per-unit model at each x. Unit contacts and restored fault traces "
        "as in the existing 2-D plots. Colour scales are shared down each column. "
        "Permeability uses cividis on a log scale; chalk uses the Mortensen / "
        "Fabricius Kozeny form (c = 0.25, S_BET = 2 m²/g literature prior).",
        fontsize=8, style="italic",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.08, hspace=0.28, wspace=0.14)
    return fig



def _fill_unit_strip(ax, xx_km, top, base, values, norm, cmap):
    """Continuous polygon fill of one unit strip without vertical white seams.

    Adjacent quads share vertices and are drawn with matching face/edge colours
    so hairline gaps between columns do not appear.
    """
    from matplotlib.collections import PolyCollection

    xx = np.asarray(xx_km, float)
    top = np.asarray(top, float)
    base = np.asarray(base, float)
    v = np.asarray(values, float)
    n = len(xx)
    if n < 2:
        return
    cm = plt.get_cmap(cmap)
    verts, facecolors = [], []
    for i in range(n - 1):
        vi = 0.5 * (v[i] + v[i + 1])
        if not (np.isfinite(vi) and np.isfinite(xx[i]) and np.isfinite(xx[i + 1])
                and np.isfinite(top[i]) and np.isfinite(top[i + 1])
                and np.isfinite(base[i]) and np.isfinite(base[i + 1])):
            continue
        verts.append([
            (xx[i], top[i]), (xx[i + 1], top[i + 1]),
            (xx[i + 1], base[i + 1]), (xx[i], base[i]),
        ])
        facecolors.append(cm(norm(vi)))
    if verts:
        ax.add_collection(PolyCollection(
            verts, facecolors=facecolors, edgecolors=facecolors,
            linewidths=0.45, antialiaseds=False, zorder=1,
        ))


def plot_property_logs_view(prop, *, limits=None, ages=None, max_frames=None,
                            show_ro=True):
    """Interactive Matplotlib slider for the fixed-layout property-log frame.

    Oldest ages sit at the left of the slider, matching
    :func:`pybasinsubs.timestep.timestep_view`. Returns ``(figure, slider)``;
    keep both alive. Inline-only notebook backends show a still — export a GIF
    with :func:`save_property_logs_animation` there instead.
    """
    from matplotlib.widgets import Slider

    limits = property_log_limits(prop) if limits is None else limits
    ages = np.asarray(
        property_log_frame_ages(prop, max_frames) if ages is None else ages,
        float,
    )
    if len(ages) < 1:
        raise ValueError("no ages to display")
    values = -ages
    index = len(ages) - 1
    fig = plt.figure(figsize=PROPERTY_LOG_FIGSIZE)
    plot_property_logs(prop, age=float(ages[index]), limits=limits,
                       show_ro=show_ro, _figure=fig)
    sa = fig.add_axes((0.10, 0.015, 0.80, 0.025), label="property_age")
    slider = Slider(sa, "Age (Ma)", float(values[0]), float(values[-1]),
                    valinit=float(values[-1]), valstep=np.unique(values))
    slider.valtext.set_text(f"{ages[index]:g}")
    sa.plot(values, np.full(len(values), 0.5), "|", color=".4", ms=10, zorder=3)
    sa.set_xlim(values[0], values[-1])

    def draw(i):
        nonlocal index
        index = i
        plot_property_logs(prop, age=float(ages[i]), limits=limits,
                           show_ro=show_ro, _figure=fig)
        slider.valtext.set_text(f"{ages[i]:g}")
        fig.canvas.draw_idle()

    slider.on_changed(lambda value: draw(int(np.argmin(np.abs(values - value)))))

    def key(event):
        if event.key in ("left", "right"):
            i = int(np.clip(index + (1 if event.key == "right" else -1),
                            0, len(ages) - 1))
            slider.eventson = False
            slider.set_val(values[i])
            slider.eventson = True
            draw(i)

    fig.canvas.mpl_connect("key_press_event", key)
    fig._property_log_slider = slider
    return fig, slider


def _write_animation(frames_rgb, path, fps):
    """Write RGB PIL frames as GIF (Pillow) or MP4 (ffmpeg if present)."""
    from PIL import Image

    path = Path(path)
    suffix = path.suffix.lower()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if suffix == ".gif":
        palette = frames_rgb[-1].quantize(colors=256)
        images = [im.quantize(palette=palette) for im in frames_rgb]
        images[0].save(
            path, save_all=True, append_images=images[1:],
            duration=round(1000 / fps), loop=0, optimize=False, disposal=2,
        )
        return path
    if suffix == ".mp4":
        import shutil
        import subprocess
        import tempfile

        if not shutil.which("ffmpeg"):
            raise ValueError(
                "MP4 export needs ffmpeg on PATH; use a .gif suffix for the "
                "dependency-free Pillow export (same convention as other "
                "package animations)."
            )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for i, im in enumerate(frames_rgb):
                im.save(folder / f"frame_{i:04d}.png")
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(folder / "frame_%04d.png"),
                "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                str(path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-500:] or "ffmpeg failed")
        return path
    raise ValueError("filename must end with .gif or .mp4")


def save_property_logs_animation(prop, filename, *, fps=1.0, dpi=110,
                                 ages=None, max_frames=None, limits=None,
                                 show_ro=True):
    """Export the fixed-layout property-log frame through time.

    Axis ranges are computed once with :func:`property_log_limits` and held
    constant. Default frames are one per backstrip / stratigraphic time step
    (:func:`property_log_frame_ages`). GIF writing matches
    :func:`pybasinsubs.timestep.save_geohistory_animation` (Pillow, shared
    palette, disposal=2). A ``.mp4`` suffix is accepted when ffmpeg is on PATH.
    """
    from PIL import Image

    limits = property_log_limits(prop) if limits is None else limits
    ages = np.asarray(
        property_log_frame_ages(prop, max_frames) if ages is None else ages,
        float,
    )
    if len(ages) < 2:
        raise ValueError("need at least two ages to animate")
    frames = []
    for age in ages:
        fig = plot_property_logs(prop, age=float(age), limits=limits, show_ro=show_ro)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            with Image.open(buf) as image:
                frames.append(image.convert("RGB"))
        finally:
            plt.close(fig)
    return _write_animation(frames, Path(filename), fps)


def section_property_limits(out, secprop, variable="porosity", stages=None):
    """Shared colour and axis limits for every restoration stage of one property."""
    r = out.restoration
    if stages is None:
        stages = list(range(len(r.ages)))
    else:
        stages = list(stages)
    xlim, ylim = _section_limits(out, stages)
    norm, display_var, scale, log = _property_norm(secprop, variable, stages)
    return dict(xlim=xlim, ylim=ylim, norm=norm, display_var=display_var,
                scale=scale, log=log, variable=variable)


def plot_section_property(out, secprop, age=0.0, variable="porosity", *,
                          limits=None, title=None, show_fault_numbers=True,
                          ax=None):
    """Restored section coloured by one overlay property at one restoration age.

    Colour and axis limits default to :func:`section_property_limits` computed
    over every restoration stage so frames do not jump. Adjacent columns are
    filled as a continuous polygon mesh (no white vertical stripes).
    """
    labels = {
        "porosity": "Porosity (fraction)",
        "permeability_m2": "Permeability (mD)",
        "permeability": "Permeability (mD)",
        "permeability_mD": "Permeability (mD)",
        "effective_stress_mpa": "Effective stress (MPa)",
        "vertical_stress_mpa": "Vertical stress (MPa)",
        "temperature_C": "Temperature (°C)",
        "quartz_cement": "Quartz cement (fraction)",
        "pore_pressure_mpa": "Pore pressure (MPa)",
    }
    if limits is None:
        limits = section_property_limits(out, secprop, variable)
    stage = _section_stage_index(out, age)
    age_used = float(out.restoration.ages[stage])
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(12.2, 5.2))
    else:
        fig = ax.figure
    norm = limits["norm"]
    display_var = limits["display_var"]
    scale = limits["scale"]
    cmap = cmap_for(variable)

    class _Scaled:
        def __init__(self, inner, sc):
            self.inner, self.sc = inner, sc

        def at_stage(self, s, name):
            return self.inner.at_stage(s, name) * self.sc

    src = _Scaled(secprop, scale) if scale != 1.0 else secprop
    _draw_section_property(ax, out, src, stage, display_var, norm, cmap,
                           show_fault_numbers=show_fault_numbers)
    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.set_xlabel("interpreted input distance (km)")
    ax.set_ylabel("depth (m)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label(labels.get(variable, variable), fontsize=9)
    ax.set_title(
        title or f"{labels.get(variable, variable)} · {age_used:g} Ma",
        loc="left", fontsize=11, fontweight="bold",
    )
    _style(ax)
    if created:
        fig.tight_layout()
    return fig


def save_section_property_animation(out, secprop, filename, *,
                                    variable="porosity", fps=1.0, dpi=110,
                                    ages=None, limits=None,
                                    show_fault_numbers=True):
    """Export one-property section frames through restoration ages.

    Default frames are one per restoration / stratigraphic time step
    (``out.restoration.ages``), not an interpolated finer grid.
    GIF writing matches other package animations (Pillow, shared palette,
    disposal=2). A ``.mp4`` suffix is accepted when ffmpeg is on PATH.
    """
    from PIL import Image

    if ages is None:
        ages = np.asarray(out.restoration.ages, float)
        # Play oldest → youngest, matching other package GIFs.
        if len(ages) >= 2 and ages[0] < ages[-1] - 1e-9:
            ages = ages[::-1].copy()
    else:
        ages = np.asarray(ages, float)
    if limits is None:
        limits = section_property_limits(out, secprop, variable)
    if len(ages) < 2:
        raise ValueError("need at least two ages to animate")
    frames = []
    for age in ages:
        fig = plot_section_property(
            out, secprop, age=float(age), variable=variable, limits=limits,
            show_fault_numbers=show_fault_numbers,
        )
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            with Image.open(buf) as image:
                frames.append(image.convert("RGB"))
        finally:
            plt.close(fig)
    return _write_animation(frames, Path(filename), fps)
