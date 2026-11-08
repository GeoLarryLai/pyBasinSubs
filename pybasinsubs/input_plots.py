# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Input/model figures shared by the browser and notebooks; no inversion or fitting.

Optional measured porosity records use columns ``depth_m`` (below seafloor),
``porosity`` (fraction), and optionally ``lithology``. Pass them explicitly, or
attach them as ``Column.meta["porosity_data"]``. Published sampled curves are
not silently relabelled as measured well data.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .plotting import _style, ics_color


def stratigraphic_column(column):
    """Present compacted units and paleo-water inputs on identical depth boundaries.

    Eroded units are not inserted into the present-day thickness axis. Ages on
    both sides of a hiatus are printed at its single physical boundary.
    """
    u = column.units
    kept = u[(u.thickness_m > 0) & (u.eroded == 0)].reset_index(drop=True)
    if kept.empty:
        raise ValueError("No preserved positive-thickness units to plot")
    edges = np.r_[0., np.cumsum(kept.thickness_m.to_numpy(float))]
    fig, (ac, aw) = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True,
                                gridspec_kw={"width_ratios": [1.2, 1]})
    ages = []
    # Keep true thicknesses; move only text, with leaders for thin adjacent beds.
    def label_positions(y):
        y = np.asarray(y, float).copy()
        gap = edges[-1] / max(22, len(y) + 1)
        for k in range(len(y) - 2, -1, -1):
            y[k] = min(y[k], y[k + 1] - gap)
        return y
    unit_labels = label_positions((edges[:-1] + edges[1:]) / 2)
    for j, row in kept.iterrows():
        color = ics_color((row.age_top_Ma + row.age_base_Ma) / 2)[0]
        if u.age_base_Ma.max() - u.age_top_Ma.min() <= 40:
            norm = plt.Normalize(u.age_top_Ma.min(), u.age_base_Ma.max())
            color = plt.get_cmap("viridis_r")(norm((row.age_top_Ma + row.age_base_Ma) / 2))
        ac.add_patch(Rectangle((0, edges[j]), .42, row.thickness_m,
                              facecolor=color, edgecolor="white", linewidth=1))
        ac.annotate(f"{row.unit} · {row.thickness_m:g} m",
                    (.42, (edges[j] + edges[j + 1]) / 2), xytext=(.54, unit_labels[j]),
                    ha="left", va="center", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": ".55", "lw": .5})
        age = f"{row.age_top_Ma:g}"
        if j and not np.isclose(kept.iloc[j - 1].age_base_Ma, row.age_top_Ma):
            age = f"{kept.iloc[j - 1].age_base_Ma:g} / {age}"
        ages.append(age)
    ages.append(f"{kept.iloc[-1].age_base_Ma:g}")
    ac.set_yticks(edges, [])
    for z, a, yz in zip(edges, ages, label_positions(edges)):
        ac.annotate(f"{z:g} m · {a} Ma", (0, z), xytext=(-.04, yz),
                    ha="right", va="center", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": ".55", "lw": .5})
    ac.set_xlim(0, 1.55); ac.set_xticks([])
    ac.set_title("Stratigraphic column", loc="left", fontsize=11)
    ac.set_ylabel("")  # Each leader already gives depth (m) and boundary age (Ma).
    water = kept.water_depth_m.to_numpy(float)
    # The basal age has an input only when an explicit matching zero-thickness node exists.
    basal = u[(u.thickness_m == 0) & np.isclose(u.age_top_Ma, kept.iloc[-1].age_base_Ma)]
    wz = edges[:-1]
    if len(basal):
        water = np.r_[water, basal.iloc[0].water_depth_m]; wz = edges
    aw.plot(water, wz, "o-", color="#2a6f8f", ms=4, lw=1.5)
    for z in edges:
        aw.axhline(z, color=".8", lw=.6)
    pad = max(20., np.ptp(water) * .12)
    aw.set_xlim(np.max(water) + pad, min(0., np.min(water) - pad))
    aw.set_xlabel("Paleo-water depth (m) · deep → shallow")
    aw.set_title("Paleo-water inputs at unit tops", loc="left", fontsize=11)
    aw.tick_params(labelleft=False)
    for ax in (ac, aw):
        ax.set_ylim(edges[-1], 0); ax.spines[["top", "right"]].set_visible(False)
    note = "Present compacted thickness; paired boundary ages mark a hiatus. Water depths are inputs, not inferred bathymetry."
    if u.eroded.any():
        note += " Eroded units excluded from present thickness."
    fig.suptitle(column.name, x=.02, ha="left", fontsize=12)
    fig.text(.02, .015, note, fontsize=8, color=".35")
    fig.tight_layout(rect=(0, .05, 1, .95))
    return fig


def compaction_relations(model, measured=None, *, measured_note=None, selected_curves=None, extra_curves=None):
    """Actual lithology-specific central curves used by a Column or Section.

    No refitting, synthetic scatter or invented uncertainty. Measurements, when
    supplied, are context overlays and do not change decompaction parameters.
    Optional selected_curves filters only this plot; extra_curves is a parameter
    table (lithology, phi0, C) of plot-only comparisons. Defaults are unchanged.
    """
    if hasattr(model, "lithologies"):
        names = model.lith_names
        active = np.any(model.fractions() > 0, axis=0)
        pars = model.lithologies.loc[names]
        phi0, decay = pars.phi0.to_numpy(float), pars.C.to_numpy(float)
        depth = model.units.loc[model.units.eroded == 0, "thickness_m"].sum()
        if measured is None:
            measured = model.meta.get("porosity_data")
    else:
        names = model.lith or [f"Lithology {i + 1}" for i in range(len(model.phi0))]
        active = np.any(model.F > 0, axis=(0, 1))
        phi0, decay = model.phi0, model.C
        depth = np.max(model.h.sum(axis=0))
    active_names = {name for name, used in zip(names, active) if used}
    selected = active_names if selected_curves is None else set(selected_curves)
    extra = pd.DataFrame(extra_curves) if extra_curves is not None else pd.DataFrame()
    if not extra.empty:
        if not {"lithology", "phi0", "C"} <= set(extra):
            raise ValueError("Extra curves require lithology,phi0,C")
        if not np.isfinite(extra[["phi0", "C"]].to_numpy(float)).all() or not (extra.phi0.between(0, 1, inclusive="right") & (extra.C > 0)).all():
            raise ValueError("Extra curves require physical finite parameters")
    available = active_names | (set(extra.lithology) if len(extra) else set())
    if not selected <= available:
        raise ValueError("Selected porosity curves are not available")
    points = pd.DataFrame(measured) if measured is not None else pd.DataFrame()
    if not points.empty:
        if not {"depth_m", "porosity"} <= set(points.columns):
            raise ValueError("Porosity data require depth_m and porosity (fraction)")
        points = points[np.isfinite(points.depth_m) & np.isfinite(points.porosity)
                        & (points.depth_m >= 0) & points.porosity.between(0, 1)]
        if len(points):
            depth = max(depth, points.depth_m.max())
    z = np.linspace(0, max(1000., depth * 1.1), 301)
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for i, name in enumerate(names):
        if active[i] and name in selected:
            ax.plot(phi0[i] * np.exp(-z / decay[i]), z, lw=2,
                    label=f"{name}: φ₀={phi0[i]:.3g}, C={decay[i]:.0f} m")
    for row in extra.itertuples():
        if row.lithology in selected and row.lithology not in active_names:
            ax.plot(row.phi0 * np.exp(-z / row.C), z, lw=1.7, ls="--",
                    label=f"{row.lithology} (reference only): φ₀={row.phi0:.3g}, C={row.C:.0f} m")
    # Curve visibility does not suppress real observations or change fitting behavior.
    if not points.empty:
        groups = points.groupby("lithology", sort=False) if "lithology" in points else [("well/column", points)]
        for label, group in groups:
            ax.scatter(group.porosity, group.depth_m, s=22, marker="o", edgecolors="white",
                       linewidths=.4, label=f"Measured: {label}", zorder=3)
    ax.set(xlabel="Porosity (fraction)", ylabel="Depth below seafloor (m)",
           xlim=(0, max(.75, max(float(np.max(phi0)), float(extra.phi0.max()) if len(extra) else 0.) * 1.08)), ylim=(z[-1], 0))
    ax.set_title("Compaction relations used for decompaction", loc="left", fontsize=11)
    _style(ax)
    if ax.lines or ax.collections:
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(.5, .5, "No porosity curves selected for plotting", transform=ax.transAxes, ha="center", fontsize=10)
    note = (measured_note or "Measured points overlaid; model parameters unchanged.") if not points.empty else "No measured porosity points supplied for this dataset; model curves only."
    if selected_curves is not None:
        ax.set_title("Porosity curves selected for plotting", loc="left", fontsize=11)
        note += "\nPlot selection only; references do not enter decompaction."
    fig.text(.02, .015, "φ(z) = φ₀ exp(−z/C). " + note, fontsize=8, color=".35")
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig


def interpreted_section(section, faults=(), show_fault_numbers=True):
    """Unrestored input horizons/strata and original sticks; no science calls."""
    from .material_colors import section_material_colors
    top = section.W0 + np.r_[np.zeros((1, len(section.x))), np.cumsum(section.h, axis=0)]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    for k, color in enumerate(section_material_colors(section.units)):
        ax.fill_between(section.x / 1000, top[k], top[k + 1], color=color,
                        label=section.units[k], lw=0)
        ax.plot(section.x / 1000, top[k], color=".25", lw=.4)
    deepest_trace = max([np.max(top)] + [float(np.max(np.asarray(f.stick)[:, 1])) for f in faults if f.stick is not None])
    for i, fault in enumerate(faults):
        if fault.stick is None:
            continue
        stick = np.asarray(fault.stick)
        ax.plot(stick[:, 0] / 1000, stick[:, 1], color=".15", lw=1.3)
        if show_fault_numbers:
            p = stick[np.argmax(stick[:, 1])]
            ax.annotate(f"#{i + 1}", (p[0] / 1000, p[1]), xytext=(3, -9),
                        textcoords="offset points", fontsize=8)
    ax.set(xlabel="Distance along section (km)", ylabel="Depth below sea level (m)",
           ylim=(deepest_trace + 500, min(0., np.min(top)) - 100))
    ax.set_title("Present-day interpreted section · input geometry", loc="left", fontsize=11)
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False)
    _style(ax); fig.tight_layout()
    return fig
