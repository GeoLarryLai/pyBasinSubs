# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Clean, single-message figures.

Geohistory diagram (:func:`geohistory`): age runs left (old) to right (young);
depth below present sea level increases downward. Each unit is a coloured band
between its top and base horizons (colour = age of deposition), so thinning of
a band to the right *is* compaction. Lines on top: water depth (seafloor),
total subsidence (basement), Airy tectonic subsidence, and the undecompacted
(present-thickness) curve. Shading: Monte Carlo 68 % and 95 % envelopes.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ICS chart colours (v2023) at epoch/series level; (top, base, name, colour)
ICS = [(0, 2.58, "Pleistocene", "#FFF2AE"), (2.58, 5.333, "Pliocene", "#FFFF99"),
       (5.333, 23.03, "Miocene", "#FFFF00"), (23.03, 33.9, "Oligocene", "#FDC07A"),
       (33.9, 56.0, "Eocene", "#FDB46C"), (56.0, 66.0, "Paleocene", "#FDA75F"),
       (66.0, 100.5, "Late Cretaceous", "#A6D84A"), (100.5, 143.1, "Early Cretaceous", "#8CCD57"),
       (143.1, 161.5, "Late Jurassic", "#B3E3EE"), (161.5, 174.7, "Middle Jurassic", "#80CFD8"),
       (174.7, 201.4, "Early Jurassic", "#42AED0"), (201.4, 237.0, "Late Triassic", "#BD8CC3"),
       (237.0, 247.2, "Middle Triassic", "#B168B1"), (247.2, 251.9, "Early Triassic", "#983999"),
       (251.9, 259.5, "Lopingian", "#FBA794"), (259.5, 273.0, "Guadalupian", "#FB745C"),
       (273.0, 298.9, "Cisuralian", "#EF5845"), (298.9, 358.9, "Carboniferous", "#67A599")]

C_TOTAL, C_TECT, C_UND, C_W = "#1f3b73", "#c0392b", "#7f7f7f", "#2a9df4"


def ics_color(age):


    """ICS (2023) chart colour for a numerical age (Ma); returns (hex colour, period/epoch name)."""
    for top, base, name, c in ICS:
        if top <= age < base:
            return c, name
    return "#cccccc", "older"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


C_WATERFILL = "#6fb7e8"
C_LAND = "#d9c49a"


def geohistory(result, ax=None, color_by="auto", show=("total", "tectonic", "undecompacted", "water"),
               envelopes=True, title=None, present_elevation=None, legend=True, cmap="viridis_r"):
    """Geohistory diagram for a :class:`~pybasinsubs.backstrip.BackstripResult`."""
    b = result.best; u = result.column.units
    ages = b["age"][0]
    order = np.argsort(-ages)
    ages_o = ages[order]
    n = len(u)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.2))
    sea = b["W"][0] - b["SL"][0]
    span = (u.age_base_Ma.max() - u.age_top_Ma.min())
    if color_by == "auto":
        color_by = "ics" if span > 40 else "age"
    norm = mpl.colors.Normalize(u.age_top_Ma.min(), u.age_base_Ma.max())
    cm = plt.get_cmap(cmap)
    handles = {}
    # paleo-water column: sea surface (-SL, eustatic, in present-day depth) to paleo-seafloor
    surf = -b["SL"][0][order]; sf = sea[order]
    ax.fill_between(ages_o, surf, sf, where=sf >= surf, interpolate=True, color=C_WATERFILL, alpha=0.25, lw=0, zorder=0)
    ax.fill_between(ages_o, surf, sf, where=sf < surf, interpolate=True, color=C_LAND, alpha=0.35, lw=0, zorder=0)
    subaerial = bool(np.any(sf < surf))
    for j in range(n):
        if u.thickness_m[j] <= 0:
            continue
        mid = 0.5 * (u.age_top_Ma[j] + u.age_base_Ma[j])
        if color_by == "ics":
            c, lab = ics_color(mid)
        else:
            c, lab = cm(norm(mid)), None
        top = b["tops"][0, :, j][order] + sea[order]
        bot = top + b["thick"][0, :, j][order]
        act = ~np.isnan(top)
        xa, ta, ba = ages_o[act], top[act], bot[act]
        # deposition wedge: zero thickness at the seafloor when deposition starts
        ab = u.age_base_Ma[j]
        if ab > xa[0]:
            s_at = np.interp(-ab, -ages_o, sea[order])
            xa = np.r_[ab, xa]; ta = np.r_[s_at, ta]; ba = np.r_[s_at, ba]
        ax.fill_between(xa, ta, ba, color=c, lw=0, alpha=0.95 if not u.eroded[j] else 0.45,
                        hatch="//" if u.eroded[j] else None, edgecolor="white")
        ax.plot(xa, ba, color="k", lw=0.35, alpha=0.6)
        if lab and lab not in handles:
            handles[lab] = Patch(color=c, label=lab)
    sty = dict(total=(b["sigma"][0], C_TOTAL, "Total subsidence (decompacted)", "sigma"),
               tectonic=(b["zeta"][0], C_TECT, "Tectonic subsidence (Airy)", "zeta"),
               undecompacted=(b["undecompacted"][0], C_UND, "Undecompacted total", "undecompacted"),
               water=(sea, C_W, "Seafloor (paleo-water depth - sea level)", None))
    lines = []
    for key in show:
        y, c, lab, q = sty[key]
        if envelopes and result.mc is not None and q is not None:
            x = result.mc[q][:, order]
            for lo, hi, al in ((2.5, 97.5, 0.10), (16, 84, 0.20)):
                ax.fill_between(ages_o, np.percentile(x, lo, 0), np.percentile(x, hi, 0), color=c, alpha=al, lw=0)
        ls = "--" if key == "undecompacted" else "-"
        ax.plot(ages_o, y[order], color=c, lw=2.2 if key in ("total", "tectonic") else 1.4, ls=ls, marker="o" if key != "water" else None, ms=3)
        lines.append(Line2D([], [], color=c, lw=2, ls=ls, label=lab))
    lines.append(Patch(color=C_WATERFILL, alpha=0.25, label="Paleo-water column"))
    if subaerial:
        lines.append(Patch(color=C_LAND, alpha=0.35, label="Subaerial (above sea level)"))
    if present_elevation is not None:
        ax.plot([ages_o[-1], 0], [b["sigma"][0][order][-1], -present_elevation], color=C_TOTAL, ls=":", lw=1.5)
        lines.append(Line2D([], [], color=C_TOTAL, ls=":", label=f"Uplift to present ({present_elevation:.0f} m)"))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(max(u.age_base_Ma.max(), ages.max()), 0)
    ax.invert_yaxis()
    ax.set_xlabel("Age (Ma)"); ax.set_ylabel("Depth below sea level (m)")
    _style(ax)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    if legend:
        from matplotlib.legend import Legend
        bottom = Legend(ax, lines, [h.get_label() for h in lines], loc="upper left", bbox_to_anchor=(0, -0.13),
                        fontsize=8, frameon=False, ncol=2)
        bottom.set_clip_on(False); bottom.set_in_layout(True)
        ax.add_artist(bottom)
        if handles:
            hs = [handles[k] for k in sorted(handles, key=lambda k: [i for i, r in enumerate(ICS) if r[2] == k][0])]
            ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, title="Unit age (ICS)",
                      title_fontsize=8, frameon=False)
        if not handles and color_by == "age":
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
            cb = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.035)
            cb.set_label("Unit deposition age (Ma)")
    return ax


def porosity_fit_plot(z, phi, fit, ax=None, title=None, legacy=None):


    """Porosity-depth scatter of the sampled published curves with the fitted exponential and its 95 % band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 5.2))
    ax.plot(phi, z / 1000, ".", color="#9a9a9a", ms=2.5, label="Published curves (sampled)")
    ax.fill_betweenx(fit["zgrid"] / 1000, fit["lo"], fit["hi"], color="#e74c3c", alpha=0.25, label="95% CI of mean (correct)")
    ax.plot(fit["mean"], fit["zgrid"] / 1000, color="#c0392b", lw=2.2,
            label=f"phi0={fit['phi0']:.3f}, C={fit['C']:.0f} m")
    if legacy is not None:
        o = np.argsort(legacy["z"])
        ax.plot(legacy["ci_lower"][o], legacy["z"][o] / 1000, color="#2c3e50", lw=1, ls="--", label="R script CI (too wide)")
        ax.plot(legacy["ci_upper"][o], legacy["z"][o] / 1000, color="#2c3e50", lw=1, ls="--")
    ax.set_xlim(0, 0.8); ax.set_ylim(6, 0)
    ax.set_xlabel("Porosity (fraction)"); ax.set_ylabel("Depth (km)")
    _style(ax); ax.legend(fontsize=7, loc="lower right")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return ax


def maturity_plot(ages, temps, ro, labels, ax=None, title=None):
    """Two panels: burial temperature and Easy%Ro vs age with oil/gas windows."""
    from .thermal import WINDOWS
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    cm = plt.get_cmap("plasma")
    for i, lab in enumerate(labels):
        c = cm(i / max(1, len(labels) - 1))
        a1.plot(ages, temps[:, i], color=c, lw=1.6, label=lab)
        a2.plot(ages, ro[:, i], color=c, lw=1.6)
    for name, lo, hi, col in WINDOWS[1:]:
        a2.axhspan(lo, hi, color=col, alpha=0.25, lw=0)
        a2.text(0.99, np.sqrt(lo * min(hi, 4)), f"{name} window\n{lo:g}–{hi:g} %Ro", fontsize=7.5, va="center", ha="right",
                transform=a2.get_yaxis_transform(), color="#333", fontweight="bold")
    a2.set_yscale("log"); a2.set_ylim(0.2, 4.5)
    a2.set_yticks([0.25, 0.5, 0.6, 1.0, 1.3, 2.0, 3.0, 4.0])
    a2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g")); a2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a1.set_ylabel("Temperature at unit top (°C)"); a2.set_ylabel("Vitrinite reflectance, Easy%Ro (%)")
    a1.set_title("Burial temperature", loc="left", fontsize=10); a2.set_title("Thermal maturity (log scale)", loc="left", fontsize=10)
    for a in (a1, a2):
        a.set_xlim(np.nanmax(ages), 0); a.set_xlabel("Age (Ma)"); _style(a)
    a1.legend(fontsize=7, ncol=2, title="Unit (top)", title_fontsize=7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def geohistory_thermal(result, tf, mf=None, ax=None, title=None, iso_step=25.0, ro_levels=(0.6, 1.3, 2.0)):
    """Burial diagram (units coloured by age, paleo-water fill) with isotherms and Easy%Ro window lines.

    tf: :func:`pybasinsubs.thermal.temperature_field`; mf: :func:`pybasinsubs.thermal.maturity_field`.
    """
    from matplotlib.tri import Triangulation
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.6))
    geohistory(result, ax=ax, show=("water",), envelopes=False, title=title, legend=True)
    T = tf["T"]; lv = np.arange(iso_step, np.nanmax(T) + iso_step, iso_step)
    A, Z = np.meshgrid(tf["age"], tf["z"], indexing="ij")
    b = result.best; ag = b["age"][0]; o = np.argsort(ag)
    sig = np.interp(tf["age"], ag[o], (b["W"][0] - b["SL"][0] + b["eps"][0])[o])
    Tm = np.where(Z <= sig[:, None] + 1, T, np.nan)            # only inside the sediment column
    cs = ax.contour(A, Z, Tm, levels=lv, colors="#b2182b", linewidths=0.8, linestyles="-", alpha=0.85)
    ax.clabel(cs, fmt="%d °C", fontsize=7, inline=True)
    extra = [Line2D([], [], color="#b2182b", lw=0.8, label=f"Isotherms (every {iso_step:g} °C)")]
    if mf is not None and len(mf["ro"]) > 3:
        sc = 50.0                                           # 1 Myr ~ 50 m so Delaunay is not anisotropic
        tri0 = Triangulation(mf["age"], mf["z"] / sc)
        a_, z_ = mf["age"][tri0.triangles], mf["z"][tri0.triangles]
        tri = Triangulation(mf["age"], mf["z"], triangles=tri0.triangles)
        tri.set_mask((np.ptp(a_, 1) > 8) | (np.ptp(z_, 1) > 800))   # no triangles across gaps/unconformities
        sty = {0.6: ("#1a9850", "oil window top (0.6 %Ro)"), 1.3: ("#e08214", "wet gas (1.3 %Ro)"), 2.0: ("#c51b7d", "dry gas (2.0 %Ro)")}
        for L in ro_levels:
            if np.nanmax(mf["ro"]) < L:
                continue
            c, lab = sty.get(L, ("k", f"{L} %Ro"))
            ax.tricontour(tri, mf["ro"], levels=[L], colors=c, linewidths=2.0, linestyles="--")
            extra.append(Line2D([], [], color=c, lw=2, ls="--", label=lab))
    ax.legend(handles=extra, loc="lower left", fontsize=7, frameon=True, framealpha=0.85)
    return ax


def section_panel(geoms, ages, xw=None, wells=None, flex_lines=None, figsize=(13, 9), title=None):
    """Multi-panel reconstructed cross-sections (one per age) from :func:`pybasinsubs.flexure.section_geometry`."""
    n = len(geoms); nc = 2 if n > 3 else n; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    handles = {}
    for k, (g, a) in enumerate(zip(geoms, ages)):
        ax = axs.flat[k]; xs = g["x"] / 1e3
        ax.fill_between(xs, g["sea_surface"], g["seafloor"], where=g["seafloor"] >= g["sea_surface"], color=C_WATERFILL, alpha=0.25, lw=0, interpolate=True)
        for i, nm in enumerate(g["units"]):
            if np.all(g["base"][i] - g["top"][i] <= 0.5):
                continue
            c, lab = ics_color(g["unit_age"][i])
            ax.fill_between(xs, g["top"][i], g["base"][i], color=c, lw=0, alpha=0.95)
            ax.plot(xs, g["base"][i], color="k", lw=0.3, alpha=0.6)
            handles.setdefault(nm, (g["unit_age"][i], Patch(color=c, label=f"{nm.title()} ({lab})")))
        ax.plot(xs, g["seafloor"], color=C_W, lw=1.2)
        ax.plot(xs, g["base"][-1] if len(g["units"]) else g["seafloor"], color="k", lw=1.4)
        if flex_lines is not None and flex_lines[k] is not None:
            fx, fz = flex_lines[k]
            ax.plot(fx / 1e3, fz, color=C_TECT, lw=1.3, ls="--")
        if xw is not None:
            for xi, w in zip(np.asarray(xw) / 1e3, wells if wells is not None else [""] * len(xw)):
                ax.axvline(xi, color="grey", lw=0.5, ls=":")
                if k < nc:
                    ax.text(xi, 0, w, rotation=90, fontsize=6, va="bottom", ha="center", color="grey")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{a:g} Ma" if a > 0 else "Present day", loc="left", fontweight="bold", fontsize=10)
        _style(ax)
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs.flat[0].invert_yaxis()
    for ax in axs[-1]:
        ax.set_xlabel("Distance east along transect (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Depth below sea level (m)")
    hs = [h for _, h in sorted(handles.values(), key=lambda t: t[0])]
    hs += [Patch(color=C_WATERFILL, alpha=0.25, label="Paleo-water column"),
           Line2D([], [], color="k", lw=1.4, label="Deepest drilled horizon (TD)")]
    if flex_lines is not None:
        hs.append(Line2D([], [], color=C_TECT, lw=1.3, ls="--", label="Flexural tectonic subsidence (best Te)"))
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig
