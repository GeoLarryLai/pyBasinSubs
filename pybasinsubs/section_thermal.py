# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Column-wise thermal histories on cached restored sections (not a 2-D PDE).

Input-x and fractional unit-thickness tracers follow the existing vertical-shear
restoration. Conductivity-aware steady geotherms feed Sweeney & Burnham Easy%Ro.
This is NOT the McKenzie thermal-subsidence reversal used in flexural unloading.
No reconstruction or fault fitting is performed here.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import thermal
from .growth_display import display_material_polygons, draw_restored_section, growth_frame_steps
from .lithology import default_lithologies
from .material_colors import section_material_colors


def section_thermal_data(out, section, *, heat_flow=0.059, T_s=4., bin_width_m=1000.,
                         nsub=12, substep_Myr=0.1, matrix_conductivity=None):
    """Compute reusable thermal data from an EXISTING growth result and its Section.

    Heat flow is W/m². Matrix conductivities default to the package lithology
    table in ``section.lith`` order; custom lithologies need an explicit array.
    Porosity uses the supplied Section's applied phi0/C and lithology fractions.
    Tracers are equal vertical-thickness midpoint samples, not organic mass.

    Each tracer starts at surface temperature at its unit's base age. Between
    that deposition anchor and the first constrained geometry, and between
    restoration nodes, temperature is linear in time (same kinetic assumption
    as the 1-D helper). No pre-depositional maturation is assumed. Easy%Ro is
    integrated oldest to youngest; a frame NEVER sees future temperatures.
    The extra oldest growth row repeats geometry but has NO supported thermal
    reconstruction: both T and Ro are NaN there, not an invented basement.

    Frame/unit mapping is EXACTLY growth_frame_steps: present -> youngest unit;
    strip row j -> unit j, geometry j+1 (capped at last constrained geometry).
    The row unit remains provenance only; the display uses ALL surviving units.
    ``fractions`` retains per-unit 0–1 fractions; plotting converts to percent.
    """
    r = out.restoration
    for value in (heat_flow, bin_width_m, substep_Myr):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Heat flow, bin width and kinetic substep must be finite positive")
    if not np.isfinite(T_s) or not isinstance(nsub, (int, np.integer)) or nsub < 2:
        raise ValueError("Surface temperature must be finite; nsub must be an integer >= 2")
    if list(section.units) != list(r.units) or not np.array_equal(section.x, r.x):
        raise ValueError("Section must match the cached growth result's units and input x")
    km = (default_lithologies().loc[section.lith, "k_m"].to_numpy(float)
          if matrix_conductivity is None else np.asarray(matrix_conductivity, float))
    if km.shape != section.phi0.shape or np.any(~np.isfinite(km)) or np.any(km <= 0):
        raise ValueError("Supply one positive matrix conductivity per Section lithology")
    ages = np.asarray(r.ages, float)
    if np.any(np.diff(ages) <= 0):
        raise ValueError("Restoration ages must increase from modern to oldest")
    nu, nx = len(r.units), len(r.x)
    f = (np.arange(nsub) + .5) / nsub
    shape = (len(ages), nu, nx, nsub)
    temp = np.full(shape, np.nan)
    ro = np.full(shape, np.nan)
    depths = np.full(shape, np.nan)
    # Geotherms are integrated within sediment only, relative to surviving top.
    for stage in range(len(ages)):
        for ix in range(nx):
            surface = r.top[stage, stage, ix]
            z_nodes, k_nodes, indices = [0.], [], []
            for unit in range(stage, nu):
                top, base = r.top[stage, unit, ix], r.base[stage, unit, ix]
                if not np.isfinite(top + base) or base - top <= 1e-8:
                    continue
                zz = top - surface + f * (base - top)
                phi = (section.F[unit, ix] * section.phi0 * np.exp(-zz[:, None] / section.C)).sum(axis=1)
                matrix = np.exp(np.dot(section.F[unit, ix], np.log(km)))
                k = thermal.bulk_conductivity(phi, matrix)
                z_nodes.extend(zz.tolist());k_nodes.extend(k.tolist());indices.append((unit, zz))
                depths[stage, unit, ix] = top + f * (base - top)
            if not k_nodes:
                continue
            # Surface conductivity is the shallowest sampled lithology at phi0.
            first = indices[0][0]
            phi_surface = np.dot(section.F[first, ix], section.phi0)
            matrix = np.exp(np.dot(section.F[first, ix], np.log(km)))
            kk = np.r_[thermal.bulk_conductivity(phi_surface, matrix), k_nodes]
            tt = thermal.steady_geotherm(z_nodes, kk, heat_flow, T_s)
            start = 1
            for unit, zz in indices:
                temp[stage, unit, ix] = tt[start:start + len(zz)]
                start += len(zz)
    # Existing 1-D kinetics, with a documented deposition anchor, no future use.
    for unit in range(nu):
        birth = float(section.age_base[unit])
        for ix in range(nx):
            for sub in range(nsub):
                ok = np.flatnonzero(np.isfinite(temp[:, unit, ix, sub]) & (ages < birth))[::-1]
                if not len(ok):
                    continue
                t = np.r_[birth, ages[ok]]
                T = np.r_[T_s, temp[ok, unit, ix, sub]]
                ro[ok, unit, ix, sub] = thermal.easy_ro(t, T, substep_Myr=substep_Myr)[1:]
    steps = growth_frame_steps(out)
    frames = []
    for index, step in enumerate(steps):
        stage = max(0, min(step["unit_index"] + 1, nu - 1))
        supported = bool(step["age_Ma"] <= ages[-1])
        frames.append(dict(**step, index=index, stage=stage, selected_unit=max(step["unit_index"], 0),
                           supported=supported, T=temp[stage].copy() if supported else np.full(shape[1:], np.nan),
                           ro=ro[stage].copy() if supported else np.full(shape[1:], np.nan)))
    edges = np.arange(r.x[0], r.x[-1] + bin_width_m, bin_width_m)
    if edges[-1] < r.x[-1]:
        edges = np.r_[edges, r.x[-1]]
    # Horizontal Voronoi widths clipped to each bin: thickness*width weighting.
    cell_edges = np.r_[r.x[0], .5 * (r.x[:-1] + r.x[1:]), r.x[-1]]
    rows = []
    for frame in frames:
        stage = frame["stage"]
        for unit in range(nu):
            th = np.maximum(r.base[stage, unit] - r.top[stage, unit], 0.) if unit >= stage else np.zeros(nx)
            th = np.nan_to_num(th)
            values = frame["ro"][unit]
            for a, b in zip(edges[:-1], edges[1:]):
                widths = np.maximum(0., np.minimum(cell_edges[1:], b) - np.maximum(cell_edges[:-1], a))
                weights = np.broadcast_to((th * widths / nsub)[:, None], values.shape)
                denominator = weights.sum()
                supported_area = np.sum(weights * np.isfinite(values))
                for name, lo, hi, _ in thermal.WINDOWS[1:]:
                    numerator = np.sum(weights * ((values >= lo) & (values < hi)))
                    rows.append(dict(frame=frame["index"], unit=r.units[unit], unit_index=unit,
                                     x_m=(a + b) / 2, window=name,
                                     fraction=float(numerator / denominator) if denominator else 0.,
                                     supported_fraction=float(supported_area / denominator) if denominator else 0.))
    return dict(frames=frames, fractions=pd.DataFrame(rows), x=r.x.copy(), bin_edges_m=edges.copy(), f=f, depths=depths,
                temperature=temp, ro=ro, out=out,
                limits=dict(x=(r.x[0] / 1e3, r.x[-1] / 1e3),
                            depth=(float(np.nanmax(r.base)) + 250, min(-1000., float(np.nanmin(r.top)) - 1000))),
                conditions=dict(heat_flow_W_m2=heat_flow, surface_temperature_C=T_s, bin_width_m=bin_width_m,
                                nsub=nsub, substep_Myr=substep_Myr, matrix_conductivity=km.tolist(),
                                model="independent steady columns + causal Easy%Ro, not 2-D heat conduction"))


def plot_section_thermal_timestep(data, index=0, *, show_fault_numbers=True):
    """Plot cached INDEX (0 = modern): section + oil/wet/dry 0–100% line strips.

    No right-hand column/geotherm and no inversion or kinetic recomputation.
    Fills are clipped to each original material partition and surviving unit.
    Every supported remaining unit contributes an independent percentage line
    series, oldest first; opaque original unit-color strokes and alpha-0.3 fills to zero, never stacked.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgb

    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(data["frames"]):
        raise ValueError("Index must select an existing thermal frame")
    frame = data["frames"][index];out = data["out"];r = out.restoration
    fig = plt.figure(figsize=(12.5, 6.8), dpi=110)
    section = fig.add_axes([.085, .50, .83, .40], label="section")
    strips = [fig.add_axes([.085, bottom, .83, .075], sharex=section, label=window[0])
              for bottom, window in zip([.34, .22, .10], thermal.WINDOWS[1:])]
    draw_restored_section(section, out, frame["stage"], show_fault_numbers=show_fault_numbers)
    # Thermal-only artist styling: the shared growth renderer and its default
    # alpha are untouched. Modify ONLY the polygon artists on this new axis.
    section_alpha = .35
    for collection in section.collections:
        collection.set_alpha(section_alpha)
        collection.set_gid("thermal-unit-fill")
    colors = section_material_colors(r.units)
    # Use the original unit RGB for readable opaque lines; section fills stay translucent.
    line_colors = [to_rgb(c) for c in colors]
    section.set_ylabel("depth (m)", fontsize=9, wrap=False)
    f = data["f"]
    # Match the display's one-sided partition interpolation; never bridge faults.
    bounds = np.r_[r.x[0], [fault.x0 for fault in out.faults], r.x[-1]]
    for p in display_material_polygons(out, frame["stage"]):
        unit = p["unit_index"];block = p["block"]
        mask = (r.x >= bounds[block]) & (r.x <= bounds[block + 1])
        if mask.sum() < 2 or not frame["supported"]:
            continue
        xx = p["x_m"]
        zz = p["top_m"][None, :] + np.r_[0., f, 1.][:, None] * (p["base_m"] - p["top_m"])[None, :]
        X = np.broadcast_to(xx[None, :] / 1e3, zz.shape)
        fields = []
        for key in ("ro", "T"):
            vals = frame[key][unit]
            sampled = np.array([np.interp(xx, r.x[mask], vals[mask, k]) for k in range(len(f))])
            fields.append(np.ma.masked_invalid(np.vstack([sampled[0], sampled, sampled[-1]])))
        R, T = fields
        if R.count():
            section.contourf(X, zz, R, levels=[.6, 1.3, 2., 4.],
                             colors=[w[3] for w in thermal.WINDOWS[1:]], alpha=.20, zorder=2)
        if T.count():
            levels = [v for v in range(50, 401, 50) if T.min() < v < T.max()]
            if levels:
                cs = section.contour(X, zz, T, levels=levels, colors="#b2182b", linewidths=.65, zorder=4)
                section.clabel(cs, fmt="%d°C", fontsize=7, inline=True)
    # Keep unit contacts readable above the translucent maturity shades.
    for p in display_material_polygons(out, frame["stage"]):
        contact_color = np.asarray(to_rgb(colors[p["unit_index"]])) * .60
        for edge in ("top_m", "base_m"):
            section.plot(p["x_m"] / 1e3, p[edge], color=contact_color, alpha=.7,
                         lw=.45, zorder=3, gid="thermal-unit-contact")
    section.set_xlim(*data["limits"]["x"]);section.set_ylim(*data["limits"]["depth"])
    section.tick_params(labelbottom=False)
    section.set_title(frame["label"], fontsize=10, loc="left")
    d = data["fractions"]
    d = d[d.frame == index]
    # Per-unit percentages have DIFFERENT denominators: do not stack/add them.
    # Supported immature units legitimately draw zero; stripped/unsupported ones
    # are omitted altogether, including every unit on the extra 252-Ma frame.
    remaining = sorted(d.loc[d.supported_fraction > 0, "unit_index"].unique(), reverse=True)
    for ax, (name, _, _, _) in zip(strips, thermal.WINDOWS[1:]):
        for layer, unit in enumerate(remaining):
            values = d[(d.unit_index == unit) & (d.window == name)].sort_values("x_m")
            ax.fill_between(values.x_m / 1e3, 0., values.fraction * 100.,
                            color=line_colors[unit], alpha=.3, linewidth=0., zorder=2.5 + layer,
                            gid=f"thermal-shade-unit-{unit}")
            ax.plot(values.x_m / 1e3, values.fraction * 100.,
                    color=line_colors[unit], alpha=1., linewidth=1.3, zorder=3 + layer,
                    label=r.units[unit], gid=f"thermal-line-unit-{unit}")
        ax.set_ylim(-3., 103.);ax.set_yticks([0., 50., 100.])
        ax.set_title(name, fontsize=8, loc="left", pad=3)
        ax.tick_params(labelbottom=ax is strips[-1], labelsize=7)
        ax.grid(alpha=.16, axis="y", zorder=0)
        ax.set_axisbelow(True)
        if not remaining:
            ax.text(.5, .55, "No supported sediment maturity at this age", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color=".4")
    strips[1].set_ylabel("% in petroleum windows", fontsize=8, wrap=False)
    strips[-1].set_xlabel("interpreted input distance (km)", fontsize=9)
    # Compact stratigraphic key in the whitespace, never covering line data.
    if remaining:
        fig.legend(handles=[Line2D([], [], color=line_colors[u], linewidth=1.3,
                                   label=r.units[u], alpha=1.) for u in sorted(remaining)],
                   loc="upper center", bbox_to_anchor=(.5, .485), ncol=5, fontsize=6.5,
                   frameon=False, handlelength=1.5, columnspacing=1.)
    section.legend(handles=[Patch(facecolor=c, label=f"{name} {lo:g}–{hi:g} %Ro", alpha=.20)
                            for name, lo, hi, c in thermal.WINDOWS[1:]],
                   loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, fontsize=8, frameon=False)
    q = data["conditions"]["heat_flow_W_m2"] * 1000;ts = data["conditions"]["surface_temperature_C"]
    fig.suptitle(f"Whole-section thermal history · {q:g} mW/m² · surface {ts:g}°C", fontsize=11, y=.99)
    fig.text(.5, .015, "Independent steady columns + causal Easy%Ro · fixed-trace restoration prototype · no water/basement maturity", ha="center", fontsize=8)
    return fig


def save_section_thermal_animation(data, filename, *, fps=1., show_fault_numbers=True):
    """Loop all cached frames oldest to modern using Pillow (GIF only)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    filename = Path(filename)
    if filename.suffix.lower() != ".gif" or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Use a .gif filename and finite positive fps")
    images = []
    for index in reversed(range(len(data["frames"]))):
        fig = plot_section_thermal_timestep(data, index, show_fault_numbers=show_fault_numbers)
        buf = io.BytesIO();fig.savefig(buf, format="png", dpi=110);plt.close(fig)
        buf.seek(0);images.append(Image.open(buf).convert("RGB"))
    # A shared palette derived from all frames retains late window colours too.
    sheet = Image.new("RGB", (images[0].width, images[0].height * len(images)))
    for i, image in enumerate(images):
        sheet.paste(image, (0, i * image.height))
    # Quantize original pixels: downsampling would blend small black text
    # into pastel fills and turn labels muddy in the GIF.
    palette = sheet.quantize(colors=256)
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    quantized[0].save(filename, save_all=True, append_images=quantized[1:], duration=round(1000 / fps), loop=0, disposal=2)
    return filename
