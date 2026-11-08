# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Display/export of the existing spatial reconstruction; NO inversion or refitting.

The old renderer dropped the original mesh cell crossing each material partition:
separate fill_between calls ended at neighbouring sample CENTRES, 250 m apart.
These were sampling slots, not inferred missing geology. Here existing input
cells are split at the exact partition and transformed one-sidedly. The original
continuous mesh area is recovered, never a restored gap interpolated/filled.
Curved traces are mapped through the SAME stratigraphic unloading and vertical
shear, clipped above the surviving unit and below the model's known base.
This remains a vertical-partition/shear prototype, not curved-fault balancing.
"""
import numpy as np
import pandas as pd

from .material_colors import section_material_colors


def _domains(out):
    r = out.restoration;x = r.x
    bounds = np.r_[x[0], [f.x0 for f in out.faults], x[-1]]
    masks = [x < out.faults[0].x0] + r["material_blocks"]
    return bounds, masks


def _shift(out, stage):
    return out.restoration["interval_throw_field_m"][:stage].sum(axis=0)


def display_material_polygons(out, stage):
    """Use complete EXISTING source cells, split at exact partition boundaries.

    One-sided throw interpolation/extrapolation extends at most one sample step
    to the partition. It changes neither inferred samples nor source horizons.
    Returned polygons retain true vertical offset/relief at boundaries; no
    restored gap, crossing, missing formation or curved cutoff is fabricated.
    """
    r = out.restoration;x = r.x;shift = _shift(out, stage)
    bounds, masks = _domains(out);rows = []
    for block, (lo, hi, mask) in enumerate(zip(bounds[:-1], bounds[1:], masks)):
        if not mask.any():continue
        xx = np.unique(np.r_[lo, x[(x > lo) & (x < hi)], hi])
        u = np.interp(xx, x[mask], shift[mask])
        for k in range(stage, len(r.units)):
            # Undo the existing shear to recover the same unloaded source mesh,
            # interpolate only that KNOWN mesh, then apply each side's field.
            rawtop = r.top[stage, k] + shift;rawbase = r.base[stage, k] + shift
            rows.append(dict(block=block, unit_index=k, x_m=xx.copy(),
                             top_m=np.interp(xx, x, rawtop) - u,
                             base_m=np.interp(xx, x, rawbase) - u))
    return rows


def display_area_audit(out):
    """Account explicitly for formerly omitted source-cell areas, not new geology."""
    r = out.restoration;_, masks = _domains(out);rows = []
    for i in range(len(r.ages)):
        polygons = display_material_polygons(out, i)
        for k in range(i, len(r.units)):
            source = float(np.trapezoid(r.base[i, k] - r.top[i, k], r.x))
            previous = sum(float(np.trapezoid((r.base[i, k] - r.top[i, k])[m], r.x[m])) for m in masks)
            drawn = sum(float(np.trapezoid(p["base_m"] - p["top_m"], p["x_m"])) for p in polygons if p["unit_index"] == k)
            rows.append(dict(stage=i, age_Ma=float(r.ages[i]), unit=r.units[k], source_mesh_area_m2=source,
                             previous_sample_center_area_m2=previous, recovered_existing_cell_area_m2=drawn - previous,
                             displayed_area_m2=drawn, display_minus_source_area_m2=drawn - source))
    return pd.DataFrame(rows)


def _trace_points(f, r):
    curve = f.curve if f.curve is not None else f.stick
    if curve is None:
        z0 = f.z_tip if f.z_tip is not None else float(np.min(r.top[0, f.tip_unit or 0]))
        z = np.linspace(z0, float(np.max(r.base[0, -1])), 250)
        return np.column_stack([f.x_at(z), z])
    curve = np.asarray(curve, float);points = []
    bounds = r.x
    for a, b in zip(curve[:-1], curve[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 40.)) + 1)
        t = np.linspace(0., 1., n)
        if b[0] != a[0]:
            tx = (bounds - a[0]) / (b[0] - a[0]);t = np.unique(np.r_[t, tx[(tx > 0) & (tx < 1)]])
        points.extend(a + t[:-1, None] * (b - a))
    return np.vstack([points, curve[-1]])


def restored_fault_traces(out, stage):
    """Slipped/decompacted trace segments within surviving known stratigraphy.

    Original interpreted nodes remain untouched. Intersections with the stripped
    top are inserted for display only. Segment breaks at material-displacement
    jumps are retained rather than pretending a solved continuous curved contact.
    Returns original depths and restored top for testable clipping provenance.
    """
    r = out.restoration;x = r.x;shift = _shift(out, stage);bounds, masks = _domains(out)
    original_edges = np.r_[r.top[0], r.base[0, -1:]]
    rawtop = r.top[stage] + shift;rawbase = r.base[stage] + shift
    rows = []
    def map_point(p, block):
        xx, zz = p
        edges = np.array([np.interp(xx, x, h) for h in original_edges])
        unit = int(np.clip(np.searchsorted(edges, zz, side="right") - 1, stage, len(r.units) - 1))
        fraction = (zz - edges[unit]) / max(edges[unit + 1] - edges[unit], 1e-10)
        u = float(np.interp(xx, x[masks[block]], shift[masks[block]]))
        top = float(np.interp(xx, x, rawtop[unit]));base = float(np.interp(xx, x, rawbase[unit]))
        return top + fraction * (base - top) - u, float(np.interp(xx, x, rawtop[stage])) - u
    for f in out.faults:
        nodes = _trace_points(f, r)
        # Split also at fixed partition crossings so the exact one-sided shear
        # can be applied on each side, without a fictitious connecting segment.
        expanded = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            t = np.array([0.])
            if b[0] != a[0]:
                tt = (bounds[1:-1] - a[0]) / (b[0] - a[0]);t = np.unique(np.r_[t, tt[(tt > 0) & (tt < 1)]])
            expanded.extend(a + t[:, None] * (b - a))
        nodes = np.vstack([expanded, nodes[-1]])
        segment = -1;last = None;last_block = None
        for a, b in zip(nodes[:-1], nodes[1:]):
            lo, hi = 0., 1.
            # Clip against surviving top, known base and horizontal model extent.
            inequalities = [(a[1] - np.interp(a[0], x, original_edges[stage]), b[1] - np.interp(b[0], x, original_edges[stage])),
                            (np.interp(a[0], x, original_edges[-1]) - a[1], np.interp(b[0], x, original_edges[-1]) - b[1]),
                            (a[0] - x[0], b[0] - x[0]), (x[-1] - a[0], x[-1] - b[0])]
            for va, vb in inequalities:
                if va < 0 and vb < 0:hi = -1.;break
                if va < 0:lo = max(lo, -va / (vb - va))
                if vb < 0:hi = min(hi, va / (va - vb))
            if hi < lo:last = None;continue
            p, q = a + lo * (b - a), a + hi * (b - a)
            if np.linalg.norm(q - p) < 1e-8:continue
            block = int(np.clip(np.searchsorted(bounds, (p[0] + q[0]) / 2, side="right") - 1, 0, len(masks) - 1))
            z0, top0 = map_point(p, block);z1, top1 = map_point(q, block)
            if last is None or last_block != block or np.linalg.norm(last - np.r_[p[0], z0]) > 1e-5:segment += 1
            for orig, z, current in [(p, z0, top0), (q, z1, top1)]:
                rows.append(dict(fault=f.label, segment=segment, block=block, x_m=float(orig[0]), original_z_m=float(orig[1]),
                                 restored_z_m=float(z), restored_top_m=float(current), stage=stage, age_Ma=float(r.ages[stage])))
            last = np.r_[q[0], z1];last_block = block
    return pd.DataFrame(rows, columns=["fault", "segment", "block", "x_m", "original_z_m", "restored_z_m", "restored_top_m", "stage", "age_Ma"])


def draw_restored_section(ax, out, stage, *, show_fault_numbers=True):
    r = out.restoration;colors = section_material_colors(r.units)
    for p in display_material_polygons(out, stage):
        ax.fill_between(p["x_m"] / 1e3, p["top_m"], p["base_m"], color=colors[p["unit_index"]], lw=0, alpha=.9)
    # No black vertical partition strokes masquerading as interpreted faults.
    traces = restored_fault_traces(out, stage)
    for _, t in traces.groupby(["fault", "segment"], sort=False):
        ax.plot(t.x_m / 1e3, t.restored_z_m, color="0.14", lw=1., zorder=5)
    if show_fault_numbers:
        for number, fault in enumerate(out.faults, 1):
            t = traces[traces.fault == fault.label]
            if len(t):
                tip = t.loc[t.restored_z_m.idxmax()]
                ax.annotate(f"#{number}", (tip.x_m / 1e3, tip.restored_z_m), xytext=(3, 3),
                            textcoords="offset points", fontsize=8, color="0.12", clip_on=True,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.7, pad=.3), zorder=7)
    if out.metadata["assumed_datum_m"] is not None:ax.axhline(out.metadata["assumed_datum_m"], color="navy", ls=":", lw=.75)
    ax.set_ylim(float(np.nanmax(r.base[stage])) + 250, min(-1000., float(np.nanmin(r.top[stage])) - 1000))
    ax.set_xlim(r.x[0] / 1e3, r.x[-1] / 1e3);ax.set_ylabel("depth\n(m)", fontsize=9)
    return traces


def _bins(out, sensitivity, bin_width_m):
    from .growth_spatial import binned_spatial_rates
    return binned_spatial_rates(out, sensitivity=sensitivity, bin_width_m=bin_width_m)


def _draw_rate(ax, bins, unit, kind, color):
    d = bins[(bins.unit == unit) & (bins.kind == kind)]
    if not len(d):
        ax.set_ylim(-.1, 1.);ax.set_yticks([0, .5, 1.])
        ax.text(.02, .65, "not inferred: no deeper horizon", transform=ax.transAxes, fontsize=8, color=color)
        return None
    y = d.rate_m_Myr.to_numpy();lo = d.range_low_m_Myr.to_numpy();hi = d.range_high_m_Myr.to_numpy()
    error = np.maximum(np.vstack([y - lo, hi - y]), 0.)
    is_throw = kind == "fault_throw"
    line = ax.errorbar(d.x_m / 1e3, y, yerr=error, fmt="o-" if is_throw else "s--", ms=4.5,
                      lw=.7, elinewidth=1.3, capsize=3.5, capthick=1.2, zorder=5 if is_throw else 4,
                      color=color, ecolor=color, markeredgecolor=color, markeredgewidth=1.,
                      markerfacecolor="white" if is_throw else color,
                      label="throw · left" if is_throw else "sed. · right")
    if np.max(hi) <= 1e-12:
        ax.set_ylim(-.1, .4);ax.set_yticks([0]);ax.axhline(0, color=color, lw=1., zorder=3)
        ax.text(.02, .6, "throw = 0 (fixed tips inactive)", transform=ax.transAxes, fontsize=8, color=color)
    else:
        low, high = float(np.min(lo)), float(np.max(hi));span = max(high - low, .08 * high, 1e-9)
        ax.set_ylim(max(0., low - .12 * span), high + .16 * span)
    return line


def _row_title(out, j):
    r = out.restoration;h = out.history[out.history.unit == r.units[j]].iloc[0]
    if j == len(r.units) - 1:
        return f"{r.units[j]}: {h.age_top_Ma:g}–{h.age_base_Ma:g} Ma · last constrained section at {r.ages[-1]:g} Ma, NOT a {h.age_base_Ma:g}-Ma reconstruction"
    return f"Strip {r.units[j]}: {h.age_top_Ma:g}–{h.age_base_Ma:g} Ma → top at {h.age_base_Ma:g} Ma"


def plot_spatial_growth_steps(out, *, bin_width_m=1000., sensitivity=(), show_fault_numbers=True):
    import matplotlib.pyplot as plt
    r = out.restoration;nu = len(r.units);bins = _bins(out, sensitivity, bin_width_m);colors = section_material_colors(r.units)
    fig = plt.figure(figsize=(13.5, 4. * nu))
    grid = fig.add_gridspec(2 * nu, 1, height_ratios=[2.5, 1.] * nu, hspace=.7, left=.085, right=.91, bottom=.018, top=.974)
    for j, unit in enumerate(r.units):
        section = fig.add_subplot(grid[2 * j]);rate = fig.add_subplot(grid[2 * j + 1], sharex=section);sediment = rate.twinx()
        draw_restored_section(section, out, min(j + 1, nu - 1), show_fault_numbers=show_fault_numbers)
        section.set_title(_row_title(out, j), fontsize=9, loc="left");section.tick_params(labelbottom=False)
        handles = [h for h in [_draw_rate(rate, bins, unit, "fault_throw", colors[j]), _draw_rate(sediment, bins, unit, "sedimentation", colors[j])] if h is not None]
        rate.set_ylabel("throw rate\n(m/Myr)", color=colors[j], fontsize=9)
        sediment.set_ylabel("sed. rate\n(m/Myr)", color=colors[j], fontsize=9)
        rate.tick_params(axis="y", colors=colors[j], labelsize=8);sediment.tick_params(axis="y", colors=colors[j], labelsize=8)
        rate.set_xlabel("interpreted input distance (km)", fontsize=9);rate.grid(alpha=.18)
        rate.legend(handles=handles, labels=[h.get_label() for h in handles], loc="upper center", bbox_to_anchor=(.5, 1.35), ncol=2, fontsize=7, frameon=False)
    fig.suptitle("SPATIAL GROWTH PROTOTYPE — slipped, top-clipped remaining traces; fixed input interpretation\n"
                 "Existing source cells retained at partitions (no invented gap fill). Vertical-shear model, NOT curved-fault balancing. Zero = ASSUMED.\n"
                 f"{bin_width_m / 1e3:g}-km input-x bins; bars = actual spatial / fit-sensitivity range, NOT confidence intervals. Axes scale independently. m/Myr = mm/kyr.\n"
                 "Open circles: throw; filled squares: sediment. Oldest row repeats last constrained geometry; no basement slip inferred.", fontsize=9, y=.996)
    return fig


def plot_growth_subjects(out, *, sensitivity=(), bin_width_m=1000., subjects=None, show_fault_numbers=True):
    """Notebook-callable individual SUBJECT figures, using existing computed results.

    No reconstruction/refitting happens here. ``subjects`` selects one or more
    names, useful for embedding each figure as its own saved notebook output.
    """
    import matplotlib.pyplot as plt
    names = ["every_unit_sections_rates", "throw_rates", "sedimentation_rates", "restored_sections", "flattening_history"]
    selected = names if subjects is None else list(subjects)
    if set(selected) - set(names):raise ValueError("Unknown growth display subject")
    figures = {};r = out.restoration;colors = section_material_colors(r.units)
    bins = _bins(out, sensitivity, bin_width_m) if any(n in selected for n in ["throw_rates", "sedimentation_rates"]) else None
    for name in selected:
        if name == "every_unit_sections_rates":
            figures[name] = plot_spatial_growth_steps(out, bin_width_m=bin_width_m, sensitivity=sensitivity, show_fault_numbers=show_fault_numbers)
        elif name in ["throw_rates", "sedimentation_rates"]:
            kind = "fault_throw" if name == "throw_rates" else "sedimentation"
            fig, axes = plt.subplots(len(r.units), 1, figsize=(12, 20), sharex=True)
            for j, ax in enumerate(axes):
                _draw_rate(ax, bins, r.units[j], kind, colors[j])
                ax.set_ylabel(("throw rate" if kind == "fault_throw" else "sed. rate") + "\n(m/Myr)", color=colors[j], fontsize=9)
                ax.tick_params(axis="y", colors=colors[j], labelsize=8)
                h = out.history[out.history.unit == r.units[j]].iloc[0]
                ax.set_title(f"{r.units[j]}: {h.age_top_Ma:g}–{h.age_base_Ma:g} Ma", fontsize=9, loc="left");ax.grid(alpha=.2)
            axes[-1].set_xlabel("interpreted input distance (km)");axes[-1].set_xlim(r.x[0] / 1e3, r.x[-1] / 1e3)
            fig.suptitle(("Nonnegative apparent throw rate" if kind == "fault_throw" else "Decompacted sedimentation rate") +
                         " — fixed-trace spatial PROTOTYPE\nActual spatial / fit-sensitivity range bars, NOT confidence intervals; each timestep scales independently.", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .973));figures[name] = fig
        elif name == "restored_sections":
            fig, axes = plt.subplots(len(r.units), 1, figsize=(13, 25), sharex=True)
            for j, ax in enumerate(axes):
                draw_restored_section(ax, out, min(j + 1, len(r.units) - 1), show_fault_numbers=show_fault_numbers);ax.set_title(_row_title(out, j), fontsize=9, loc="left")
            axes[-1].set_xlabel("interpreted input distance (km)")
            fig.suptitle("Restored sections + slipped/top-clipped remaining traces — PROTOTYPE\n"
                         "Known input cells retained; zero is ASSUMED; vertical shear, not curved-fault balancing.\n"
                         "Oldest row: last constrained 170.3-Ma geometry, NOT a new 252-Ma reconstruction.", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .97));figures[name] = fig
        else:
            figures[name] = plot_fault_throw_history(out.spatial, [f.label for f in out.faults], r.units)
    return figures


def fault_throw_ranges(spatial, fault_labels):
    """Mean/min/max of EXISTING throw rates inside each fixed HW partition.

    This is spatial spread along the partition, not a confidence interval or
    resolved down-dip slip distribution. The oldest undefined interval is omitted.
    """
    rows = []
    for number, fault in enumerate(fault_labels, 1):
        data = spatial[(spatial.block == fault) & spatial.throw_m_Myr.notna()]
        for unit, d in data.groupby("unit", sort=False):
            rows.append(dict(fault=fault, fault_number=number, label=f"Fault #{number}", unit=unit,
                             age_Ma=float(d.age_base_Ma.iloc[0]), mean_rate_m_Myr=float(d.throw_m_Myr.mean()),
                             min_rate_m_Myr=float(d.throw_m_Myr.min()), max_rate_m_Myr=float(d.throw_m_Myr.max()),
                             n=int(len(d))))
    return pd.DataFrame(rows)


def plot_fault_throw_history(spatial, fault_labels, units):
    """One compact square: per-fault mean throw and spatial min/max through time."""
    import matplotlib.pyplot as plt
    colors = section_material_colors(units);markers = ["o", "s", "^", "D", "v"]
    table = fault_throw_ranges(spatial, fault_labels)
    fig = plt.figure(figsize=(6, 6), dpi=180);ax = fig.add_axes([.15, .20, .79, .75]);ax.set_box_aspect(1)
    for k, fault in enumerate(fault_labels):
        d = table[table.fault == fault].sort_values("age_Ma");color = colors[(2 * k) % len(colors)]
        ax.fill_between(d.age_Ma, d.min_rate_m_Myr, d.max_rate_m_Myr, color=color, alpha=.16, linewidth=0, zorder=1)
        ax.scatter(d.age_Ma, d.mean_rate_m_Myr, color=color, marker=markers[k % len(markers)],
                   s=37, edgecolor="white", linewidth=.4, zorder=3, label=f"Fault #{k + 1}")
    ages = sorted(table.age_Ma.unique())
    ax.set_xticks(ages, [f"{age:g}" for age in ages], rotation=90, fontsize=7)
    maximum = float(table.max_rate_m_Myr.max())
    ax.set_xlim(0, max(ages) + 6);ax.set_ylim(-.03 * maximum, 1.08 * maximum)
    ax.set_xlabel("Restored top age (Ma)", fontsize=10);ax.set_ylabel("Throw rate (m/Myr)", fontsize=10)
    ax.set_title("Fault throw through time", fontsize=12, pad=9)
    ax.axhline(0, color="0.5", lw=.6, ls=":");ax.grid(alpha=.13, lw=.5);ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.text(.53, .075, "Mean + min–max along each hanging-wall partition; not confidence limits.", ha="center", fontsize=7)
    fig.text(.53, .04, "Apparent throw, not resolved down-dip slip.", ha="center", fontsize=8, color="0.35")
    return fig


def growth_frame_limits(out, *, bins=None, sensitivity=(), bin_width_m=1000.):
    """ONE set of plot limits from ALL stored timesteps; no inversion/refit."""
    if bins is None:bins = _bins(out, sensitivity, bin_width_m)
    r = out.restoration
    def rate_limits(kind):
        d = bins[bins.kind == kind];high = max(float(d.range_high_m_Myr.max()), 1.)
        return (-.03 * high if kind == "fault_throw" else 0., high * 1.12)
    return dict(x=(float(r.x[0] / 1e3), float(r.x[-1] / 1e3)),
                depth=(float(np.nanmax(r.base)) + 250, min(-1000., float(np.nanmin(r.top)) - 1000)),
                throw=rate_limits("fault_throw"), sediment=rate_limits("sedimentation"))


def growth_frame_steps(out, *, include_present=True):
    """Discrete unit steps, modern at top/first; no invented intermediate ages."""
    rows = []
    if include_present:rows.append(dict(unit_index=-1, age_Ma=0., geometry_age_Ma=0., label="Present day · 0 Ma"))
    for j, unit in enumerate(out.restoration.units):
        h = out.history[out.history.unit == unit].iloc[0]
        geometry_age = float(out.restoration.ages[min(j + 1, len(out.restoration.ages) - 1)])
        label = f"{h.age_base_Ma:g} Ma · strip {unit}"
        if j == len(out.restoration.units) - 1:label = f"{h.age_base_Ma:g} Ma · no deeper horizon; last geometry {geometry_age:g} Ma"
        rows.append(dict(unit_index=j, age_Ma=float(h.age_base_Ma), geometry_age_Ma=geometry_age, label=label))
    return rows


def plot_growth_frame(out, unit_index, *, sensitivity=(), bin_width_m=1000., bins=None,
                      limits=None, show_fault_numbers=True):
    """ONE timestep: fixed-size section + aligned dual-y rates, fixed ALL-step limits.

    ``unit_index=-1`` is present day; oldest row retains the last constrained
    geometry and explicitly reports that no deeper section/slip is inferred.
    """
    import matplotlib.pyplot as plt
    r = out.restoration;nu = len(r.units)
    if not -1 <= unit_index < nu:raise ValueError("Frame index must be -1 (present) or a valid unit index")
    if bins is None:bins = _bins(out, sensitivity, bin_width_m)
    if limits is None:limits = growth_frame_limits(out, bins=bins)
    fig = plt.figure(figsize=(12.5, 5.5), dpi=110)
    section = fig.add_axes([.085, .38, .83, .49]);rate = fig.add_axes([.085, .12, .83, .15], sharex=section);sediment = rate.twinx()
    stage = max(0, min(unit_index + 1, nu - 1));j = max(unit_index, 0);unit = r.units[j]
    draw_restored_section(section, out, stage, show_fault_numbers=show_fault_numbers)
    title = _row_title(out, j) if unit_index >= 0 else "Present day · 0 Ma (rates: youngest 0–5.3 Ma interval)"
    section.set_title(title, fontsize=9, loc="left");section.tick_params(labelbottom=False)
    color = section_material_colors(r.units)[j]
    handles = [h for h in [_draw_rate(rate, bins, unit, "fault_throw", color), _draw_rate(sediment, bins, unit, "sedimentation", color)] if h is not None]
    section.set_ylim(*limits["depth"]);section.set_xlim(*limits["x"])
    rate.set_ylim(*limits["throw"]);sediment.set_ylim(*limits["sediment"])
    rate.set_ylabel("throw rate\n(m/Myr)", color=color, fontsize=9);sediment.set_ylabel("sed. rate\n(m/Myr)", color=color, fontsize=9)
    rate.tick_params(axis="y", colors=color, labelsize=8);sediment.tick_params(axis="y", colors=color, labelsize=8)
    rate.set_xlabel("interpreted input distance (km)", fontsize=9);rate.grid(alpha=.18)
    rate.legend(handles=handles, labels=[h.get_label() for h in handles], loc="upper center", bbox_to_anchor=(.5, 1.34), ncol=2, frameon=False, fontsize=7)
    fig.suptitle("Fixed-trace spatial reconstruction · PROTOTYPE", fontsize=11, y=.98)
    datum = out.metadata.get("assumed_datum_m")
    datum_label = "datum target off" if datum is None else ("assumed zero" if datum == 0 else f"assumed datum {datum:g} m")
    fig.text(.5, .015, f"Fixed all-step axes · {datum_label} · vertical shear, not curved-fault balancing · bars: spatial/sensitivity range, not CI", ha="center", fontsize=8)
    return fig


def plot_growth_frames(out, *, sensitivity=(), bin_width_m=1000., bins=None, show_fault_numbers=True, include_present=True):
    """Notebook-callable separate timestep figures, identical shape and axis ranges."""
    if bins is None:bins = _bins(out, sensitivity, bin_width_m)
    limits = growth_frame_limits(out, bins=bins)
    return [plot_growth_frame(out, step["unit_index"], bins=bins, limits=limits, show_fault_numbers=show_fault_numbers)
            for step in growth_frame_steps(out, include_present=include_present)]


def save_growth_animation(out, filename, *, fps=1., sensitivity=(), bin_width_m=1000., bins=None,
                          show_fault_numbers=True, include_present=True):
    """Save a fixed-frame GIF from EXISTING results; no external encoder or refit."""
    import io
    from pathlib import Path
    import matplotlib.pyplot as plt
    from PIL import Image
    if not np.isfinite(fps) or fps <= 0:raise ValueError("fps must be finite positive")
    filename = Path(filename)
    if filename.suffix.lower() != ".gif":raise ValueError("Use a .gif filename (no external encoder required)")
    if bins is None:bins = _bins(out, sensitivity, bin_width_m)
    limits = growth_frame_limits(out, bins=bins);images = []
    for step in growth_frame_steps(out, include_present=include_present):
        fig = plot_growth_frame(out, step["unit_index"], bins=bins, limits=limits, show_fault_numbers=show_fault_numbers)
        buf = io.BytesIO();fig.savefig(buf, format="png", dpi=110);plt.close(fig)
        buf.seek(0);images.append(Image.open(buf).convert("RGB"))
    palette = images[0].quantize(colors=256)
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    quantized[0].save(filename, save_all=True, append_images=quantized[1:], duration=round(1000 / fps), loop=0, disposal=2)
    return filename
