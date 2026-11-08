# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Area-preserving unilateral placement for the fixed-trace growth PROTOTYPE.

This is not a curved-fault contact solver. The previous prototype partitions a
continuous digitized mesh at vertical oldest-trace columns, then rotates each
block and closes heave independently. Those columns are not interpreted curved
cutoff pairs. Independent translations can overlap their horizontal envelopes;
250 m between neighbouring sampled columns is also counted as a baseline gap.

Here a west-to-east unilateral envelope constraint registers entire rigid blocks
AFTER slip inference. Each translated block stays wholly east of its neighbour's
horizontal envelope. No polygon is clipped, no gap is filled and no vertical
coordinate, inferred normal throw or rotation is changed. Auxiliary horizontal
placement is exported SEPARATELY from slip-derived planar heave: it is not a new
measured slip component or an input-trace correction. Remaining wedge gaps and
vertical mismatch require actual curved cutoff/contact reconstruction.
"""
from copy import copy, deepcopy

import numpy as np
import pandas as pd

from . import flex2d
from .growth import reconstruct_growth_history, plot_structural_history, plot_growth_space_time
from .material_colors import section_material_colors


def _sidewall_distance(left, right):
    """Minimum Euclidean distance between sampled piecewise-linear material sidewalls."""
    def cross(a, b):return a[0] * b[1] - a[1] * b[0]
    def point_segment(p, a, b):
        v = b - a
        t = np.clip(np.dot(p - a, v) / max(np.dot(v, v), 1e-30), 0., 1.)
        return float(np.linalg.norm(p - a - t * v))
    distance = np.inf
    for a, b in zip(left[:-1], left[1:]):
        for c, d in zip(right[:-1], right[1:]):
            v, w = b - a, d - c;den = cross(v, w)
            if abs(den) > 1e-12:
                t, u = cross(c - a, w) / den, cross(c - a, v) / den
                if 0 <= t <= 1 and 0 <= u <= 1:return 0.
            distance = min(distance, point_segment(a, c, d), point_segment(b, c, d),
                           point_segment(c, a, b), point_segment(d, a, b))
    return distance


def projected_contacts(r, faults):
    """Endpoint projection metrics, comparable to the previous prototype (not overlap area)."""
    masks = [r.x < faults[0].x0] + r["material_blocks"]
    rows = []
    for i, age in enumerate(r.ages):
        for b in range(1, len(masks)):
            lm, rm = masks[b - 1], masks[b]
            left = float(r["material_x_top"][i, i, lm][-1])
            right = float(r["material_x_top"][i, i, rm][0])
            lx = np.r_[r["material_x_top"][i, i:, lm].ravel(), r["material_x_base"][i, i:, lm].ravel()]
            rx = np.r_[r["material_x_top"][i, i:, rm].ravel(), r["material_x_base"][i, i:, rm].ravel()]
            def face(mask, edge):
                ix = np.flatnonzero(mask)[edge]
                return np.column_stack([np.r_[r["material_x_top"][i, i, ix], r["material_x_base"][i, i:, ix]],
                                        np.r_[r.top[i, i, ix], r.base[i, i:, ix]]])
            lf, rf = face(lm, -1), face(rm, 0)
            rows.append(dict(stage=i, age_Ma=float(age), fault=faults[b - 1].label,
                             endpoint_overlap_m=max(left - right, 0.), endpoint_gap_m=max(right - left, 0.),
                             sidewall_separation_m=_sidewall_distance(lf, rf),
                             common_sidewall_depth_span_m=float(min(lf[:, 1].max(), rf[:, 1].max()) - max(lf[:, 1].min(), rf[:, 1].min())),
                             envelope_overlap_m=max(float(np.nanmax(lx) - np.nanmin(rx)), 0.),
                             envelope_gap_m=max(float(np.nanmin(rx) - np.nanmax(lx)), 0.)))
    return pd.DataFrame(rows)


def tighten_block_contacts(previous):
    """Return a new, area-preserving constrained-placement export; inputs untouched.

    Only stages with nonzero inferred fault movement are registered. Pre-fault
    stages retain their original sampling gaps and geometry. Report both raw
    endpoint metrics (including sampling gaps) and active-stage metrics.
    """
    out = copy(previous);out.restoration = deepcopy(previous.restoration)
    out.scores = previous.scores.copy()
    out.scores.loc[out.scores.model != "current default", "model"] = "contact-constrained material"
    out.history = previous.history.copy();out.metadata = deepcopy(previous.metadata)
    out.figures = {};r = out.restoration;fs = out.faults
    masks = [r.x < fs[0].x0] + r["material_blocks"]
    before = projected_contacts(r, fs)
    shifts = np.zeros((len(r.ages), len(masks)));active = np.zeros(len(r.ages), bool)
    for i in range(1, len(r.ages)):
        active[i] = any(abs(p[0]) + abs(p[1]) > 1e-9 for st in r["steps"][:i] for p in st["params"])
        if not active[i]:continue
        for b in range(1, len(masks)):
            left, right = masks[b - 1], masks[b]
            left_edge = max(np.nanmax(r["material_x_top"][i, i:, left]), np.nanmax(r["material_x_base"][i, i:, left]))
            right_edge = min(np.nanmin(r["material_x_top"][i, i:, right]), np.nanmin(r["material_x_base"][i, i:, right]))
            shift = float(left_edge - right_edge)
            r["material_x_top"][i, :, right] += shift;r["material_x_base"][i, :, right] += shift
            shifts[i, b] = shift
    after = projected_contacts(r, fs)
    out.contacts = before.merge(after, on=["stage", "age_Ma", "fault"], suffixes=("_before", "_after"))
    out.contacts["placement_translation_m"] = shifts[:, 1:].ravel()
    out.contacts["active_restoration_stage"] = np.repeat(active, len(fs))
    r["contact_translation_m"] = shifts
    r["seam_x"] = [[(float(r["material_x_top"][i, i, m][0]), float(r["material_x_top"][i, i, m][-1]))
                     for m in masks] for i in range(len(r.ages))]
    def maximum(frame, name):return float(frame.groupby("stage")[name].sum().max())
    out.metadata.update(slip_convention="Non-negative normal-slip inference; auxiliary contact registration is NOT slip",
                        contact_method="unilateral rigid-block horizontal-envelope placement; not curved-fault balancing",
                        contact_auxiliary_translation_is_fault_slip=False, clipped_area_m2=0., filled_gap_area_m2=0.,
                        horizontal_placement_m=shifts.tolist(), maximum_placement_translation_m=float(abs(shifts).max()),
                        previous_maximum_overlap_m=maximum(before, "endpoint_overlap_m"),
                        previous_maximum_gap_m=maximum(before, "endpoint_gap_m"),
                        maximum_overlap_m=maximum(after, "endpoint_overlap_m"), maximum_gap_m=maximum(after, "endpoint_gap_m"),
                        active_stage_maximum_overlap_m=maximum(after[after.stage.isin(np.flatnonzero(active))], "endpoint_overlap_m") if active.any() else 0.,
                        active_stage_maximum_gap_m=maximum(after[after.stage.isin(np.flatnonzero(active))], "endpoint_gap_m") if active.any() else 0.,
                        maximum_sidewall_separation_m=float(after.sidewall_separation_m.max()),
                        maximum_active_envelope_overlap_m=float(after[after.stage.isin(np.flatnonzero(active))].envelope_overlap_m.max()) if active.any() else 0.,
                        contact_metric="sum of projected endpoint lengths per stage, not polygon overlap area; baseline 250-m sampling gaps retained at inactive stages",
                        warning="PROTOTYPE contact placement: polygon areas and heights preserved; auxiliary horizontal registration is not recovered fault heave. Curved cutoff geometry and vertical contacts remain approximate.")
    corrections = {(r.units[j], f.label): (shifts[j + 1, k + 1] - shifts[j, k + 1])
                   for j in range(len(r.units) - 1) for k, f in enumerate(fs)}
    out.history["auxiliary_contact_translation_increment_m"] = [corrections.get((row.unit, row.fault), np.nan) for row in out.history.itertuples()]
    out.history["auxiliary_contact_translation_m_Myr"] = out.history.auxiliary_contact_translation_increment_m / out.history.duration_Myr
    return out


def reconstruct_contact_history(sec, faults, *, make_figures=True, **kwargs):
    """Fixed-input primary reconstruction plus explicit contact placement; no tip assist.

    Returns the same structured growth result, with ``contacts`` audit table and
    separately named auxiliary translations. ``figures["every_unit"]`` pairs
    every restored underlying top with binned throw/sediment rates. Supply small
    fit-sensitivity runs to :func:`plot_growth_layer_steps` for sensitivity bars;
    default bars describe only within-bin spatial / fault spread, NOT a CI.
    """
    if kwargs.get("tip_candidates") is not None:
        raise ValueError("Contact-aware primary path keeps interpreted tips fixed; tip assist is separate")
    previous = reconstruct_growth_history(sec, faults, make_figures=False, **kwargs)
    out = tighten_block_contacts(previous)
    if make_figures:
        out.figures["history"] = plot_structural_history(out)
        out.figures["space_time"] = plot_growth_space_time(out)
        out.figures["every_unit"] = plot_growth_layer_steps(out)
        default = flex2d.restore(sec, faults, method="domino", Te=kwargs.get("Te", 1500.), Hd=kwargs.get("Hd", 8000.),
                                 beta=kwargs.get("beta", 1.15), rift_end_Ma=kwargs.get("rift_end_Ma", 155.))
        out.figures["comparison"] = plot_contact_comparison(default, previous, out)
    return out


def draw_growth_section(ax, out, i):
    r = out.restoration;x = r.x
    masks = [x < out.faults[0].x0] + r["material_blocks"]
    for k in range(i, len(r.units)):
        for m in masks:
            xt, xb = r["material_x_top"][i, k, m], r["material_x_base"][i, k, m]
            ax.fill(np.r_[xt, xb[::-1]] / 1e3, np.r_[r.top[i, k, m], r.base[i, k, m][::-1]],
                    color=section_material_colors(r.units)[k], lw=0, alpha=.8)
    for m in masks:ax.plot(r["material_x_top"][i, i, m] / 1e3, r.top[i, i, m], "k", lw=.6)
    datum = out.metadata["assumed_datum_m"]
    if datum is not None:ax.axhline(datum, color="navy", ls="--", lw=.7)
    ax.set_ylim(float(np.nanmax(r.base[i])) + 300, min(-1000., float(np.nanmin(r.top[i])) - 1000))
    ax.set_ylabel("Depth (m; down +)")


def plot_contact_comparison(default, previous, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True)
    for row, age in enumerate([100.5, 140., 170.3]):
        i = int(np.argmin(abs(default.ages - age)))
        for k in range(i, len(default.units)):
            axes[row, 0].fill_between(default.x / 1e3, default.top[i, k], default.base[i, k],
                                       color=section_material_colors(default.units)[k], alpha=.8, lw=0)
        axes[row, 0].plot(default.x / 1e3, default.top[i, i], "k", lw=.6)
        for col, obj in [(1, previous), (2, out)]:draw_growth_section(axes[row, col], obj, i)
        for col, name in enumerate(["UNCHANGED default / grid", "Previous fixed-trace / material", "Contact-constrained / material"]):
            axes[row, col].set_title(f"{name}: {age:g} Ma", fontsize=10)
            axes[row, col].axhline(0, color="navy", lw=.7, ls="--")
            axes[row, col].set_ylim(max(np.nanmax(default.base[i]), np.nanmax(out.restoration.base[i])) + 300,
                                    min(-1000., float(np.nanmin(default.top[i])) - 1000, float(np.nanmin(out.restoration.top[i])) - 1000))
    for ax in axes[-1]:ax.set_xlabel("Restored distance (km)")
    fig.suptitle("CONTACT PROTOTYPE — rigid horizontal placement, not curved-fault balancing\n"
                 "No trimming, clipping, gap fill or vertical re-datum. Zero is ASSUMED. Auxiliary placement is not fault slip.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def binned_growth_rates(out, *, bin_width_m=10000., sensitivity=()):
    """Along-input-x bins with declared non-statistical error-bar semantics.

    Sediment bars are within-bin spatial min/max of decompacted thickness / unit
    duration. Fault bars span the bin's faults AND supplied fitting-sensitivity
    runs (not measurement uncertainty or confidence intervals). A single fault
    without sensitivity has zero range, not invented uncertainty. Oldest-unit
    slip remains NaN: there is no underlying horizon to reconstruct.
    """
    if not np.isfinite(bin_width_m) or bin_width_m <= 0:raise ValueError("bin_width_m must be positive")
    rows = [];x = out.restoration.x
    edges = np.arange(x[0], x[-1] + bin_width_m, bin_width_m)
    if edges[-1] < x[-1]:edges = np.r_[edges, x[-1]]
    for unit in out.restoration.units:
        for a, b in zip(edges[:-1], edges[1:]):
            sediment = out.spatial[(out.spatial.unit == unit) & (out.spatial.x_m >= a) & (out.spatial.x_m < b if b < edges[-1] else out.spatial.x_m <= b)]
            fault_labels = [f.label for f in out.faults if a <= f.x0 < b or (b == edges[-1] and f.x0 == b)]
            nominal = out.history[(out.history.unit == unit) & out.history.fault.isin(fault_labels)].throw_m_Myr.dropna().to_numpy()
            samples = list(nominal)
            for trial in sensitivity:
                samples.extend(trial.history[(trial.history.unit == unit) & trial.history.fault.isin(fault_labels)].throw_m_Myr.dropna().tolist())
            for kind, values, center in [("sedimentation", sediment.accumulation_m_Myr.to_numpy(), None),
                                          ("fault_throw", np.asarray(samples), float(np.mean(nominal)) if len(nominal) else np.nan)]:
                if not len(values):continue
                rows.append(dict(unit=unit, x_bin_lo_m=float(a), x_bin_hi_m=float(b), x_m=float((a + b) / 2),
                                 kind=kind, rate_m_Myr=float(np.mean(values)) if center is None else center,
                                 range_low_m_Myr=float(np.min(values)), range_high_m_Myr=float(np.max(values)), n=len(values),
                                 errorbar="within-bin spatial min/max" if kind == "sedimentation" else "fault-bin / fit-sensitivity range; NOT confidence interval"))
    return pd.DataFrame(rows)


def plot_growth_layer_steps(out, *, sensitivity=(), bin_width_m=10000., unit_indices=None):
    """Every stripped unit paired with along-x binned slip and sediment RATE panels."""
    import matplotlib.pyplot as plt
    r = out.restoration;bins = binned_growth_rates(out, bin_width_m=bin_width_m, sensitivity=sensitivity)
    units = list(range(len(r.units))) if unit_indices is None else list(unit_indices)
    fig, axes = plt.subplots(len(units), 3, figsize=(18, 3.2 * len(units)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1, 1]})
    colors = section_material_colors(r.units)
    for row, j in enumerate(units):
        ax, slip, sediment = axes[row]
        unit = r.units[j];h = out.history[out.history.unit == unit].iloc[0]
        if j < len(r.units) - 1:
            draw_growth_section(ax, out, j + 1)
            ax.set_title(f"Strip {unit} ({h.age_top_Ma:g}–{h.age_base_Ma:g} Ma)\nUnderlying top at {h.age_base_Ma:g} Ma", fontsize=9)
        else:
            ax.text(.5, .5, f"{unit}: {h.age_top_Ma:g}–{h.age_base_Ma:g} Ma\nNo deeper horizon: restoration / slip NOT inferred.\nSedimentation uses digitized truncated unit base.", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_axis_off()
        for rate_ax, kind, title in [(slip, "fault_throw", "Normal fault throw rate"), (sediment, "sedimentation", "Decompacted sedimentation rate")]:
            d = bins[(bins.unit == unit) & (bins.kind == kind)]
            if len(d):
                y = d.rate_m_Myr.to_numpy();lo = d.range_low_m_Myr.to_numpy();hi = d.range_high_m_Myr.to_numpy()
                rate_ax.errorbar(d.x_m / 1e3, y, yerr=np.maximum(np.vstack([y - lo, hi - y]), 0),
                                 xerr=(d.x_bin_hi_m - d.x_bin_lo_m) / 2e3, fmt="o", ms=4,
                                 capsize=3, color=colors[j], mec="0.25", mew=.5)
            elif kind == "fault_throw":rate_ax.text(.5, .5, "Not inferred\n(no underlying horizon)", transform=rate_ax.transAxes, ha="center")
            rate_ax.set_title(title, fontsize=9);rate_ax.set_ylabel("m/Myr (= mm/kyr)")
            rate_ax.set_ylim(bottom=0);rate_ax.set_xlim(r.x[0] / 1e3, r.x[-1] / 1e3);rate_ax.grid(alpha=.2)
            rate_ax.set_xlabel("Interpreted input distance (km)")
        ax.set_xlabel("Restored material distance (km)")
    fig.suptitle("FIXED-TRACE CONTACT PROTOTYPE — every unit, decompaction + flexural unloading + domino restoration\n"
                 "Zero = assumed datum; envelopes placed without trimming / gap fill, NOT curved-fault balancing. Rates at INPUT x.\n"
                 "Bars: sediment spatial min/max; faults bin + fit-sensitivity range (not CI). Non-negative normal slip only.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - .027 * 10 / len(units)))
    return fig
