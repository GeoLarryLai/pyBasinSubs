# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fixed-trace spatial growth PROTOTYPE: nonnegative interval vertical-shear fields.

Each fault's hanging-wall interval has a piecewise-linear throw field along input
x, independent through unit time steps. A bounded linear inverse problem jointly
fits stripped/decompacted/flexurally unloaded underlying tops, with block-centred
smoothness, an optional assumed datum, growth-thickness priors and spatial
curvature regularization. This is NOT a curved-fault or down-dip slip solution:
throw is an apparent distributed vertical displacement, not slip on a fault plane.
No trace/tip is moved. The vertical map (x,z)->(x,z-u(x)) preserves column thickness
and polygon area; there is no inferred horizontal geological heave or gap infill.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear

from . import flex2d
from .growth import _blocks, _tip_age, structural_history
from .growth_contacts import projected_contacts, tighten_block_contacts


def _prescribed_rates(value, nu, nx, fs, masks):
    result = np.full((nu - 1, nx), np.nan)
    if value is None:return result
    a = np.asarray(value, float)
    if np.any(np.isinf(a)) or np.any(a[np.isfinite(a)] < 0):
        raise ValueError("Provided throw rates must be nonnegative finite or NaN (infer)")
    if a.ndim == 0:return np.full_like(result, float(a))
    if a.shape == (nu - 1, nx):return a.copy()
    if a.shape == (len(fs),):a = np.broadcast_to(a, (nu - 1, len(fs)))
    if a.shape == (nu - 1, len(fs)):
        for k, m in enumerate(masks):result[:, m] = a[:, k, None]
        return result
    raise ValueError("Rates must be scalar, per fault, interval/fault, or (nunit-1, nx) field")


def reconstruct_spatial_growth(sec, faults, *, throw_rates_m_Myr=None, assumed_datum_m=0.,
                               Te=1500., Hd=8000., beta=1.15, rift_end_Ma=155., margin_m=1500.,
                               knot_spacing_m=1000., smooth_sigma_m=100., datum_sigma_m=150.,
                               growth_sigma_m=600., curvature_sigma_m=600., max_throw_m=2500., fault_face_weight=1.,
                               bin_width_m=1000., make_figures=True, contact_placement="auto",
                               footwall_flattening=True):
    """Infer spatial/time-variable apparent normal throw, with interpreted traces FIXED.

    Optional supplied rates: scalar; per fault; (nunit-1,nfault); or a full
    (nunit-1,nx) spatial field. Each interval/block must be wholly prescribed or
    wholly NaN/inferred. Prescribed values are honored (scaled by eligible activity
    fraction), not smoothed into different inputs. Rates use the MODEL durations.
    No independent down-dip distribution is recoverable from these 2-D constraints.

    ``footwall_flattening`` (default on) gives the westernmost fault's footwall,
    which is not the hanging wall of any other fault, the same regularized
    vertical flattening field as every hanging-wall partition (fault #1's
    footwall was otherwise left at flexural-unloading depth only). It uses the
    same datum/smoothness/curvature terms with a zero growth target, is gated by
    fault #1 activity, and is reported as footwall flattening, NEVER as throw.
    """
    if len(sec.units) != 10 or str(sec.units[3]) != "Palaeocene":
        raise ValueError("Spatial prototype requires the ten-unit VG1 stratigraphy")
    scales = [knot_spacing_m, smooth_sigma_m, datum_sigma_m, growth_sigma_m, curvature_sigma_m, max_throw_m, bin_width_m, fault_face_weight]
    if np.any(~np.isfinite(scales)) or min(scales) <= 0:raise ValueError("Fitting scales and bin width must be finite positive")
    if contact_placement not in {"auto", "off", "tighten"}:
        raise ValueError("contact_placement must be auto, off or tighten")
    fs, interior = _blocks(sec.x, deepcopy(faults), margin_m)
    tips = [float(f.tip_unit if f.tip_unit is not None else 6.) for f in fs]
    if any(t < 3.5 or t > 8. for t in tips):raise ValueError("Input tips must respect the existing mid-Palaeocene cap")
    nu, nx = len(sec.units), len(sec.x);x = np.asarray(sec.x)
    masks = [(x >= f.x0) & (x < (fs[k + 1].x0 if k + 1 < len(fs) else x[-1] + 1)) for k, f in enumerate(fs)]
    duration = np.asarray(sec.age_base) - sec.age_top
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0):raise ValueError("Unit durations must be finite positive")
    cessation = np.array([_tip_age(sec, t) for t in tips])
    active = np.clip((np.asarray(sec.age_base[:-1])[:, None] - cessation) / duration[:-1, None], 0., 1.)
    prescribed = _prescribed_rates(throw_rates_m_Myr, nu, nx, fs, masks)
    base = flex2d.restore_faulted(sec, [], Te=Te, Hd=Hd, beta=beta, rift_end_Ma=rift_end_Ma)
    reference = np.array([base.top[i, i] for i in range(nu)])
    trends = gaussian_filter1d(reference, 30e3 / np.median(np.diff(x)), axis=-1)
    q = np.zeros((nu - 1, nx));fit_rows = [];knot_rows = []
    partitions = list(zip(range(len(fs)), fs, masks, interior, [False] * len(fs)))
    footwall = x < fs[0].x0
    if footwall_flattening and (footwall & (x < fs[0].x0 - margin_m)).sum() >= 4:
        partitions.append((0, fs[0], footwall, x < fs[0].x0 - margin_m, True))
    for k, f, m, mi, is_footwall in partitions:
        xx = x[m];n = len(xx)
        knots = np.linspace(xx[0], xx[-1], max(3, int(np.ceil((xx[-1] - xx[0]) / knot_spacing_m)) + 1))
        basis = np.column_stack([np.interp(xx, knots, v) for v in np.eye(len(knots))])
        eligible = np.flatnonzero(active[:, k] > 0)
        variable = []
        for j in eligible:
            given = prescribed[j, m]
            if is_footwall:variable.append(j);continue
            if np.all(np.isfinite(given)):
                q[j, m] = given * duration[j] * active[j, k]
                if q[j, m].max() > max_throw_m + 1e-9:raise ValueError("Supplied interval throw exceeds max_throw_m")
            elif np.all(np.isnan(given)):variable.append(j)
            else:raise ValueError("Prescribe the whole interval/block field or leave it all NaN to infer")
        nv = len(variable) * len(knots)
        if not nv:continue
        def design(j):
            a = np.zeros((n, nv))
            if j in variable:
                v = variable.index(j);a[:, v * len(knots):(v + 1) * len(knots)] = basis * active[j, k]
            return a
        A = [];b = []
        # Interior scores retain the same fixed original-material windows as before.
        # Fit fault-face samples too (equal weight by default); do not hide
        # over-restoration spikes by optimizing interiors alone.
        weights = np.where(mi[m], 1., fault_face_weight);weights /= np.linalg.norm(weights)
        for i in range(1, nu):
            if not np.any(active[:i, k]):continue
            mat = sum((design(j) for j in range(i)), np.zeros((n, nv)))
            fixed = q[:i, m].sum(axis=0)
            z = reference[i, m] - fixed
            centered = z - trends[i, m];centered -= np.average(centered, weights=weights)
            ac = mat - np.average(mat, axis=0, weights=weights)
            A.append(ac * weights[:, None] / smooth_sigma_m);b.append(centered * weights / smooth_sigma_m)
            if assumed_datum_m is not None:
                A.append(mat * weights[:, None] / datum_sigma_m);b.append((z - assumed_datum_m) * weights / datum_sigma_m)
        for j in variable:
            h = base.base[j, j] - base.top[j, j]
            target = np.zeros(n) if is_footwall else h[m] - np.interp(f.x0 - margin_m, x, h)
            A.append(design(j) * weights[:, None] / growth_sigma_m);b.append(target * weights / growth_sigma_m)
            curv = np.zeros((len(knots) - 2, nv));v = variable.index(j)
            curv[:, v * len(knots):(v + 1) * len(knots)] = np.diff(np.eye(len(knots)), n=2, axis=0)
            A.append(curv / curvature_sigma_m / np.sqrt(len(knots)));b.append(np.zeros(len(curv)))
        aa = np.vstack(A);bb = np.concatenate(b)
        fit = lsq_linear(aa, bb, bounds=(0., max_throw_m), method="bvls", tol=1e-9, max_iter=500)
        for j in variable:
            v = variable.index(j);coeff = fit.x[v * len(knots):(v + 1) * len(knots)]
            q[j, m] = basis @ coeff * active[j, k]
            if is_footwall:continue
            knot_rows.extend(dict(unit=sec.units[j], fault=f.label, x_m=float(xk), interval_throw_m=float(a * active[j, k])) for xk, a in zip(knots, coeff))
        fit_rows.append(dict(fault=f.label + (" footwall flattening" if is_footwall else ""), converged=bool(fit.success), iterations=int(fit.nit), objective=float(np.sum((aa @ fit.x - bb) ** 2)), n_variables=nv))
    top = base.top.copy();bot = base.base.copy()
    for i in range(1, nu):
        displacement = q[:i].sum(axis=0)
        top[i, i:] -= displacement;bot[i, i:] -= displacement
    footwall_field = np.where(footwall, q, 0.) if footwall_flattening else np.zeros_like(q)
    xt = np.broadcast_to(x, top.shape).copy()
    steps = [dict(params=[(0., float(np.mean(q[j, m])), 0.) for m in masks], c=1., corr=None) for j in range(nu - 1)]
    r = flex2d.Restoration2D(sec.age_top.copy(), x.copy(), list(sec.units), top, bot,
                             np.array([top[i, i] for i in range(nu)]), method="spatial_vertical_shear",
                             extras=dict(steps=steps, material_x_top=xt.copy(), material_x_base=xt.copy(), material_blocks=masks,
                                         interval_throw_field_m=q.copy(),
                                         footwall_flattening_field_m=footwall_field.copy(), coordinate_system="input material x; vertical-shear field"))
    out = structural_history(sec, fs, method="domino", restoration=r, Te=Te, Hd=Hd, beta=beta,
                             rift_end_Ma=rift_end_Ma, margin_m=margin_m, assumed_datum_m=assumed_datum_m, make_figures=False)
    # Do not relabel a vertical-shear proxy as recovered horizontal or down-dip slip.
    for column in ["heave_m", "heave_m_Myr", "dip_slip_m", "dip_slip_m_Myr", "rotation_deg", "rotation_deg_Myr"]:out.history[column] = np.nan
    out.history["rotation_kind"] = "not inferred: vertical-shear field, not rigid domino"
    out.history["inference"] = np.where(out.history.unit == sec.units[-1], "not inferred: no underlying horizon", "apparent distributed vertical throw; NOT resolved down-dip slip")
    out.scores["model"] = "spatial vertical-shear prototype"
    spatial = out.spatial.copy();fields = np.r_[q, np.full((1, nx), np.nan)]
    # Footwall has no fault throw definition, not a measured zero.
    fields[:, x < fs[0].x0] = np.nan
    spatial["throw_m"] = fields.ravel();spatial["throw_m_Myr"] = (fields / duration[:, None]).ravel()
    out.spatial = spatial;out.search = pd.DataFrame(fit_rows)
    out.metadata.update(prototype=True, method="spatial_vertical_shear", geometry_mode="fixed interpreted traces; no assists",
                        slip_convention="Nonnegative apparent interval normal throw; down-dip slip and geological heave NOT inferred",
                        slip_input="inferred spatial field" if throw_rates_m_Myr is None else "supplied field/constant with NaN blocks inferred",
                        age_boundaries_Ma=np.r_[sec.age_top, sec.age_base[-1]].tolist(),
                        active_fraction=active.tolist(), tip_positions=tips, cessation_Ma=cessation.tolist(),
                        assumed_datum_m=assumed_datum_m, datum_is_observed=False,
                        knot_spacing_m=knot_spacing_m, bin_width_m=bin_width_m, smooth_sigma_m=smooth_sigma_m,
                        datum_sigma_m=datum_sigma_m, growth_sigma_m=growth_sigma_m, curvature_sigma_m=curvature_sigma_m,
                        fault_face_weight=fault_face_weight,
                        maximum_interval_geological_throw_proxy_m=float(np.max(q)),
                        maximum_cumulative_geological_throw_proxy_m=float(np.max(np.sum(q, axis=0))),
                        along_fault_down_dip_resolved=False, geological_heave_inferred=False,
                        dip_slip_definition="Not inferred by vertical-shear prototype",
                        age_source="Existing VG1 numeric model ages, never replaced by ICS lookup",
                        field_basis=f"Piecewise-linear interval throw at ~{knot_spacing_m:g}-m knots per fixed hanging-wall partition",
                        summary_throw_definition="Arithmetic mean of distributed block samples",
                        max_throw_m=max_throw_m,
                        material_area_preserved=True, minimum_thickness_m=float(np.nanmin(bot - top)),
                        added_cutoffs=False, clipped_crossings=False, default_files_modified=False,
                        coordinate_system="Restored depths plotted at original input x for rate alignment; actual material coordinates exported separately",
                        warning="SPATIAL PROTOTYPE: vertical-shear flattening, not curved-fault balancing. Fixed vertical material partitions; nonunique regularized throw proxy. Zero is ASSUMED. No down-dip/3-D slip inference.")
    out.metadata.update(footwall_flattening=bool(footwall_flattening and footwall_field.any()),
                        footwall_flattening_max_m=float(footwall_field.sum(axis=0).max()),
                        footwall_flattening_definition="Regularized vertical flattening of fault #1 footwall (x < first fault x0); NOT fault throw")
    out.knots = pd.DataFrame(knot_rows)
    contacts = projected_contacts(r, fs)
    # Pure vertical shear already has disjoint horizontal supports. Do not slide
    # all blocks merely to erase the pre-existing 250-m sampling spaces.
    if contact_placement == "tighten" or (contact_placement == "auto" and contacts.envelope_overlap_m.max() > 1e-7):
        out = tighten_block_contacts(out)
    else:
        out.contacts = contacts.merge(contacts, on=["stage", "age_Ma", "fault"], suffixes=("_before", "_after"))
        out.contacts["placement_translation_m"] = 0.;out.contacts["active_restoration_stage"] = np.repeat([False] + np.any(np.cumsum(active, axis=0) > 0, axis=1).tolist(), len(fs))
        r["contact_translation_m"] = np.zeros((nu, len(fs) + 1))
        out.history["auxiliary_contact_translation_increment_m"] = np.where(out.history.unit == sec.units[-1], np.nan, 0.)
        out.history["auxiliary_contact_translation_m_Myr"] = out.history.auxiliary_contact_translation_increment_m / out.history.duration_Myr
        out.metadata.update(contact_method="unilateral envelope check: no interpenetration, no auxiliary placement needed",
                            contact_auxiliary_translation_is_fault_slip=False, maximum_placement_translation_m=0.,
                            maximum_overlap_m=0., maximum_gap_m=float(contacts.groupby("stage").endpoint_gap_m.sum().max()),
                            maximum_sidewall_separation_m=float(contacts.sidewall_separation_m.max()), clipped_area_m2=0., filled_gap_area_m2=0.)
    out.metadata["contact_placement_option"] = contact_placement
    if contact_placement == "off":
        out.metadata["contact_method"] = "Contact placement disabled; input-x placement retained"
    if make_figures:out.figures["every_unit"] = plot_spatial_growth_steps(out, bin_width_m=bin_width_m)
    return out


def binned_spatial_rates(out, *, bin_width_m=1000., sensitivity=()):
    """Fine input-x bins of ACTUAL inferred fields, not repeated scalar fault values.

    Error bars are pooled within-bin spatial and optional fit-sensitivity min/max,
    NOT confidence intervals. Sediment uses the decompacted thickness field.
    """
    if not np.isfinite(bin_width_m) or bin_width_m <= 0:raise ValueError("Bin width must be finite positive")
    x = out.restoration.x;edges = np.arange(x[0], x[-1] + bin_width_m, bin_width_m)
    if edges[-1] < x[-1]:edges = np.r_[edges, x[-1]]
    rows = []
    for unit in out.restoration.units:
        for a, b in zip(edges[:-1], edges[1:]):
            for kind, column in [("fault_throw", "throw_m_Myr"), ("sedimentation", "accumulation_m_Myr")]:
                def values(o):
                    d = o.spatial[(o.spatial.unit == unit) & (o.spatial.x_m >= a) &
                                  (o.spatial.x_m < b if b < edges[-1] else o.spatial.x_m <= b)]
                    return d[column].dropna().to_numpy()
                nominal = values(out)
                if not len(nominal):continue
                pool = np.concatenate([nominal] + [values(o) for o in sensitivity])
                rows.append(dict(unit=unit, x_bin_lo_m=float(a), x_bin_hi_m=float(b), x_m=float((a + b) / 2),
                                 kind=kind, rate_m_Myr=float(np.mean(nominal)),
                                 range_low_m_Myr=float(np.min(pool)), range_high_m_Myr=float(np.max(pool)), n=len(nominal),
                                 errorbar="within-bin spatial + fit-sensitivity range; NOT confidence interval"))
    return pd.DataFrame(rows)


def plot_spatial_growth_steps(out, *, bin_width_m=1000., sensitivity=(), show_fault_numbers=True):
    """Slipped/clipped traces and short aligned independently-scaled rate panels."""
    from .growth_display import plot_spatial_growth_steps as plot
    return plot(out, bin_width_m=bin_width_m, sensitivity=sensitivity, show_fault_numbers=show_fault_numbers)
