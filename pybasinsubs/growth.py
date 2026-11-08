# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Trace-constrained growth-history export with optional tip assist; never edits input files.

Example (not an alternative default VG1 interpretation)::

    from pybasinsubs.fault_tracer import vg1_example
    from pybasinsubs import faults
    from pybasinsubs.growth import reconstruct_growth_history
    sec, tops, base = vg1_example()
    fs = faults.auto_faults("curved_midramp", sec.x, tops, base)
    out = reconstruct_growth_history(sec, fs, assumed_datum_m=0.)  # traces fixed; slip inferred
    out.history.to_csv("growth_history.csv", index=False)
    out.figures["history"].savefig("diagnostic_growth_history.png")

The explorer reuses the package's decompaction, flexural unloading and thermal
reversal, but tracks rigid material blocks with independent interval slip/rotation and
hypothetical upper tips. No Cretaceous cutoffs are inserted, no crossings are
clipped and the digitized section is not balanced by construction. Fixed material
block boundaries are the oldest-horizon fault traces, as in the existing domino
method. Every material point is retained: no gap interpolation or overlap trimming.
Overlaps/gaps between moved blocks are reported, not silently hidden; a full
contact-balanced reconstruction and fault-cutoff interpretation are still needed.

Zero is an OPTIONAL ASSUMED paleo-datum, not measured bathymetry. The fit is an
in-sample diagnostic with growth-thickness priors, not a geological validation.
Tip age within a unit assumes linear deposition in time, not a calibrated age-depth law.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, least_squares

from . import flex2d
from .material_colors import section_material_colors


@dataclass
class GrowthHistory:
    """Tables use m, Myr, degrees; rates are m/Myr (= mm/kyr), degrees/Myr.

    ``history`` has one row per unit/fault; ``sediment`` includes section and block
    means of decompacted depositional thickness. The oldest unit has no inferred
    fault increment (no underlying horizon), represented by NaN, never invented zero.
    """
    restoration: object
    history: pd.DataFrame
    sediment: pd.DataFrame
    scores: pd.DataFrame
    metadata: dict
    figures: dict = field(default_factory=dict)
    search: pd.DataFrame = field(default_factory=pd.DataFrame)
    faults: list = field(default_factory=list)
    spatial: pd.DataFrame = field(default_factory=pd.DataFrame)
    contacts: pd.DataFrame = field(default_factory=pd.DataFrame)


def _blocks(x, faults, margin_m):
    fs = sorted(faults, key=lambda f: f.x0)
    if not fs or any(f.down != 1 for f in fs):
        raise ValueError("Prototype supports nonempty, east-dipping (+x) ordered domino blocks only")
    bounds = [f.x0 for f in fs] + [x[-1]]
    masks = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (x > lo + margin_m) & (x < hi - margin_m)
        if m.sum() < 4:
            raise ValueError("Each hanging-wall block needs >=4 interior samples; reduce margin or refine x")
        masks.append(m)
    return fs, masks


def block_surface_scores(x, surfaces, faults, ages, *, margin_m=1500., reference=None,
                         assumed_datum_m=0., longwave_km=30.):
    """Score each HW interior separately, excluding a declared fault-face margin.

    Smoothness = RMS of (z - fixed regional flexural trend), after removing its
    block mean; mean depth and absolute datum RMSE are reported separately. The
    trend is a 30 km Gaussian of the *unfaulted unloaded* reference surface, never
    a best-fit global horizontal line. A zero datum is a hypothesis only.
    """
    x = np.asarray(x, float); surfaces = np.asarray(surfaces, float)
    fs, masks = _blocks(x, faults, margin_m)
    reference = surfaces if reference is None else np.asarray(reference, float)
    trends = gaussian_filter1d(reference, max(longwave_km * 1e3 / np.median(np.diff(x)), .01), axis=-1)
    rows = []
    for i, age in enumerate(ages):
        for f, m in zip(fs, masks):
            z = surfaces[i, m]; residual = z - trends[i, m]; residual -= np.mean(residual)
            rows.append(dict(stage=i, age_Ma=float(age), fault=f.label, n=int(m.sum()),
                             mean_depth_m=float(np.mean(z)), within_block_std_m=float(np.std(z)),
                             smooth_rms_m=float(np.sqrt(np.mean(residual ** 2))),
                             datum_rmse_m=float(np.sqrt(np.mean((z - assumed_datum_m) ** 2)))
                             if assumed_datum_m is not None else np.nan,
                             relief_p95_p5_m=float(np.ptp(np.percentile(z, [5, 95])))))
    return pd.DataFrame(rows)


def _weighted_mean(x, values, mask=None):
    if mask is not None:
        x, values = x[mask], values[mask]
    return float(np.trapezoid(values, x) / (x[-1] - x[0]))


def structural_history(sec, faults, *, method="domino", restoration=None, Te=1500., Hd=8000.,
                       beta=1.15, rift_end_Ma=155., margin_m=1500., assumed_datum_m=0.,
                       make_figures=True):
    """Export increments AND duration-normalized rates from an existing restoration.

    This function does not search. For kinematic methods it exports fitted steps;
    cantilever exports signed differences of horizon-wise cumulative slip solutions
    (these are not necessarily a monotone forward history). Non-domino rotation is
    explicitly a bulk surface-dip change, not rigid-block rotation.
    """
    fs, masks = _blocks(sec.x, faults, margin_m)
    if len(set(f.label for f in fs)) != len(fs):
        raise ValueError("Fault labels must be unique for an unambiguous history export")
    durations = np.asarray(sec.age_base) - sec.age_top
    if np.any(durations <= 0) or not np.all(np.isfinite(durations)):
        raise ValueError("Every unit must have a finite positive duration")
    if method not in ("domino", "listric", "cantilever"):
        raise ValueError("method must be domino, listric or cantilever")
    base = flex2d.restore_faulted(sec, [], Te=Te, Hd=Hd, beta=beta, rift_end_Ma=rift_end_Ma)
    r = restoration if restoration is not None else flex2d.restore(sec, fs, method=method, Te=Te, Hd=Hd,
                                                                   beta=beta, rift_end_Ma=rift_end_Ma)
    nu = len(sec.units); rows = []; sediment = []
    steps = r.get("steps", [])
    first = nu - len(steps)
    # Explorer supplies a step for EVERY stripped unit (including inactive zeros).
    interval_steps = {first + k - 1: st for k, st in enumerate(steps)}
    for j, (unit, dt) in enumerate(zip(sec.units, durations)):
        h = base.base[j, j] - base.top[j, j]
        for name, mask in [("section", None)] + [(f.label, m) for f, m in zip(fs, masks)]:
            acc = _weighted_mean(sec.x, h, mask)
            sediment.append(dict(unit=unit, age_top_Ma=float(sec.age_top[j]), age_base_Ma=float(sec.age_base[j]),
                                 duration_Myr=float(dt), block=name, accumulation_m=acc, accumulation_m_Myr=acc / dt))
        for k, (f, m) in enumerate(zip(fs, masks)):
            a = hv = phi = 0.; known = j < nu - 1
            st = interval_steps.get(j)
            if known and st is not None:
                if method == "domino":
                    phi, a, hv = st["params"][k]
                else:
                    hv = st["heave"][k]
                    from .flex2d.kinematic import _fault_fx, _shear_subsidence
                    z0 = st["z0"][k]
                    fx = _fault_fx(f, z0, r["listric_dips"][k], Hd)
                    xx = np.array([min(f.x0 + margin_m, sec.x[-1])])
                    a = float(_shear_subsidence(xx, np.array([z0]), fx, hv, f.down, r.get("alpha", 65.))[1][0]) if hv > 0 else 0.
            elif known and method == "cantilever":
                a = float(r["slip"][j + 1, k] - r["slip"][j, k])
                hv = a / np.tan(np.radians(f.cantilever_geometry(Hd)[0]))
            if method != "domino" and known:
                before = np.polyfit(sec.x[m], base.top[j + 1, j + 1, m], 1)[0]
                after = np.polyfit(sec.x[m], r.top[j + 1, j + 1, m], 1)[0]
                # Increment of the cumulative inferred bulk-dip correction, not rigid rotation.
                bulk = np.degrees(np.arctan(before) - np.arctan(after))
                prior = next((rr["rotation_deg"] for rr in reversed(rows) if rr["fault"] == f.label), 0.)
                cumulative_prior = sum(rr["rotation_deg"] for rr in rows if rr["fault"] == f.label)
                phi = np.radians(bulk - cumulative_prior) if np.isfinite(prior) else 0.
            rot = -np.degrees(phi) if method == "domino" else np.degrees(phi)
            rows.append(dict(unit=unit, fault=f.label, age_top_Ma=float(sec.age_top[j]), age_base_Ma=float(sec.age_base[j]),
                             duration_Myr=float(dt), throw_m=float(a) if known else np.nan,
                             heave_m=float(hv) if known else np.nan,
                             dip_slip_m=float(np.hypot(a, hv)) if known else np.nan,
                             rotation_deg=float(rot) if known else np.nan,
                             throw_m_Myr=float(a / dt) if known else np.nan,
                             heave_m_Myr=float(hv / dt) if known else np.nan,
                             dip_slip_m_Myr=float(np.hypot(a, hv) / dt) if known else np.nan,
                             rotation_deg_Myr=float(rot / dt) if known else np.nan,
                             rotation_kind="rigid domino forward rotation" if method == "domino" else "bulk dip correction (not rigid rotation)",
                             inference="interval estimate" if known else "not inferred: no underlying horizon"))
    surfaces = np.array([r.top[i, i] for i in range(nu)])
    scores = block_surface_scores(sec.x, surfaces, fs, sec.age_top, margin_m=margin_m,
                                  reference=np.array([base.top[i, i] for i in range(nu)]), assumed_datum_m=assumed_datum_m)
    out = GrowthHistory(r, pd.DataFrame(rows), pd.DataFrame(sediment), scores,
                        dict(prototype=True, method=method, assumed_datum_m=assumed_datum_m,
                             datum_is_observed=False, margin_m=margin_m, Te_m=Te, Hd_m=Hd, beta=beta, thermal_rift_end_Ma=rift_end_Ma,
                             warning="Diagnostic estimates, not a balanced/validated reconstruction; deposition ages and paleobathymetry uncertain.",
                             accumulation="decompacted unit thickness at its deposition surface; no erosion correction",
                             slip_convention="normal-slip positive; cantilever horizon differences may be signed",
                             dip_slip_definition="planar sqrt(throw^2+heave^2); listric value is an apparent resultant, not a curved-path integral"), faults=deepcopy(fs))
    spatial = []
    for j, unit in enumerate(sec.units):
        h = base.base[j, j] - base.top[j, j]
        for ix, xx in enumerate(sec.x):
            k = max(np.searchsorted([f.x0 for f in fs], xx, side="right") - 1, -1)
            extra = np.nan if k < 0 else float(h[ix] - np.interp(fs[k].x0 - margin_m, sec.x, h))
            spatial.append(dict(unit=unit, age_top_Ma=float(sec.age_top[j]), age_base_Ma=float(sec.age_base[j]),
                                duration_Myr=float(durations[j]), x_m=float(xx), block="footwall" if k < 0 else fs[k].label,
                                accumulation_m=float(h[ix]), accumulation_m_Myr=float(h[ix] / durations[j]),
                                excess_accommodation_proxy_m=extra, excess_accommodation_proxy_m_Myr=extra / durations[j]))
    out.spatial = pd.DataFrame(spatial)
    out.metadata["accommodation_proxy"] = "Decompacted HW minus FW thickness; includes nontectonic sediment supply and bathymetry"
    if make_figures:
        out.figures["history"] = plot_structural_history(out)
        out.figures["space_time"] = plot_growth_space_time(out)
    return out


def _tip_age(sec, position):
    k = int(np.floor(position)); frac = position - k
    return float(sec.age_top[k] + frac * (sec.age_base[k] - sec.age_top[k]))


def _tip_copy(sec, fault, position):
    """Extrapolate a COPY to a candidate horizon fraction; never insert horizon offsets."""
    f = deepcopy(fault); k = int(np.floor(position)); frac = position - k
    tops = sec.W0 + np.r_[np.zeros((1, len(sec.x))), np.cumsum(sec.h[:-1], axis=0)]
    target = tops[k] + frac * sec.h[k]
    zs = np.linspace(float(np.min(target)), float(np.max(target)), 500)
    fun = lambda z: z - np.interp(f.x_at(z), sec.x, target)
    roots = [brentq(fun, a, b) for a, b in zip(zs[:-1], zs[1:]) if fun(a) * fun(b) <= 0]
    if not roots:
        raise ValueError("Candidate tip has no intersection with its horizon inside the section")
    zt = min(roots, key=lambda z: abs(z - (f.z_tip if f.z_tip is not None else np.mean(target))))
    xt = float(f.x_at(zt))
    if not sec.x[0] <= xt <= sec.x[-1]:
        raise ValueError("Candidate tip lies outside the section")
    if f.curve is not None:
        c = np.asarray(f.curve);f.curve = np.vstack([[xt, zt], c[c[:, 1] > zt]])
    f.z_tip = float(zt); f.tip_unit = k
    f.source = "PROTOTYPE: hypothesized tip, not applied to published VG1 linework"
    return f


def reconstruct_growth_history(sec, faults, *, tip_candidates=None, throw_rates_m_Myr=None, assumed_datum_m=0.,
                           Te=1500., Hd=8000., beta=1.15, rift_end_Ma=155., margin_m=1500.,
                           smooth_sigma_m=100., datum_sigma_m=250., growth_sigma_m=300.,
                           max_throw_m=2000., max_rotation_deg=8., search_sweeps=2, max_nfev=60,
                           make_figures=True):
    """Infer interval slip/rotation with interpreted traces FIXED by default (domino).

    Geometry is a user-controlled constraint. Optional ``throw_rates_m_Myr`` fixes
    throw rates: scalar, one per fault, or (nu-1,nf) interval/fault values; NaN
    entries remain inferred. Constants apply during eligible activity only;
    partial-unit rates are multiplied by the active fraction. Units are m/Myr.
    ``tip_candidates`` enables an optional tip-extension assist, not the main path.

    Tip positions are unit index + fractional thickness/age; 3.5 is mid-Palaeocene
    on VG1, the shallowest permitted tip. Each eligible unit has independent throw
    and rotation; heave follows the effective planar dip. No hard global fault-end
    age is imposed. Normal throw is nonnegative. Activity within a tip-containing
    unit is scaled by its older fraction, assuming linear deposition in time.

    Objective: block-centred residual to FIXED longwave flexural reference at every
    stripped stage; an optional absolute oldest-top datum term; and growth-wedge
    thickness residuals, with declared tolerances, plus 0.05 per active interval as
    a weak complexity penalty. Equal block/stage weighting, not grid-count weighting.
    Data residuals exclude fault-face margins; entire profiles, gaps and overlaps
    remain visible in diagnostic figures. Search is local, not a global optimum.
    """
    if len(sec.units) != 10 or str(sec.units[3]) != "Palaeocene":
        raise ValueError("This first tip-cap explorer requires the ten-unit VG1 stratigraphy")
    assist = tip_candidates is not None
    candidates = np.asarray(tip_candidates if assist else [6.], float)
    if not len(candidates) or np.any(~np.isfinite(candidates)) or np.any(candidates < 3.5) or np.any(candidates > 8.):
        raise ValueError("Tip cap: no candidate younger/shallower than mid-Palaeocene (3.5), or deeper than unit 8")
    if any(v <= 0 for v in [smooth_sigma_m, datum_sigma_m, growth_sigma_m, max_throw_m, max_rotation_deg]):
        raise ValueError("All fitting scales/bounds must be positive")
    if search_sweeps < 1 or max_nfev < 1:
        raise ValueError("search_sweeps and max_nfev must be positive")
    fs, masks = _blocks(sec.x, deepcopy(faults), margin_m)
    nf = len(fs); nu = len(sec.units); x = sec.x
    base = flex2d.restore_faulted(sec, [], Te=Te, Hd=Hd, beta=beta, rift_end_Ma=rift_end_Ma)
    reference = np.array([base.top[i, i] for i in range(nu)])
    trends = gaussian_filter1d(reference, 30e3 / np.median(np.diff(x)), axis=-1)
    hdep = np.array([base.base[j, j] - base.top[j, j] for j in range(nu - 1)])
    targets = np.array([[h - np.interp(f.x0 - margin_m, x, h) for f in fs] for h in hdep])
    phi_max = np.radians(max_rotation_deg)
    params = np.zeros((nu - 1, nf, 2))  # interval normal throw (m), restoration rotation (rad)
    for j in range(nu - 1):
        for k, (f, m) in enumerate(zip(fs, masks)):
            b, a = np.polyfit(x[m] - f.x0, targets[j, k, m], 1)
            params[j, k] = [np.clip(a, 0, max_throw_m), np.clip(np.arctan(-b), -phi_max, phi_max)]
    tips = np.array([float(f.tip_unit if f.tip_unit is not None else 6) for f in fs])
    if np.any(tips < 3.5):
        raise ValueError("Input faults already exceed the mid-Palaeocene cap")
    candidate_geometry = {(k, float(t)): (_tip_copy(sec, f, t) if assist else deepcopy(f))
                          for k, f in enumerate(fs) for t in ((set(candidates) | {tips[k]}) if assist else {tips[k]})}
    provided = np.full((nu - 1, nf), np.nan) if throw_rates_m_Myr is None else np.broadcast_to(
        np.asarray(throw_rates_m_Myr, float), (nu - 1, nf)).copy()
    if np.any(np.isinf(provided)) or np.any(provided[np.isfinite(provided)] < 0):
        raise ValueError("Provided normal throw rates must be finite nonnegative or NaN (infer)")
    supplied_throw = provided * (np.asarray(sec.age_base[:-1]) - sec.age_top[:-1])[:, None]
    if np.any(supplied_throw[np.isfinite(supplied_throw)] > max_throw_m):
        raise ValueError("Provided interval throw exceeds max_throw_m; explicitly increase the bound")
    params[..., 0] = np.where(np.isfinite(supplied_throw), supplied_throw, params[..., 0])

    def fractions(tp):
        cessation = np.array([_tip_age(sec, t) for t in tp])
        return np.clip((np.asarray(sec.age_base[:-1])[:, None] - cessation[None, :]) /
                       (np.asarray(sec.age_base[:-1]) - sec.age_top[:-1])[:, None], 0, 1)

    def simulate(p, tp, full=False):
        # Track every material point, without compositing, trimming or filling gaps.
        # All horizons in a block share ONE rigid pivot at each step (unlike
        # horizon-wise grid recombination). This preserves block polygon areas.
        active = fractions(tp);steps = []
        for j in range(nu - 1):
            ps = []
            for k, f in enumerate(fs):
                a, phi = p[j, k] * active[j, k]
                ps.append((phi, a, a / np.tan(np.radians(f.dip))))
            steps.append(dict(params=ps, c=1., corr=None))
        surfaces = reference.copy()
        TOP = np.array(base.top, copy=True) if full else None
        BOT = np.array(base.base, copy=True) if full else None
        XT = np.broadcast_to(x, base.top.shape).copy() if full else None
        XB = XT.copy() if full else None
        bounds = [f.x0 for f in fs] + [x[-1] + 1.]
        full_masks = [(x >= lo) & (x < hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
        matrices = np.tile(np.eye(2), (nf, 1, 1)); offsets = np.zeros((nf, 2))
        seams = []
        for i in range(nu):
            if i:
                for k, f in enumerate(fs):
                    phi, a, hv = steps[i - 1]["params"][k]
                    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
                    pivot0 = np.array([f.x0, np.interp(f.x0, x, base.top[i, i])])
                    pivot = matrices[k] @ pivot0 + offsets[k]
                    offsets[k] = rotation @ offsets[k] + pivot - rotation @ pivot + [-hv, -a]
                    matrices[k] = rotation @ matrices[k]
            edge_x = []
            for k, (f, mask) in enumerate(zip(fs, full_masks)):
                matrix, offset = matrices[k], offsets[k]
                points = np.vstack([x[mask], reference[i, mask]])
                moved = matrix @ points + offset[:, None]
                surfaces[i, mask] = moved[1]
                edge_x.append((float(moved[0, 0]), float(moved[0, -1])))
                if full:
                    for u in range(i, nu):
                        t = matrix @ np.vstack([x[mask], base.top[i, u, mask]]) + offset[:, None]
                        b = matrix @ np.vstack([x[mask], base.base[i, u, mask]]) + offset[:, None]
                        TOP[i, u, mask], XT[i, u, mask] = t[1], t[0]
                        BOT[i, u, mask], XB[i, u, mask] = b[1], b[0]
            fw = x[x < fs[0].x0]
            seams.append(([(float(x[0]), float(fw[-1]))] if len(fw) else []) + edge_x)
        if full:
            # These are material-coordinate arrays: actual horizontal coordinates
            # are material_x_top/base. Do NOT feed them to grid-only plotting helpers.
            return flex2d.Restoration2D(sec.age_top.copy(), x.copy(), list(sec.units), TOP, BOT, surfaces,
                                        method="domino", extras=dict(steps=steps, material_x_top=XT, material_x_base=XB,
                                                                     material_blocks=full_masks, seam_x=seams,
                                                                     coordinate_system="untrimmed material points; use material_x_top/base"))
        return surfaces

    def residual(p, tp):
        z = simulate(p, tp);active = fractions(tp);res = []
        for i in range(1, nu):
            for k, m in enumerate(masks):
                rr = z[i, m] - trends[i, m];rr -= np.mean(rr)
                res.extend(rr / smooth_sigma_m / np.sqrt(m.sum()))
        if assumed_datum_m is not None:
            for m in masks:
                res.extend((z[-1, m] - assumed_datum_m) / datum_sigma_m / np.sqrt(m.sum()))
        for j in range(nu - 1):
            for k, (f, m) in enumerate(zip(fs, masks)):
                a, phi = p[j, k] * active[j, k]
                predicted = a - np.tan(phi) * (x[m] - f.x0)
                res.extend((predicted - targets[j, k, m]) / growth_sigma_m / np.sqrt(m.sum()))
        # Penalize unneeded early activity, not the datum or long-wave paleobathymetry.
        res.append(np.sqrt(.05 * np.count_nonzero(active)))
        return np.nan_to_num(np.asarray(res), nan=1e6, posinf=1e6, neginf=-1e6)

    log = []
    def objective(p, tp):
        rr = residual(p, tp);return float(rr @ rr)
    initial = objective(params, tips)
    for sweep in range(search_sweeps):
        changed = False
        for k, f in enumerate(fs):
            best = (objective(params, tips), params.copy(), tips.copy())
            for tip in sorted((set(candidates) | {tips[k]}) if assist else {tips[k]}):
                trial_tips = tips.copy();trial_tips[k] = tip
                eligible = np.flatnonzero(fractions(trial_tips)[:, k] > 0)
                p0 = params.copy();p0[fractions(trial_tips)[:, k] == 0, k] = 0.
                variables = [(j, col) for j in eligible for col in [0, 1]
                             if col == 1 or not np.isfinite(supplied_throw[j, k])]
                def unpack(v):
                    p = p0.copy()
                    for (j, col), val in zip(variables, v):p[j, k, col] = val
                    return p
                fit = least_squares(lambda v: residual(unpack(v), trial_tips),
                                    np.array([p0[j, k, col] for j, col in variables]),
                                    bounds=([0. if col == 0 else -phi_max for _, col in variables],
                                            [max_throw_m if col == 0 else phi_max for _, col in variables]),
                                    x_scale=[500. if col == 0 else .05 for _, col in variables], diff_step=1e-3,
                                    max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
                pp = unpack(fit.x);score = objective(pp, trial_tips)
                log.append(dict(sweep=sweep + 1, fault=f.label, tip_position=float(tip), cessation_Ma=_tip_age(sec, tip),
                                objective=score, nfev=fit.nfev, converged=bool(fit.success), accepted=False))
                if score < best[0] - 1e-8:
                    best = (score, pp, trial_tips)
            if best[0] < objective(params, tips) - 1e-8:
                params, tips = best[1], best[2];changed = True
                for row in reversed(log):
                    if row["fault"] == f.label and row["sweep"] == sweep + 1 and row["tip_position"] == tips[k]:
                        row["accepted"] = True;break
        if not changed:
            break
    r = simulate(params, tips, full=True)
    out = structural_history(sec, fs, method="domino", restoration=r, Te=Te, Hd=Hd, beta=beta,
                             rift_end_Ma=rift_end_Ma, margin_m=margin_m, assumed_datum_m=assumed_datum_m, make_figures=False)
    default = flex2d.restore(sec, fs, method="domino", Te=Te, Hd=Hd, beta=beta, rift_end_Ma=rift_end_Ma)
    default_scores = block_surface_scores(x, np.array([default.top[i, i] for i in range(nu)]), fs, sec.age_top,
                                           margin_m=margin_m, reference=reference, assumed_datum_m=assumed_datum_m)
    out.scores["model"] = "prototype searched"
    default_scores["model"] = "current default"
    out.scores = pd.concat([default_scores, out.scores], ignore_index=True)
    out.search = pd.DataFrame(log)
    out.faults = [candidate_geometry[(k, float(t))] for k, t in enumerate(tips)]
    out.metadata.update(geometry_mode="optional tip-extension assist" if assist else "fixed interpreted traces (primary workflow)",
                        slip_input="optional supplied throw rates with remaining entries inferred" if throw_rates_m_Myr is not None else "inferred output",
                        age_boundaries_Ma=np.r_[sec.age_top, sec.age_base[-1]].tolist(), age_source="existing section numeric ages; never overwritten by ICS lookup",
                        initial_objective=initial, final_objective=objective(params, tips), tip_positions=tips.tolist(),
                        cessation_Ma=[_tip_age(sec, t) for t in tips], active_fraction=fractions(tips).tolist(),
                        assumed_datum_m=assumed_datum_m, datum_is_observed=False,
                        smooth_sigma_m=smooth_sigma_m, datum_sigma_m=datum_sigma_m, growth_sigma_m=growth_sigma_m,
                        search="local per-fault coordinate search; finite-budget, no global-optimum claim",
                        max_throw_m=max_throw_m, max_rotation_deg=max_rotation_deg,
                        minimum_thickness_m=float(np.nanmin(r.base - r.top)),
                        maximum_overlap_m=max(sum(max(a[1] - b[0], 0.) for a, b in zip(row[:-1], row[1:])) for row in r["seam_x"]),
                        maximum_gap_m=max(sum(max(b[0] - a[1], 0.) for a, b in zip(row[:-1], row[1:])) for row in r["seam_x"]),
                        material_area_preserved=True, coordinate_system=r["coordinate_system"],
                        comparison_note="Default scores sample its existing composited grid; prototype scores retain every material point without interpolation or overlap trimming. Same original block interiors.",
                        added_cutoffs=False, clipped_crossings=False, default_files_modified=False,
                        shallowest_allowed_tip_position=3.5,
                        warning="PROTOTYPE: continuous green sheets partitioned into hypothetical movable blocks; tips/activity and inferred slip are nonunique. No default interpretation replaced.")
    if make_figures:
        out.figures["history"] = plot_structural_history(out)
        out.figures["space_time"] = plot_growth_space_time(out)
        out.figures["restoration"] = plot_growth_diagnostic(out, default)
    return out


def plot_structural_history(out):
    """Notebook/app-ready increment + rate strip; unit duration is explicit on x."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(14, 14), sharex=True)
    hist = out.history;sed = out.sediment
    for fault, group in hist.groupby("fault", sort=False):
        age = (group.age_top_Ma + group.age_base_Ma) / 2
        axes[0, 0].plot(age, group.throw_m, "o-", ms=3, label=fault)
        axes[0, 1].plot(age, group.throw_m_Myr, "o-", ms=3)
        axes[1, 0].plot(age, group.heave_m, "o-", ms=3)
        axes[1, 1].plot(age, group.heave_m_Myr, "o-", ms=3)
        axes[2, 0].plot(age, group.dip_slip_m, "o-", ms=3)
        axes[2, 1].plot(age, group.dip_slip_m_Myr, "o-", ms=3)
        axes[3, 0].plot(age, group.rotation_deg, "o-", ms=3)
        axes[3, 1].plot(age, group.rotation_deg_Myr, "o-", ms=3)
    for block, group in sed.groupby("block", sort=False):
        age = (group.age_top_Ma + group.age_base_Ma) / 2
        axes[4, 0].plot(age, group.accumulation_m, "o-", ms=3, label=block)
        axes[4, 1].plot(age, group.accumulation_m_Myr, "o-", ms=3)
    for ax, label in zip(axes.flat, ["Throw increment (m)", "Throw rate (m/Myr)", "Heave increment (m)", "Heave rate (m/Myr)",
                                    "Dip-slip increment (m)", "Dip-slip rate (m/Myr)", "Forward rotation (degrees)",
                                    "Rotation rate (degrees/Myr)", "Decompacted accumulation (m)", "Accumulation rate (m/Myr)"]):
        ax.set_ylabel(label);ax.axhline(0, color="0.6", lw=.5);ax.grid(alpha=.2)
        for age in hist.age_top_Ma.unique():ax.axvline(age, color="0.8", lw=.4)
    if out.metadata["method"] != "domino":
        axes[3, 0].set_ylabel("Bulk dip correction (degrees)");axes[3, 1].set_ylabel("Bulk dip correction rate (degrees/Myr)")
    axes[0, 0].legend(fontsize=7, ncol=3);axes[4, 0].legend(fontsize=7, ncol=3)
    boundaries = np.r_[hist.age_top_Ma.unique(), hist.age_base_Ma.max()]
    for ax in axes[-1]:
        ax.set_xlabel("Age (Ma); each interval = one stratigraphic unit")
        ax.set_xticks(boundaries, [f"{v:g}" for v in boundaries], rotation=45, fontsize=7)
    fig.suptitle("PROTOTYPE / DIAGNOSTIC structural history — increments AND duration-normalized rates\n"
                 "Inferred growth accommodation; not measured slip. Oldest unit faulting not inferred. Datum is assumed, not observed.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def plot_growth_diagnostic(out, default):
    """Full, uncropped profiles with >=1000 m headroom, zero explicitly hypothetical."""
    import matplotlib.pyplot as plt
    r = out.restoration;x = r.x
    indices = [int(np.argmin(abs(r.ages - age))) for age in (100.5, 140., 170.3)]
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
    for row, i in enumerate(indices):
        for ax, model, name in [(axes[row, 0], default, "current DEFAULT (unchanged)"),
                                (axes[row, 1], r, "searched PROTOTYPE (untrimmed material blocks)")]:
            for k in range(i, len(r.units)):
                color = section_material_colors(r.units)[k]
                if "material_x_top" in model:
                    partition = [x < out.faults[0].x0] + model["material_blocks"]
                    for mask in partition:
                        xt, xb = model["material_x_top"][i, k, mask], model["material_x_base"][i, k, mask]
                        ax.fill(np.r_[xt, xb[::-1]] / 1e3, np.r_[model.top[i, k, mask], model.base[i, k, mask][::-1]],
                                color=color, alpha=.8, lw=0)
                else:
                    ax.fill_between(x / 1e3, model.top[i, k], model.base[i, k], color=color, alpha=.8, lw=0)
            if "material_x_top" in model:
                for mask in [x < out.faults[0].x0] + model["material_blocks"]:
                    ax.plot(model["material_x_top"][i, i, mask] / 1e3, model.top[i, i, mask], "k", lw=.8)
            else:
                ax.plot(x / 1e3, model.top[i, i], "k", lw=.8)
            if out.metadata["assumed_datum_m"] is not None:
                ax.axhline(out.metadata["assumed_datum_m"], color="navy", ls="--", lw=.9, label="ASSUMED paleo-datum (not measured)")
            for f in out.faults:ax.axvline(f.x0 / 1e3, color="0.4", lw=.5, ls=":")
            ax.set_ylim(max(np.nanmax(default.base[i]), np.nanmax(r.base[i])) + 300,
                        min(-1000., float(np.nanmin(default.top[i])) - 1000, float(np.nanmin(r.top[i])) - 1000))
            ax.set_title(f"{name} — {r.ages[i]:g} Ma", fontsize=10);ax.set_ylabel("Depth (m; down +)")
    axes[0, 1].legend(fontsize=7, loc="lower right")
    for ax in axes[-1]:ax.set_xlabel("Distance (km); fault-face margins excluded from fitting, profiles not cropped")
    fig.suptitle("DIAGNOSTIC ONLY — " + out.metadata["geometry_mode"] + "; unit-by-unit slip inference\n"
                 "Strip + decompact + flexural unloading + thermal reversal + domino restoration; zero is an assumption, not validation", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def explore_growth_history(sec, faults, *, tip_candidates=(3.5, 4., 5., 6.), **kwargs):
    """OPTIONAL tip-extension assist; primary workflow is reconstruct_growth_history.

    All interpreted inputs stay untouched. This explores hypothetical extensions
    of interpreted traces, not a replacement for user-controlled geometry.
    """
    return reconstruct_growth_history(sec, faults, tip_candidates=tip_candidates, **kwargs)


def plot_growth_space_time(out):
    """Separate along-x/time panels for fault slip and sediment/accommodation fields."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    hist, spatial = out.history, out.spatial
    x = np.sort(spatial.x_m.unique()) / 1e3
    edges = np.r_[x[0] - (x[1] - x[0]) / 2, (x[1:] + x[:-1]) / 2, x[-1] + (x[-1] - x[-2]) / 2]
    ages = np.r_[hist.age_top_Ma.unique(), hist.age_base_Ma.max()]
    units = list(hist.unit.unique())
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True, sharey=True)
    for col, field in enumerate(["throw_m", "throw_m_Myr"]):
        norm = Normalize(0, max(float(hist[field].max()), 1.))
        for f in out.faults:
            for row in hist[hist.fault == f.label].itertuples():
                val = getattr(row, field)
                if np.isfinite(val):
                    axes[0, col].add_patch(Rectangle((f.x0 / 1e3 - .65, row.age_top_Ma), 1.3, row.duration_Myr,
                                                       color=plt.cm.viridis(norm(val))))
            axes[0, col].text(f.x0 / 1e3, -3., f.label, ha="center", va="bottom", fontsize=8)
        fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes[0, col], label="m" if col == 0 else "m/Myr")
        axes[0, col].set_title("Fault throw increments at interpreted x" if col == 0 else "Fault throw rates at interpreted x")
    for rr, fields in enumerate([("accumulation_m", "accumulation_m_Myr"),
                                 ("excess_accommodation_proxy_m", "excess_accommodation_proxy_m_Myr")], 1):
        for col, field in enumerate(fields):
            values = np.array([spatial[spatial.unit == unit].sort_values("x_m")[field] for unit in units])
            limit = max(float(np.nanmax(np.abs(values))), 1.)
            kw = dict(cmap="viridis", vmin=0, vmax=limit) if rr == 1 else dict(cmap="RdBu_r", vmin=-limit, vmax=limit)
            mesh = axes[rr, col].pcolormesh(edges, ages, values, shading="flat", **kw)
            fig.colorbar(mesh, ax=axes[rr, col], label="m" if col == 0 else "m/Myr")
            axes[rr, col].set_title(("Decompacted sediment accumulation" if rr == 1 else "HW–FW excess accommodation PROXY") +
                                    (" increments" if col == 0 else " rates"))
    for ax in axes.flat:
        ax.set_xlim(edges[0], edges[-1]);ax.set_ylim(ages[-1], -10)
        ax.set_yticks(ages, [f"{age:g}" for age in ages], fontsize=7)
        for age in ages:ax.axhline(age, color="0.6", lw=.3)
        ax.set_ylabel("Unit boundaries (Ma; dataset ages)")
    for ax in axes[-1]:ax.set_xlabel("Distance along original section (km; material coordinates)")
    fig.suptitle("PROTOTYPE EXPORT — fixed input traces by default; slip inferred unless supplied\n"
                 "One unit per time step; VG1 dataset ages preserved. Oldest unit slip not inferred; excess thickness is not uniquely tectonic.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig
