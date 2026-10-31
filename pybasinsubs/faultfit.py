# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Full-length fault planes, regularized fault-geometry fitting and restoration ensembles.

Fault planes
------------
``fault_plane`` returns (x, z) of a planar or listric fault from the top of the section down to the
detachment depth Hd (planar faults end at Hd; listric faults flatten into the detachment, which is
drawn for ``flat`` m). ``plot_fault_planes`` draws them on a section axis.

Joint fit of fault dips (and optionally Hd) to pre-rift flatness
----------------------------------------------------------------
The restored pre-rift horizon (default the oldest top) should be flat. With z_k(x) its restored depth
and d the fault dips, the objective is a negative log posterior

    NLL(d, Hd) = (N_eff / 2) ln var[z(x; d, Hd)]  +  (1/2) sum_k ((d_k - d0_k) / sigma_d)^2

(Gaussian flatness residual with unknown variance; Gaussian prior on the detected/picked dips d0 with
sd sigma_d; hard bounds). N_eff is the number of independent residual samples, estimated as section
length / e-folding length of the residual autocorrelation, so that a smooth 80 km residual does not
count as 321 independent points (guards against over-fitting). Uncertainty: conditional profile of NLL
for each dip with the other dips held at the optimum; the ~68 % interval is where NLL - min <= 0.5.
An AIC comparison (2 dNLL_data + 2 k) is reported: dAIC > 0 means the fitted dips do not improve
flatness enough to justify k extra parameters.

Ensemble
--------
``restoration_ensemble`` runs a list of method / geometry cases over Te values and returns the flatness
residual of every run, from which an uncertainty band (P5-P95 across runs) is drawn.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import flex2d


# ---------------------------------------------------------------- planes
def _anchor(f, x, tops):
    """Point (x, z) the plane passes through: trace x0 at the depth of the oldest top."""
    return float(f.x0), float(np.interp(f.x0, x, np.asarray(tops)[-1]))


def fault_plane(f, Hd, kind="planar", x=None, tops=None, z_top=None, dip=None, z0=None, x_top=None, n=200, flat=5000.0):
    """(xs, zs) of the full fault plane from depth z_top (default: the fault's upper tip, else 0) down to the detachment Hd (m).

    planar : straight line at ``dip`` (default Fault.dip) through the trace x0 at the depth of the oldest top.
    listric: ``flex2d.listric_depth`` through (x_top, z0) with near-surface ``dip``; by default fitted to the
             fault stick (``fit_listric_stick``) or, without a stick, the planar anchor and Fault.dip; it
             flattens into the detachment, drawn for ``flat`` m beyond 98 % of the depth range."""
    if z_top is None:                                     # upper tip (Fault.z_tip) or the surface
        z_top = f.z_tip if getattr(f, "z_tip", None) is not None else 0.0
    if getattr(f, "curve", None) is not None and dip is None:     # curved trace (fitted or user stick): drawn as is
        zs = np.linspace(z_top, Hd, n)
        return f.x_at(zs), zs
    d = f.dip if dip is None else dip; t = np.tan(np.radians(d))
    if x_top is None or z0 is None:
        if kind == "listric" and f.stick is not None and len(np.asarray(f.stick)) >= 2 and dip is None:
            x_top, z0, d = flex2d.fit_listric_stick(f.stick, f.down, Hd); t = np.tan(np.radians(d))
        elif x is not None and tops is not None:
            x_top, z0 = _anchor(f, x, tops)
        else:
            x_top, z0 = float(f.x0), 0.0
    if kind == "planar":
        zs = np.linspace(z_top, Hd, n)
        return x_top + f.down * (zs - z0) / t, zs
    if kind != "listric":
        raise ValueError("kind must be 'planar' or 'listric'")
    H = max(Hd - z0, 1.0)
    zu = np.linspace(z_top, z0, max(n // 4, 2))                       # up-dip of the origin: near-surface dip
    xu = x_top + f.down * (zu - z0) / t
    zl = np.linspace(z0, z0 + 0.98 * H, n)
    sl = -H / t * np.log(1 - (zl - z0) / H)
    xl = x_top + f.down * sl
    xd = xl[-1] + f.down * np.linspace(0, flat, 20)
    zd = z0 + H * (1 - np.exp(-(sl[-1] + np.linspace(0, flat, 20)) * t / H))
    return np.r_[xu, xl[1:], xd[1:]], np.r_[zu, zl[1:], zd[1:]]


def plot_fault_planes(ax, faults, Hd, kind="planar", x=None, tops=None, z_top=None, km=True, labels=True, dips=None, **plot_kw):
    """Draw full-length fault planes (to Hd) on a depth-section axis (x in km if km=True)."""
    kw = dict(color="r", lw=1.6); kw.update(plot_kw); s = 1e-3 if km else 1.0
    for k, f in enumerate(faults):
        xs, zs = fault_plane(f, Hd, kind, x, tops, z_top, dip=None if dips is None else dips[k])
        ax.plot(xs * s, zs, **kw)
        if labels:
            dd = f.dip if dips is None else dips[k]
            ax.text(xs[0] * s, zs[0], f" {dd:.0f}°", color=kw["color"], fontsize=7, va="bottom")
    return ax


# ---------------------------------------------------------------- fit
def _flat_z(r, unit, x, xlim):
    z = r["top"][unit][unit]; m = (x >= xlim[0]) & (x <= xlim[1])
    return z[m]


def effective_samples(z, dx):
    """N_eff = length / e-folding length of the autocorrelation of the detrended residual (>= 3)."""
    z = np.asarray(z, float); z = z[np.isfinite(z)]
    r = z - np.polyval(np.polyfit(np.arange(len(z)), z, 1), np.arange(len(z)))
    ac = np.correlate(r, r, "full")[len(r) - 1:]; ac = ac / ac[0]
    below = np.where(ac < np.exp(-1))[0]
    ell = (below[0] if len(below) else len(r)) * dx
    return max(len(r) * dx / max(ell, dx), 3.0), ell


def _with_dips(faults, dips):
    """Faults with new effective dips; curved faults keep their shape (faults.curve_shift_dip)."""
    from . import faults as _F
    return [_F.curve_shift_dip(f, d) if getattr(f, "curve", None) is not None else dataclasses.replace(f, dip=float(d))
            for f, d in zip(faults, dips)]


def fit_fault_geometry(sec, faults, method="cantilever", unit=None, prior=None, prior_sd=None, bounds=(15.0, 75.0),
                       Hd=None, Hd_grid=None, grid_step=2.5, sweeps=2, n_eff=None, xlim=(2e3, 80e3), cutoff_k=2.5, **restore_kw):
    """Fit fault dips (and optionally Hd from ``Hd_grid``) by maximizing flatness of the restored pre-rift
    horizon ``unit`` (default oldest top), regularized toward ``prior`` dips (default: the input dips; for
    listric the stick-fitted near-surface dips) with sd ``prior_sd`` deg, inside ``bounds``.
    Coordinate descent on a grid of ``grid_step`` deg. Returns a dict (see module docstring).

    Faults placed on observed block cutoffs (faults.faults_from_cutoffs, Fault.cutoffs set) are held to them:
    with prior_sd=None the prior sd of each such fault is its cutoff-fit standard error (clipped to 2-6 deg)
    and its search range is limited to prior +- ``cutoff_k`` sd, so a fitted plane cannot leave the observed
    cutoffs; other faults use 10 deg and ``bounds``."""
    x = sec.x; dx = x[1] - x[0]; unit = len(sec.age_top) - 1 if unit is None else unit
    fs = sorted(faults, key=lambda f: f.x0); nf = len(fs)
    kw = dict(restore_kw)
    if method == "cantilever" and Hd is None:
        Hd = kw.pop("Hd", 15e3)
    elif method == "listric" and Hd is None:
        Hd = kw.pop("Hd", 10e3)
    else:
        kw.pop("Hd", None)

    curved = any(getattr(f, "curve", None) is not None for f in fs)

    def run(dips, hd):
        if method == "listric" and curved:      # curved traces: vary their effective dip, listric uses the curves
            return flex2d.restore(sec, fs if dips is None else _with_dips(fs, dips), method="listric", Hd=hd, **kw)
        if method == "listric":
            return flex2d.restore(sec, fs, method="listric", Hd=hd, listric_dips=dips, **kw)
        k2 = dict(kw)
        if method == "cantilever":
            k2["Hd"] = hd
        return flex2d.restore(sec, _with_dips(fs, dips), method=method, **k2)

    r0 = run(None if method == "listric" else [f.dip for f in fs], Hd)
    d_in = np.array(r0["listric_dips"] if method == "listric" and not curved else [f.dip for f in fs], float)
    d0 = d_in.copy() if prior is None else np.asarray(prior, float)
    z0 = _flat_z(r0, unit, x, xlim)
    if n_eff is None:
        n_eff, ell = effective_samples(z0, dx)
    else:
        ell = np.nan
    cut = [getattr(f, "cutoffs", None) for f in fs]
    if prior_sd is None:
        prior_sd = np.array([float(np.clip(c["dip_se"], 2.0, 6.0)) if c else 10.0 for c in cut])
    psd = np.broadcast_to(np.asarray(prior_sd, float), (nf,))
    lim = [(max(bounds[0], d0[k] - cutoff_k * psd[k]), min(bounds[1], d0[k] + cutoff_k * psd[k])) if cut[k] else tuple(bounds)
           for k in range(nf)]
    grids = [np.unique(np.r_[np.arange(lo, hi + 1e-9, grid_step), np.clip(d0[k], lo, hi)]) for k, (lo, hi) in enumerate(lim)]
    grid = np.arange(bounds[0], bounds[1] + 1e-9, grid_step)
    cache = {}

    def nll(dips, hd):
        key = (tuple(np.round(dips, 3)), hd)
        if key not in cache:
            r = run(list(dips), hd); z = _flat_z(r, unit, x, xlim)
            v = float(np.nanvar(z))
            data = 0.5 * n_eff * np.log(v); pri = 0.5 * np.sum(((np.asarray(dips) - d0) / psd) ** 2)
            cache[key] = (data + pri, data, pri, np.sqrt(v))
        return cache[key]

    hds = [Hd] if Hd_grid is None else list(Hd_grid)
    best = None; Hd_prof = []
    for hd in hds:
        d = np.array([np.clip(d_in[k], *lim[k]) for k in range(nf)]); prof = {}
        for _ in range(sweeps):
            for k in range(nf):
                vals = []
                for g in grids[k]:
                    dd = d.copy(); dd[k] = g; vals.append(nll(dd, hd)[0])
                prof[k] = np.array(vals); d[k] = grids[k][int(np.argmin(vals))]
        res = nll(d, hd)
        Hd_prof.append((hd, res[0], res[3]))
        if best is None or res[0] < best[0][0]:
            best = (res, d.copy(), hd, prof)
    (tot, data, pri, sd_flat), d_fit, Hd_fit, prof = best
    # final conditional profiles at the optimum
    prof = {}
    for k in range(nf):
        prof[k] = np.array([nll(np.where(np.arange(nf) == k, g, d_fit), Hd_fit)[0] for g in grids[k]])
    ci = []
    for k in range(nf):
        ok = grids[k][prof[k] - prof[k].min() <= 0.5]
        ci.append((float(ok.min() - grid_step / 2), float(ok.max() + grid_step / 2)))
    start = nll(np.array([np.clip(d_in[k], *lim[k]) for k in range(nf)]), Hd)
    d_aic = 2 * (data - start[1]) + 2 * nf
    r_fit = run(list(d_fit), Hd_fit)
    return dict(method=method, faults=fs if method == "listric" and not curved else _with_dips(fs, d_fit), dips0=d_in, prior=d0,
                dips=d_fit, ci=ci, Hd=Hd_fit, Hd_profile=pd.DataFrame(Hd_prof, columns=["Hd", "nll", "std"]),
                grid=grid, grids=grids, limits=lim, prior_sd=psd, profiles=prof, n_eff=n_eff, corr_length_m=ell, nll=tot, prior_penalty=pri,
                std_before=float(np.nanstd(z0)), std_after=sd_flat, delta_aic=float(d_aic),
                n_runs=len(cache), result=r_fit, restore_kw=dict(kw))


def fit_table(fit):
    """Per-fault table of a fit: x0, input dip, fitted dip, ~68 % conditional-profile interval."""
    return pd.DataFrame(dict(x0_km=[f.x0 / 1e3 for f in fit["faults"]], dip_input=fit["dips0"], dip_fit=fit["dips"],
                             ci_lo=[c[0] for c in fit["ci"]], ci_hi=[c[1] for c in fit["ci"]]))


# ---------------------------------------------------------------- ensemble
def restoration_ensemble(sec, cases, Te_values=(1000.0, 2000.0, 3000.0), unit=None, xlim=(2e3, 80e3), **common):
    """Run every case (dict(label, method, faults, kw)) for every Te. Returns (table, Z) where Z[n, x] is the
    restored pre-rift horizon minus its mean (flatness residual) and table holds std / P95-P5 per run."""
    x = sec.x; unit = len(sec.age_top) - 1 if unit is None else unit; m = (x >= xlim[0]) & (x <= xlim[1])
    rows, Z = [], []
    for c in cases:
        for Te in Te_values:
            kw = dict(common); kw.update(c.get("kw", {})); kw["Te"] = Te
            r = flex2d.restore(sec, c["faults"], method=c["method"], **kw)
            z = r["top"][unit][unit].copy(); z = z - np.nanmean(z[m]); z[~m] = np.nan
            Z.append(z)
            rows.append(dict(label=c["label"], method=c["method"], Te=Te, std=float(np.nanstd(z)),
                             relief_p95_p5=float(np.nanpercentile(z[m], 95) - np.nanpercentile(z[m], 5)),
                             mean_fault_offset=flex2d.restoration_metrics(r, unit, x, c["faults"], xlim=xlim)["mean_fault_offset"]))
    return pd.DataFrame(rows), np.array(Z)
