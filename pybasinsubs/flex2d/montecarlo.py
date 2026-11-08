# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Monte Carlo uncertainty of a 2-D restoration.

Each draw perturbs the compaction parameters of every lithology (surface porosity phi0, decay length C), the
paleo-water depth of every unit and, optionally, the elastic thickness Te; then it backstrips the section
(water-loaded tectonic subsidence, :func:`backstrip_section`) and, if faults are given, restores it with
:func:`restore` (cantilever, listric or domino). Percentile envelopes are returned for the restored horizons
and the tectonic subsidence. Draws use ``numpy.random.default_rng(seed)``: the same seed gives the same result.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd

from .section import Section
from .restoration import backstrip_section


@dataclass
class MonteCarlo2D:
    ages: np.ndarray            # (nt,)
    x: np.ndarray
    units: list
    q: tuple                    # percentiles
    top: np.ndarray             # (nq, nt, nu, nx) restored unit tops (depth, m)
    Z: np.ndarray               # (nq, nt, nx) water-loaded tectonic subsidence (flexural)
    best_top: np.ndarray        # (nt, nu, nx) at the mean parameters
    best_Z: np.ndarray          # (nt, nx)
    params: pd.DataFrame        # one row per draw
    method: str = "backstrip"
    seed: int = 0
    samples: dict = field(default_factory=dict)   # raw draws if keep=True

    def envelope(self, i, unit, what="top"):
        """(lo, mid, hi) for age step i: unit (index or name) top, or what='Z' (unit ignored)."""
        if what == "Z":
            return self.Z[0, i], self.Z[len(self.q) // 2, i], self.Z[-1, i]
        k = self.units.index(unit) if isinstance(unit, str) else int(unit)
        return self.top[0, i, k], self.top[len(self.q) // 2, i, k], self.top[-1, i, k]

    def width(self, what="top"):
        """Mean envelope width (outer percentiles, m) per age step."""
        a = self.Z if what == "Z" else self.top
        w = a[-1] - a[0]
        return np.nanmean(w.reshape(w.shape[0], -1), 1)


def _draw_section(sec: Section, rng, phi0_sd, C_rel_sd, water_sd):
    nl = len(sec.phi0)
    phi0 = np.clip(sec.phi0 + np.broadcast_to(phi0_sd, (nl,)) * rng.standard_normal(nl), 0.05, 0.9)
    C = np.maximum(sec.C * (1 + np.broadcast_to(C_rel_sd, (nl,)) * rng.standard_normal(nl)), 200.0)
    dW = np.broadcast_to(water_sd, (len(sec.units),)) * rng.standard_normal(len(sec.units))
    W = np.maximum(sec.W + dW[:, None], 0.0)
    return replace(sec, phi0=phi0, C=C, W=W), dict(dW_mean=float(dW.mean()),
                                                   **{f"phi0_{j}": v for j, v in enumerate(phi0)},
                                                   **{f"C_{j}": v for j, v in enumerate(C)})


def monte_carlo_section(sec: Section, faults=None, method=None, n=200, seed=0, Te=1500.0, Te_sd=0.0,
                        phi0_sd=0.03, C_rel_sd=0.05, water_sd=50.0, q=(5, 50, 95), keep=False,
                        vary_faults=False, fault_scale=1.0, **kw):
    """Monte Carlo over porosity (phi0 +- phi0_sd, absolute), compaction length (C * (1 +- C_rel_sd)), paleo-water
    depth (W +- water_sd, m, per unit) and optionally Te (+- Te_sd, m, truncated at 0).

    faults/method: None -> backstripping only (restored tops = bathymetric restoration); otherwise
    ``restore(sec, faults, method, Te=..., **kw)`` (method default 'cantilever'). phi0_sd, C_rel_sd, water_sd may be
    scalars or per-lithology / per-unit arrays.
    vary_faults: also draw the curved fault geometry of fitted curves (faults.faults_from_curves) from their fit
    covariance (x_r, dip0, lam) x ``fault_scale`` each realization (faults.perturb_curved_faults); the drawn shallow
    dips are recorded in the params table (dip0_<k>). Returns :class:`MonteCarlo2D` with percentile envelopes ``q``."""
    from . import restore
    rng = np.random.default_rng(seed)
    method = (method or "cantilever") if faults else "backstrip"

    def run(s, te, fs=None):
        b = backstrip_section(s, Te=te)
        if method == "backstrip":
            return b.top, b.Z, b.ages
        r = restore(s, faults if fs is None else fs, method=method, Te=te, **kw)
        ages = np.asarray(r.ages, float)
        iz = [int(np.argmin(np.abs(b.ages - a))) for a in ages]          # Z at the restoration ages
        return np.asarray(r.top, float), b.Z[iz], ages

    best_top, best_Z, ages = run(sec, Te)
    tops, Zs, rows = [], [], []
    for _ in range(int(n)):
        s, prm = _draw_section(sec, rng, phi0_sd, C_rel_sd, water_sd)
        te = max(Te + Te_sd * rng.standard_normal(), 0.0) if Te_sd else Te
        fs = None
        if vary_faults and method != "backstrip":
            from ..faults import perturb_curved_faults
            tops_now = sec.W0 + np.r_[np.zeros((1, sec.h.shape[1])), np.cumsum(sec.h[:-1], 0)]
            Hd = kw.get("Hd", 10e3)
            fs = perturb_curved_faults(faults, rng, sec.x, tops_now, Hd=Hd, scale=fault_scale)
            prm = dict(prm, **{f"dip0_{k}": (f.curve_params or {}).get("dip0", np.nan) for k, f in enumerate(fs)})
        t, z, _a = run(s, te, fs)
        tops.append(t); Zs.append(z); rows.append(dict(Te=te, **prm))
    T = np.array(tops); Z = np.array(Zs)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        topq = np.nanpercentile(T, q, axis=0); Zq = np.nanpercentile(Z, q, axis=0)
    return MonteCarlo2D(ages=ages, x=sec.x, units=list(sec.units), q=tuple(q), top=topq, Z=Zq, best_top=best_top,
                        best_Z=best_Z, params=pd.DataFrame(rows), method=method, seed=seed,
                        samples=dict(top=T, Z=Z) if keep else {})


def plot_mc_section(ax, mc: MonteCarlo2D, i=0, colors=None, units=None, alpha=.3):
    """Draw the restored horizons at age step i with their envelopes (lo-hi shaded, median line)."""
    from ..material_colors import material_color
    xs = mc.x / 1e3
    for k, u in enumerate(mc.units):
        if units is not None and u not in units and k not in units:
            continue
        lo, mid, hi = mc.envelope(i, k)
        if np.all(np.isnan(mid)):
            continue
        c = colors[k] if colors is not None else material_color(u, f"C{k % 10}")
        ax.fill_between(xs, lo, hi, color=c, alpha=alpha, lw=0)
        ax.plot(xs, mid, color=c, lw=1, label=u)
    ax.invert_yaxis() if not ax.yaxis_inverted() else None
    ax.set_xlabel("distance (km)"); ax.set_ylabel("depth (m)")
    return ax
