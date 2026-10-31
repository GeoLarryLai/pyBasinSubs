# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Generic 1-D decompaction and Airy backstripping with Monte Carlo uncertainty.

Equations (Lai et al. 2022, after Sclater & Christie 1980; Steckler & Watts 1978)
------------------------------------------------------------------------------
Porosity        phi(z) = phi0 exp(-z/C)                                  (Eq. 5)
Solid thickness G = T - C phi0 e^{-D/C} (1 - e^{-T/C})   for a layer [D, D+T]
Decompaction    G is conserved: solve  T_i - C phi0 e^{-D_i/C}(1-e^{-T_i/C}) = G  (Eqs. 6-7)
Mixed units     paper mode: T_i = sum_k F_k T_i^k  (Eq. 7, solved per lithology k)
                mixed mode: one solve with phi(z) = sum_k F_k phi_k(z)
Total subsid.   sigma = eps + W - dSL              (eps = decompacted sediment thickness)
Mass            Omega = rho_g T + (rho_w - rho_g) C phi0 e^{-D/C}(1 - e^{-T/C})   (Eq. 12)
Bulk density    rho_s = sum_i sum_k F_ik Omega_ik / eps                          (Eq. 11)
Airy tectonic   zeta = eps (rho_a - rho_s)/(rho_a - rho_w) + W - dSL rho_a/(rho_a - rho_w)   (Eq. 10)

Erosion / over-compaction: porosity is irreversible. Each unit's porosity is
evaluated at its *effective* depth max(current depth, deepest depth reached so
far). Present-day solid thickness is computed at the present effective depth,
found by fixed-point iteration over the burial history.

Uncertainty: Monte Carlo (default 2000 draws) over thickness, ages, W, dSL,
lithology parameters (phi0, C, rho_g - shared by all units, i.e. correlated) and
rho_w. The delta method of the R code lives in :mod:`pybasinsubs.legacy`.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .column import Column

from .constants import RHO_W, RHO_W_ERR, RHO_M as RHO_A


# ------------------------------------------------------------------ kernels
def solid_thickness(T, D, phi0, C, F=None):
    """Grain (solid) thickness of a layer of thickness T whose top is at depth D.
    With F (weights on the last axis) the lithologies are mixed."""
    g = C * phi0 * np.exp(-D / C) * (1 - np.exp(-T / C))
    if F is None:
        return T - g
    return T - np.sum(F * g, axis=-1)


def thickness_from_solid(G, D, phi0, C, F=None, tol=1e-9, maxit=60):
    """Invert :func:`solid_thickness` for T (vectorised Newton; f is convex, f' > 0)."""
    G = np.asarray(G, float)
    if F is None:
        surf = phi0 * np.exp(-D / C)
    else:
        surf = np.sum(F * phi0 * np.exp(-D[..., None] / C), -1)
    T = G / np.clip(1 - surf, 0.05, 1)
    for _ in range(maxit):
        if F is None:
            e = np.exp(-(D + T) / C)
            f = T - C * phi0 * (np.exp(-D / C) - e) - G
            fp = 1 - phi0 * e
        else:
            Dk = D[..., None]; Tk = T[..., None]
            e = np.exp(-(Dk + Tk) / C)
            f = T - np.sum(F * C * phi0 * (np.exp(-Dk / C) - e), -1) - G
            fp = 1 - np.sum(F * phi0 * e, -1)
        dT = f / fp
        T = np.maximum(T - dT, 0.0)
        if np.nanmax(np.abs(dT)) < tol:
            break
    return np.where(G <= 0, 0.0, T)


def unit_mass(T, D, phi0, C, rho_g, rho_w):
    """Paper Eq. 12 for one lithology (mass per unit area, kg/m^2)."""
    return rho_g * T + (rho_w - rho_g) * C * phi0 * np.exp(-D / C) * (1 - np.exp(-T / C))


# ------------------------------------------------------------------ inputs
def sample_inputs(col: Column, n=2000, seed=0, deterministic=False, vary=None):
    """Draw Monte Carlo inputs (or the means, if deterministic). Returns a dict.

    vary: optional set of groups to perturb, among
          {'thickness','age','water','sealevel','lithology','rho_w'} (default: all).
    """
    u = col.units
    lt = col.lithologies.loc[col.lith_names]
    S = 1 if deterministic else int(n)
    rng = np.random.default_rng(seed)
    vary = set(vary) if vary is not None else {"thickness", "age", "water", "sealevel", "lithology", "rho_w"}

    def draw(mean, sd, grp, shape=None):
        mean = np.asarray(mean, float); sd = np.asarray(sd, float)
        if deterministic or grp not in vary:
            return np.broadcast_to(mean, (S,) + mean.shape).copy()
        return mean + sd * rng.standard_normal((S,) + mean.shape)

    T0 = np.maximum(draw(u.thickness_m, u.thickness_err_m, "thickness"), 0)
    age = draw(u.age_top_Ma, u.age_top_err_Ma, "age")
    age = np.maximum.accumulate(np.maximum(age, 0), axis=1)
    Wm = u.water_depth_m.to_numpy(float)
    W = draw(Wm, u.water_depth_err_m, "water")
    W = np.where(Wm >= 0, np.maximum(W, 0), W)
    SL = draw(u.sea_level_m, u.sea_level_err_m, "sealevel")
    phi0 = np.clip(draw(lt.phi0, lt.phi0_err, "lithology"), 1e-6, 0.9)
    C = np.maximum(draw(lt.C, lt.C_err, "lithology"), 50.0)
    rho_g = draw(lt.rho_g, lt.rho_g_err, "lithology")
    rho_w = draw(np.array(RHO_W), np.array(RHO_W_ERR), "rho_w")
    return dict(T0=T0, age=age, W=W, SL=SL, phi0=phi0, C=C, rho_g=rho_g, rho_w=rho_w,
                rho_a=np.full(S, RHO_A), F=col.fractions(), S=S,
                eroded=u.eroded.to_numpy(bool), eroded_at=u.eroded_at_Ma.to_numpy(float),
                W_eros=u.water_depth_at_erosion_m.to_numpy(float),
                age_base_med=u.age_base_Ma.to_numpy(float), age_med=u.age_top_Ma.to_numpy(float))


def _nodes(p):
    """Time-ordered list of time nodes (oldest first) as ('top', i) / ('erosion', i)."""
    nodes = [(p["age_med"][i], "top", i) for i in range(len(p["age_med"]))]
    for i in np.where(p["eroded"] & ~np.isnan(p["eroded_at"]))[0]:
        nodes.append((p["eroded_at"][i], "preerosion", i))   # just before removal
        nodes.append((p["eroded_at"][i], "erosion", i))
    tops = set(np.round(p["age_med"], 9))
    for i in range(len(p["age_med"]) - 1):
        ab = p["age_base_med"][i]
        if np.round(ab, 9) not in tops and ab > p["age_med"][i]:
            nodes.append((ab, "base", i))      # start of deposition after a hiatus
    order = {"preerosion": 0, "erosion": 1, "base": 2, "top": 3}
    nodes.sort(key=lambda x: (-x[0], order[x[1]]))
    return nodes


def _active(p, t, kind, i):
    a = p["age_med"]
    act = a >= t - 1e-12
    gone = p["eroded"] & ~np.isnan(p["eroded_at"]) & (p["eroded_at"] >= t - 1e-12)
    if kind == "erosion":
        gone[i] = True
        act = a > t
    if kind == "preerosion":
        gone[i] = False
        act = a > t
    if kind == "base":
        act = a > t
    return act & ~gone


# ------------------------------------------------------------------ engine
def _run(p, mixing="paper", n_iter=6):
    S, n = p["T0"].shape
    F = p["F"]
    phi0, C, rg = p["phi0"], p["C"], p["rho_g"]          # (S,K)
    rw = p["rho_w"].reshape(S); ra = p["rho_a"]
    nodes = _nodes(p)
    present = ~(p["eroded"] & ~np.isnan(p["eroded_at"]))   # units in today's column
    T0 = p["T0"]
    # present-day top depths (eroded-without-age units still count, BasinSubs convention)
    D0 = np.zeros((S, n))
    D0[:, 1:] = np.cumsum(np.where(present, T0, 0), axis=1)[:, :-1]
    Dmax_present = D0.copy()
    for _ in range(n_iter):
        # solid thickness per unit & lithology (S,n,K)
        Dref = np.where(p["eroded"] & ~np.isnan(p["eroded_at"]), 0.0, Dmax_present)[..., None]
        Gk = solid_thickness(T0[..., None], Dref, phi0[:, None, :], C[:, None, :])
        out = _history(p, nodes, Gk, F, phi0, C, rg, rw, ra, mixing)
        new = np.maximum(D0, out["Dmax_final"])
        if np.allclose(new, Dmax_present, atol=1e-6):
            break
        Dmax_present = new
    out["nodes"] = nodes
    return out


def _history(p, nodes, Gk, F, phi0, C, rg, rw, ra, mixing):
    S, n, K = Gk.shape
    m = len(nodes)
    Dmax = np.zeros((S, n))
    tops = np.full((S, m, n), np.nan); thick = np.full((S, m, n), np.nan); efft = np.full((S, m, n), np.nan)
    eps = np.zeros((S, m)); mass = np.zeros((S, m)); und = np.zeros((S, m))
    W = np.zeros((S, m)); SL = np.zeros((S, m)); age = np.zeros((S, m))
    Gmix = np.sum(F[None] * Gk, -1)                         # (S,n)
    for k_, (t, kind, i) in enumerate(nodes):
        act = _active(p, t, kind, i)
        if kind == "top":
            W[:, k_] = p["W"][:, i]; SL[:, k_] = p["SL"][:, i]; age[:, k_] = p["age"][:, i]
        elif kind == "base" or (kind == "preerosion" and k_ > 0):
            # hiatus / pre-erosion: conditions unchanged since the previous node
            W[:, k_] = W[:, k_ - 1]; SL[:, k_] = SL[:, k_ - 1]; age[:, k_] = t
        else:
            W[:, k_] = p["W_eros"][i]; SL[:, k_] = 0.0; age[:, k_] = t
        D = np.zeros(S)
        for j in np.where(act)[0]:
            De = np.maximum(D, Dmax[:, j])
            if mixing == "paper":
                Tk = thickness_from_solid(Gk[:, j, :], De[:, None], phi0, C)
                T = np.sum(F[j] * Tk, -1)
            else:
                T = thickness_from_solid(Gmix[:, j], De, phi0, C, F=np.broadcast_to(F[j], phi0.shape))
            Om = unit_mass(T[:, None], De[:, None], phi0, C, rg, rw[:, None])
            mass[:, k_] += np.sum(F[j] * Om, -1)
            tops[:, k_, j] = D; thick[:, k_, j] = T; efft[:, k_, j] = De
            Dmax[:, j] = np.maximum(Dmax[:, j], D)
            D = D + T
            und[:, k_] += p["T0"][:, j]
        eps[:, k_] = D
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_s = np.where(eps > 0, mass / np.where(eps > 0, eps, 1), 0.0)
    rwm = rw[:, None]; ram = ra[:, None]
    sigma = eps + W - SL
    zeta = eps * (ram - rho_s) / (ram - rwm) + W - SL * ram / (ram - rwm)
    return dict(age=age, eps=eps, sigma=sigma, zeta=zeta, undecompacted=und + W - SL, rho_s=rho_s,
                W=W, SL=SL, tops=tops, thick=thick, eff_tops=efft, Dmax_final=Dmax)


# ------------------------------------------------------------------ results
@dataclass
class BackstripResult:
    column: Column
    best: dict           # deterministic run at mean inputs (S=1)
    mc: dict | None      # Monte Carlo runs (S draws)
    mixing: str
    delta: dict | None = None   # mode='delta': first-order 1-sigma of each quantity (R parity)

    QUANTS = ("undecompacted", "sigma", "zeta", "eps", "rho_s", "W", "SL")

    @property
    def node_labels(self):
        u = self.column.units.unit.astype(str).to_numpy()
        lab = dict(top="{}", erosion="erosion of {}", base="start of {}", preerosion="before erosion of {}")
        return [lab[k].format(u[i]) for _, k, i in self.best["nodes"]]

    def summary(self, ci=(2.5, 16, 50, 84, 97.5)):
        """One row per time node (oldest first): best estimate + MC mean, sd, percentiles."""
        b = self.best
        df = pd.DataFrame(dict(node=self.node_labels, age_Ma=b["age"][0]))
        names = dict(undecompacted="undecompacted_total_m", sigma="total_subsidence_m",
                     zeta="tectonic_subsidence_m", eps="sediment_thickness_m", rho_s="rho_s",
                     W="water_depth_m", SL="sea_level_m")
        for q in self.QUANTS:
            df[names[q]] = b[q][0]
            if self.mc is not None:
                x = self.mc[q]
                df[names[q] + "_sd"] = x.std(0)
                for c in ci:
                    df[f"{names[q]}_p{c:g}"] = np.percentile(x, c, axis=0)
            elif self.delta is not None:
                df[names[q] + "_sd"] = self.delta[q]
        if self.mc is not None:
            df["age_Ma_sd"] = self.mc["age"].std(0)
        return df

    def burial(self):
        """Long table of unit top depth below seafloor/thickness at each node (best run)."""
        b = self.best; u = self.column.units
        rows = []
        for k_, lab in enumerate(self.node_labels):
            for j in range(len(u)):
                if not np.isnan(b["tops"][0, k_, j]):
                    rows.append(dict(node=lab, age_Ma=b["age"][0, k_], unit=u.unit[j],
                                     top_bsf_m=b["tops"][0, k_, j], thickness_m=b["thick"][0, k_, j],
                                     top_bsl_m=b["tops"][0, k_, j] + b["W"][0, k_] - b["SL"][0, k_]))
        return pd.DataFrame(rows)


def _delta(col, mixing, vary=None, rel_step=1e-2):
    """First-order (delta-method) 1-sigma, as the R code: sd = sqrt(sum_j (df/dx_j s_j)^2), with the Jacobian
    by forward differences (step rel_step * s_j) of the same engine. Ages are not varied (fixed time nodes)."""
    p0 = sample_inputs(col, deterministic=True)
    vary = set(vary) if vary is not None else {"thickness", "water", "sealevel", "lithology", "rho_w"}
    u = col.units; lt = col.lithologies.loc[col.lith_names]
    groups = [("T0", u.thickness_err_m, "thickness"), ("W", u.water_depth_err_m, "water"),
              ("SL", u.sea_level_err_m, "sealevel"), ("phi0", lt.phi0_err, "lithology"),
              ("C", lt.C_err, "lithology"), ("rho_g", lt.rho_g_err, "lithology"), ("rho_w", [RHO_W_ERR], "rho_w")]
    pert = [(k, j, float(sd)) for k, sds, g in groups if g in vary
            for j, sd in enumerate(np.atleast_1d(np.asarray(sds, float))) if sd > 0]
    S = 1 + len(pert); p = dict(p0); p["S"] = S
    for k in ("T0", "W", "SL", "phi0", "C", "rho_g", "rho_w", "age"):
        p[k] = np.repeat(p0[k], S, axis=0)
    p["rho_a"] = np.full(S, RHO_A)
    for r, (k, j, sd) in enumerate(pert, start=1):
        if p[k].ndim == 1:
            p[k][r] += rel_step * sd
        else:
            p[k][r, j] += rel_step * sd
    out = _run(p, mixing)
    res = {}
    for q in BackstripResult.QUANTS:
        x = np.asarray(out[q], float)
        J = (x[1:] - x[:1]) / rel_step                     # = df/dx_j * s_j
        res[q] = np.sqrt(np.nansum(J ** 2, 0)) if len(pert) else np.zeros(x.shape[1:])
    return res


def backstrip(col: Column, n=2000, seed=0, mixing="paper", vary=None, mc=True, mode="mc") -> BackstripResult:
    """Decompact and backstrip a column.

    n      : Monte Carlo draws (set mc=False for the deterministic run only)
    mixing : 'paper' (Eq. 7: solve each lithology, weight by F) or 'mixed'
    mode   : 'mc' (Monte Carlo, default) or 'delta' (first-order error propagation as in the R code,
             no sampling; summary() reports the *_sd columns). 'delta' ignores n and seed.
    """
    best = _run(sample_inputs(col, deterministic=True), mixing)
    if mode == "delta":
        return BackstripResult(col, best, None, mixing, delta=_delta(col, mixing, vary))
    if mode != "mc":
        raise ValueError("mode must be 'mc' or 'delta'")
    res_mc = _run(sample_inputs(col, n=n, seed=seed, vary=vary), mixing) if mc else None
    return BackstripResult(col, best, res_mc, mixing)
