# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Flexural backstripping of a section and reverse post-rift modelling."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..flexure import G
from ..constants import RHO_W, RHO_M, MCK
from .plate import plate_deflection
from .section import Section, _solid, _thickness, _mass


@dataclass
class Restoration:
    ages: np.ndarray              # (nt,) top age of the youngest remaining unit
    x: np.ndarray
    units: list
    top: np.ndarray               # (nt, nu, nx) depth below sea level of unit tops (bathymetric restoration), NaN = stripped
    base: np.ndarray
    seafloor: np.ndarray          # (nt, nx) W - dSL
    sea_surface: np.ndarray       # (nt,) -dSL
    S: np.ndarray                 # (nt, nx) decompacted sediment thickness
    rho_s: np.ndarray             # (nt, nx)
    w: np.ndarray                 # (nt, nx) flexural deflection under sediment load
    w_airy: np.ndarray            # (nt, nx)
    Z: np.ndarray                 # (nt, nx) flexural water-loaded tectonic subsidence
    Z_airy: np.ndarray
    unloaded_base: np.ndarray     # (nt, nx) base of section after pure flexural unloading from present
    unloaded_seafloor: np.ndarray
    Te: float = 0.0


def backstrip_section(sec: Section, Te=1500.0, method="fft", break_index=None, rho_m=RHO_M, rho_w=RHO_W):
    """Flexural backstripping of a correlated :class:`Section` (see module docstring)."""
    nu, nx = sec.h.shape; drho = rho_m - rho_w; dx = sec.x[1] - sec.x[0]
    # present-day grain thicknesses (compacted at present burial)
    D = np.zeros(nx); Gs = np.zeros((nu, nx))
    for k in range(nu):
        Gs[k] = _solid(sec.h[k], D, sec.F[k], sec.phi0, sec.C); D = D + sec.h[k]
    B0 = sec.W0 + D                                   # present base (TD horizon) depth below sea level
    nt = nu
    out = {k: [] for k in ("top", "base", "seafloor", "ss", "S", "rho", "w", "wa", "Z", "Za", "ub", "us")}
    w0 = None
    for i in range(nt):
        Dz = np.zeros(nx); top = np.full((nu, nx), np.nan); base = np.full((nu, nx), np.nan); M = np.zeros(nx)
        for k in range(i, nu):
            T = _thickness(Gs[k], Dz, sec.F[k], sec.phi0, sec.C)
            M += _mass(T, Dz, sec.F[k], sec.phi0, sec.C, sec.rho_g, rho_w)
            top[k] = Dz; base[k] = Dz + T; Dz = Dz + T
        S = Dz; rho = np.where(S > 0, M / np.maximum(S, 1e-9), rho_w)
        q = (rho - rho_w) * G * S
        w = plate_deflection(q, dx, Te, drho, method, break_index)
        wa = q / (drho * G)
        Wt, SLt = sec.W[i], sec.SL[i]
        if i == 0:
            Wt = sec.W0; w0 = w
        sf = Wt - SLt
        Z = Wt - SLt * rho_m / drho + S - w
        Za = Wt - SLt * rho_m / drho + S - wa
        ub = B0 - (w0 - w); us = ub - S
        out["top"].append(top + sf); out["base"].append(base + sf); out["seafloor"].append(sf); out["ss"].append(-SLt)
        out["S"].append(S); out["rho"].append(rho); out["w"].append(w); out["wa"].append(wa)
        out["Z"].append(Z); out["Za"].append(Za); out["ub"].append(ub); out["us"].append(us)
    A = lambda k: np.array(out[k])
    return Restoration(ages=sec.age_top.copy(), x=sec.x, units=sec.units, top=A("top"), base=A("base"),
                       seafloor=A("seafloor"), sea_surface=A("ss"), S=A("S"), rho_s=A("rho"), w=A("w"), w_airy=A("wa"),
                       Z=A("Z"), Z_airy=A("Za"), unloaded_base=A("ub"), unloaded_seafloor=A("us"), Te=Te)


def te_sensitivity(sec, Te_values, **kw):
    """RMS difference (m) between flexural and Airy tectonic subsidence for each Te."""
    out = []
    for Te in Te_values:
        r = backstrip_section(sec, Te, **kw)
        out.append(np.sqrt(np.nanmean((r.Z - r.Z_airy) ** 2)))
    return np.array(out)


def reverse_postrift(rest: Restoration, sec: Section, beta, rift_end_Ma, rho_m=RHO_M, rho_w=RHO_W):
    """Reverse post-rift modelling (Roberts et al. 1998, steps 1-6) for ages <= rift end.

    Starting from the present section, each restoration is (a) flexurally
    unloaded (``rest.unloaded_*``: base raised by w_0 - w_t, top layers stripped,
    rest decompacted), (b) lifted by the McKenzie water-loaded thermal subsidence
    that happened since t, dZ_th = S_th(t_rift - 0) - S_th(t_rift - t), with
    beta scalar or array on x, and (c) corrected for eustasy, water-loaded:
    base += (dSL_t - dSL_0) rho_w/(rho_m - rho_w); the paleo sea surface is at -dSL_t
    (depths measured below present sea level).
    Returns dict(ages, seafloor[t,x], sea_surface[t], water_depth[t,x], base[t,x]).
    Water depth < 0 means the restored seafloor is above sea level.
    """
    from ..thermal import mckenzie_postrift
    drho = rho_m - rho_w
    keep = rest.ages <= rift_end_Ma + 1e-9
    beta = np.broadcast_to(np.asarray(beta, float), rest.x.shape)
    sf, base, ss, wd = [], [], [], []
    SL0 = -rest.sea_surface[0]
    for i in np.where(keep)[0]:
        t = rest.ages[i]
        p = dict(MCK, rho_m=rho_m, rho_w=rho_w)
        dth = (np.array([mckenzie_postrift(np.array([rift_end_Ma]), b, p)[0] - mckenzie_postrift(np.array([rift_end_Ma - t]), b, p)[0]
                         for b in beta]))
        SLt = -rest.sea_surface[i]
        b_ = rest.unloaded_base[i] - dth + (SLt - SL0) * rho_w / drho
        s_ = b_ - rest.S[i]
        sf.append(s_); base.append(b_); ss.append(-SLt); wd.append(s_ + SLt)
    return dict(ages=rest.ages[keep], seafloor=np.array(sf), base=np.array(base), sea_surface=np.array(ss),
                water_depth=np.array(wd))


def fit_beta_section(rest, sec, rift_end_Ma, betas=np.linspace(1.0, 2.0, 101), at=None):
    """Constant beta minimising RMS(predicted - input paleo-water depth) at the wells, post-rift ages only."""
    at = sec.xw if at is None else at
    idx = [int(np.argmin(abs(rest.x - xi))) for xi in at]
    keep = rest.ages <= rift_end_Ma + 1e-9
    Wobs = np.array([sec.W0 if i == 0 else sec.W[i] for i in np.where(keep)[0]])[:, idx]
    mis = []
    for b in betas:
        rp = reverse_postrift(rest, sec, b, rift_end_Ma)
        mis.append(np.sqrt(np.mean((rp["water_depth"][:, idx] - Wobs) ** 2)))
    mis = np.array(mis); k = int(np.argmin(mis))
    return betas[k], mis[k], betas, mis
