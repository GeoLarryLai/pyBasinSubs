# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""2-D (profile) flexural backstripping on a thin elastic plate.

Plate equation (Turcotte & Schubert 2014; Watts 2001)
    d2/dx2 ( D(x) d2w/dx2 ) + (rho_a - rho_infill) g w = q(x),
    D = E Te^3 / (12 (1 - nu^2)),  E = 70 GPa, nu = 0.25.
Constant Te: Fourier solution  W(k) = Q(k) / (D k^4 + drho g)
Variable Te: finite differences (pentadiagonal, free ends far from the load).

Flexural backstripping (Roberts et al. 1998; Kusznir et al. 1995): the Airy
equation (paper Eq. 10) can be written
    zeta = sigma - eps (rho_s - rho_w)/(rho_a - rho_w) - dSL rho_w/(rho_a - rho_w)
where eps (rho_s - rho_w)/(rho_a - rho_w) is the *local* isostatic response to the
sediment load (water-filled). In the flexural case it is replaced by w_s(x),
the plate deflection under the load q = (rho_s - rho_w) g eps(x) with water infill:
    zeta_flex(x) = sigma(x) - w_s(x) - dSL rho_w/(rho_a - rho_w).
As Te -> 0, w_s -> local Airy response and zeta_flex -> zeta_Airy.

Te fitting: Te is chosen to make the post-rift tectonic subsidence most
consistent with a McKenzie (1978) cooling curve (min RMS misfit over wells and
post-rift time nodes). This is one defensible criterion among several; with a
handful of wells Te is only weakly constrained - always show the misfit curve.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .constants import G, E_YOUNG, NU, RHO_M, RHO_W


def rigidity(Te, E=E_YOUNG, nu=NU):


    """Flexural rigidity D = E Te^3 / (12 (1 - nu^2)) (N m) for elastic thickness Te (m)."""
    return E * np.asarray(Te, float) ** 3 / (12 * (1 - nu ** 2))


def deflection_fft(q, dx, Te, drho, pad=4):
    """Deflection (m, + down) for load q (Pa) on a uniform grid, constant Te (m)."""
    q = np.asarray(q, float)
    n = len(q); N = int(2 ** np.ceil(np.log2(n * pad)))
    qp = np.zeros(N); off = (N - n) // 2
    qp[off:off + n] = q
    qp[:off] = q[0]; qp[off + n:] = q[-1]          # extend edges (no step at the ends)
    taper = np.ones(N)
    ramp = max(1, off // 2)
    taper[:ramp] = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
    taper[-ramp:] = taper[:ramp][::-1]
    qp = qp * taper
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    W = np.fft.fft(qp) / (rigidity(Te) * k ** 4 + drho * G)
    w = np.real(np.fft.ifft(W))
    return w[off:off + n]


def deflection_fd(q, dx, Te, drho):
    """Finite-difference solution with variable Te(x) (array) and free ends."""
    q = np.asarray(q, float); n = len(q)
    D = rigidity(np.broadcast_to(Te, q.shape))
    # operator: (D w'')'' + drho g w ; second-difference matrix L
    L = diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csr") / dx ** 2
    L = L.tolil(); L[0, :] = 0; L[-1, :] = 0; L = L.tocsr()           # zero moment at ends
    A = L @ diags(D) @ L + diags(np.full(n, drho * G))
    return spsolve(A.tocsc(), q)


def interpolate_columns(x_wells, values, x):
    """Linear interpolation of per-well values (wells x ...) onto profile x (flat beyond ends)."""
    v = np.asarray(values, float)
    return np.array([np.interp(x, x_wells, v[:, j]) for j in range(v.shape[1])]).T if v.ndim == 2 \
        else np.interp(x, x_wells, v)


def flexural_tectonic(x, eps, rho_s, sigma, dSL, Te, rho_a=RHO_M, rho_w=RHO_W, method="fft"):
    """Flexural tectonic subsidence along a profile at one time.

    x (m, uniform), eps sediment thickness, rho_s bulk density, sigma total
    subsidence, dSL sea level: arrays on x. Te scalar (fft) or array (fd).
    Returns zeta_flex, w_sed.
    """
    dx = x[1] - x[0]
    q = (rho_s - rho_w) * G * eps
    drho = rho_a - rho_w
    if np.ndim(Te) == 0 and Te <= 0:
        w = q / (drho * G)
    elif method == "fft" and np.ndim(Te) == 0:
        w = deflection_fft(q, dx, Te, drho)
    else:
        w = deflection_fd(q, dx, Te, drho)
    return sigma - w - dSL * rho_w / drho, w


def section_backstrip(results, x_wells, Te_values, dx=2000.0, margin=150e3, ages=None):
    """Airy vs flexural tectonic subsidence along a well transect.

    results : list of BackstripResult, x_wells : distance along section (m).
    ages    : common time grid (Ma). Default: union of all well node ages. Each
              well's best-run history is linearly interpolated onto this grid
              (before its oldest node the well is treated as undeposited).
    Returns dict(x, ages, airy[time, x], flex{Te: [time, x]}, eps, sigma, w{Te}).
    The section is extended by ``margin`` on both sides with the end-well values
    so that the plate is not artificially truncated.
    """
    xw = np.asarray(x_wells, float)
    x = np.arange(xw.min() - margin, xw.max() + margin + dx, dx)
    if ages is None:
        ages = np.unique(np.concatenate([r.best["age"][0] for r in results]))[::-1]
    ages = np.asarray(ages, float)

    def get(key):                                                         # wells x nodes
        rows = []
        for r in results:
            a = r.best["age"][0]; v = r.best[key][0]
            o = np.argsort(a, kind="stable")
            rows.append(np.interp(ages, a[o], v[o], right=v[o][-1] if key in ("rho_s", "W", "SL") else 0.0))
        return np.array(rows)
    eps, rho, sig, sl = get("eps"), get("rho_s"), get("sigma"), get("SL")
    rw = RHO_W; ra = RHO_M
    out = dict(x=x, ages=ages, xw=xw, eps=[], sigma=[], airy=[], flex={Te: [] for Te in Te_values},
               w={Te: [] for Te in Te_values})
    for t in range(len(ages)):
        e = np.interp(x, xw, eps[:, t]); s = np.interp(x, xw, sig[:, t])
        r = np.interp(x, xw, np.where(eps[:, t] > 0, rho[:, t], ra)); d = np.interp(x, xw, sl[:, t])
        z_airy = s - e * (r - rw) / (ra - rw) - d * rw / (ra - rw)
        out["eps"].append(e); out["sigma"].append(s); out["airy"].append(z_airy)
        for Te in Te_values:
            zf, w = flexural_tectonic(x, e, r, s, d, Te, ra, rw)
            out["flex"][Te].append(zf); out["w"][Te].append(w)
    for k in ("eps", "sigma", "airy"):
        out[k] = np.array(out[k])
    for Te in Te_values:
        out["flex"][Te] = np.array(out["flex"][Te]); out["w"][Te] = np.array(out["w"][Te])
    return out


def fit_te(sec, rift_end_Ma, Te_values=None):
    """Misfit of McKenzie post-rift curves to flexural tectonic subsidence at the wells.

    For each Te and well, beta is fitted to zeta_flex(t) - zeta_flex(rift end);
    returns (Te_values, rms misfit per Te, best Te, betas at best Te).
    """
    from .thermal import fit_beta
    Te_values = list(sec["flex"].keys()) if Te_values is None else Te_values
    ages = sec["ages"]
    misfit = []; betas = {}
    for Te in Te_values:
        zw = np.array([np.interp(sec["xw"], sec["x"], sec["flex"][Te][t]) for t in range(len(ages))]).T
        errs = []; bs = []
        for z in zw:
            b, rms = fit_beta(ages, z, rift_end_Ma)
            errs.append(rms); bs.append(b)
        misfit.append(np.sqrt(np.mean(np.square(errs)))); betas[Te] = bs
    i = int(np.argmin(misfit))
    return np.array(Te_values), np.array(misfit), Te_values[i], betas[Te_values[i]]


def section_geometry(results, x_wells, age, x=None, sea_surface=True):
    """Reconstructed cross-section at ``age`` (Ma) from the wells' backstrip.

    Units are matched by name across wells. At each well the decompacted unit
    tops/thicknesses at ``age`` are linearly interpolated between backstrip
    nodes (zero thickness before deposition) and placed below the paleo-seafloor
    (W - dSL); horizons are then linearly interpolated laterally to ``x``.
    Returns dict(x, units[list of names, oldest last], top[u, x], base[u, x],
    unit_age[u] (mid deposition age), seafloor[x], sea_surface[x]).
    """
    xw = np.asarray(x_wells, float)
    x = xw if x is None else np.asarray(x, float)
    names = []
    for r in results:
        for nm, t in zip(r.column.units.unit, r.column.units.thickness_m):
            if t > 0 and nm not in names:
                names.append(nm)
    mid = {}
    for r in results:
        u = r.column.units
        for nm, a0, a1 in zip(u.unit, u.age_top_Ma, u.age_base_Ma):
            mid.setdefault(nm, 0.5 * (a0 + a1))
    names.sort(key=lambda n: mid[n])
    sf = []; ss = []; TOP = np.full((len(names), len(xw)), np.nan); TH = np.zeros((len(names), len(xw)))
    for i, r in enumerate(results):
        b = r.best; ag = b["age"][0]; o = np.argsort(ag); u = r.column.units
        sea = b["W"][0] - b["SL"][0]
        s = np.interp(age, ag[o], sea[o]); sf.append(s); ss.append(-np.interp(age, ag[o], b["SL"][0][o]))
        for j, nm in enumerate(u.unit):
            if nm not in names or u.thickness_m[j] <= 0 or u.age_base_Ma[j] <= age:
                continue
            k = names.index(nm)
            th = np.nan_to_num(b["thick"][0, :, j])
            # before deposition: thickness 0 at seafloor
            a_n = np.r_[ag, u.age_base_Ma[j]]; th_n = np.r_[th, 0.0]; oo = np.argsort(a_n)
            TH[k, i] = max(0.0, np.interp(age, a_n[oo], th_n[oo]))
    # stack from seafloor down in stratigraphic order (youngest first) -> robust tops
    for i in range(len(xw)):
        z = sf[i]
        for k in range(len(names)):
            TOP[k, i] = z; z += TH[k, i]
    interp = lambda v: np.interp(x, xw, v)
    top = np.array([interp(TOP[k]) for k in range(len(names))])
    base = np.array([interp(TOP[k] + TH[k]) for k in range(len(names))])
    return dict(x=x, units=names, top=top, base=base, unit_age=np.array([mid[n] for n in names]),
                seafloor=interp(np.array(sf)), sea_surface=interp(np.array(ss)))
