# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Listric (inclined simple shear) and domino fault restoration driven by growth strata."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

from ..flexure import G
from ..constants import RHO_M, RHO_C
from .plate import plate_deflection
from .section import Section
from .cantilever import restore_faulted
from .result import Restoration2D


# ---------------------------------------------------------------- kinematic fault restoration driven by growth strata
# Two alternatives to the planar cantilever above, both applied after stripping, decompaction,
# flexural unloading and reverse post-rift thermal subsidence:
#  (A) listric fault + inclined (or vertical) simple shear of the hangingwall (Verrall 1981;
#      White, Jackson & McKenzie 1986; Kusznir & Egan 1989): fault depth f(x) decays exponentially
#      from a near-surface dip to a detachment at Hd; for heave e the hangingwall surface subsides
#      s(x) = f(x_b) - f(x_b - e_eff) along shear lines at angle alpha (alpha = 90 deg: vertical shear).
#  (B) rigid domino blocks on planar faults: each block rotates by phi and translates by (heave, a).
# In both, the slip of each interval is fitted to the decompacted thickness of the stripped
# (syn-rift) unit across the hangingwall, D(x) = h(x) - h(footwall cutoff), i.e. the extra
# accommodation the fault created while that unit was deposited (growth wedge), so the vertical
# displacement varies with x. At the first syn-rift step the paleo-bathymetric relief left at the
# end of faulting (base Cretaceous) is added to D (relief assumed created in the rift-climax interval).

def listric_depth(x, x0, z0, dip0, Hd, down=1):
    """Depth (m) of a listric fault: f = z0 + H (1 - exp(-s tan(dip0)/H)), H = Hd - z0, s = down (x - x0) > 0;
    extended up-dip at the near-surface dip for s < 0."""
    x = np.asarray(x, float); s = down * (x - x0); H = max(Hd - z0, 1.0); t = np.tan(np.radians(dip0))
    return np.where(s > 0, z0 + H * (1 - np.exp(-np.maximum(s, 0) * t / H)), z0 + s * t)


def fit_listric(x, ztop, x0, down=1, Hd=10e3, win=1500.0, dips=np.arange(30.0, 75.1, 1.0)):
    """Fit the near-surface dip of a listric fault to the digitized fault face: the segment of the
    offset horizon between the footwall cutoff (x0 - win) and hangingwall cutoff (x0 + win).
    Returns (z0, dip0). With 4x vertical exaggeration and a ~1-2 km high face this is an ESTIMATE."""
    m = (down * (x - x0) >= -win) & (down * (x - x0) <= win)
    xs, zs = x[m], ztop[m]
    z0 = float(np.interp(x0 - down * win, x, ztop))
    best = None
    for d in dips:
        xc = x0 - down * win
        f = listric_depth(xs, xc, z0, d, Hd, down)
        r = np.sqrt(np.mean((np.clip(f, zs.min(), zs.max()) - zs) ** 2))
        if best is None or r < best[0]:
            best = (r, d)
    return x0 - down * win, z0, float(best[1])


def fit_listric_stick(stick, down=1, Hd=10e3, dips=np.arange(30.0, 85.1, 0.5)):
    """Fit the near-surface dip of a listric fault (fixed detachment Hd) to fault-plane points (x, z).
    The origin is the shallowest point. Returns (x_top, z_top, dip0)."""
    st = np.asarray(stick, float); st = st[np.argsort(st[:, 1])]
    xt, zt = st[0]
    best = min(dips, key=lambda d: np.sum((listric_depth(st[:, 0], xt, zt, d, Hd, down) - st[:, 1]) ** 2))
    return float(xt), float(zt), float(best)


def _shear_subsidence(xc, zc, f_of_x, e, down, alpha):
    """Restoration of points (xc, zc) in the hangingwall for heave e: returns (x_orig, s) where s is the
    vertical displacement accrued (subsidence) and x_orig the pre-slip position. Inclined shear lines dip
    towards the fault at alpha from horizontal (alpha=90: vertical shear)."""
    ca = 0.0 if abs(alpha - 90.0) < 1e-9 else 1.0 / np.tan(np.radians(alpha))
    xb = xc.copy()
    for _ in range(30):                                   # shear-line base on the fault
        xb = xc - down * (f_of_x(xb) - zc) * ca
    s = np.zeros_like(xc)
    for _ in range(40):
        s = f_of_x(xb) - f_of_x(xb - down * (e - s * ca))
    return xc - down * (e - s * ca), s


def _fit_heave(xs, zs, fx, D, down, alpha, heaves, solver="brent"):
    """Heave (m) minimising mean (s(e) - D)^2 over the hangingwall points. Returns (misfit, heave)."""
    cost = lambda e: float(np.mean((_shear_subsidence(xs, zs, fx, e, down, alpha)[1] - D) ** 2))
    lo, hi = float(np.min(heaves)), float(np.max(heaves))
    if solver == "grid":
        return min(((cost(e), float(e)) for e in heaves), key=lambda t: t[0])
    if solver != "brent":
        raise ValueError("heave_solver must be 'brent' or 'grid'")
    coarse = np.linspace(lo, hi, max(int((hi - lo) / 500.0), 2) + 1)
    c = [cost(e) for e in coarse]; k = int(np.argmin(c)); step = coarse[1] - coarse[0]
    a, b = max(lo, coarse[k] - step), min(hi, coarse[k] + step)
    res = minimize_scalar(cost, bounds=(a, b), method="bounded", options=dict(xatol=1.0))
    return (res.fun, float(res.x)) if res.fun <= c[k] else (c[k], float(coarse[k]))


def _composite(pieces, x):
    """Merge restored pieces [(xs, zs), ...] (priority order: earlier wins overlaps) onto grid x.
    Returns z, overlap length (m), gap length (m)."""
    dx = x[1] - x[0]; z = np.full_like(x, np.nan); cover = np.zeros(len(x), int)
    segs = []
    for xs, zs in pieces:
        if len(xs) < 2:
            continue
        o = np.argsort(xs); xs, zs = xs[o], zs[o]
        cut = np.where(np.diff(xs) > 1.5 * dx)[0] + 1          # split pieces with internal gaps
        segs += [(a, b) for a, b in zip(np.split(xs, cut), np.split(zs, cut)) if len(a) >= 2]
    for xs, zs in segs:
        inside = (x >= xs[0]) & (x <= xs[-1])
        cover += inside
        fill = inside & np.isnan(z)
        z[fill] = np.interp(x[fill], xs, zs)
    gap = np.isnan(z)
    if gap.any() and (~gap).any():
        z[gap] = np.interp(x[gap], x[~gap], z[~gap])
    return z, float((cover > 1).sum() * dx), float(gap.sum() * dx)


def _fault_fx(f, z0, d0, Hd, use_curve=True):
    """Fault depth function z(x) for the listric restoration: the curved trace (Fault.curve) translated vertically
    to pass through (x0, z0), or the exponential listric shape with near-surface dip d0."""
    if use_curve and getattr(f, "curve", None) is not None:
        zc0 = float(f.z_of_x(f.x0))
        return lambda xx: np.minimum(f.z_of_x(xx) - zc0 + z0, max(Hd, z0 + 1.0))
    return lambda xx: listric_depth(xx, f.x0, z0, d0, Hd, f.down)


def _apply_listric(surfs, x, faults, heaves, alpha, Hd, z0s, dips, use_curve=True):
    """Restore list of surfaces z(x) for slips on listric faults (processed in order)."""
    ov = gp = 0.0
    out = [z.copy() for z in surfs]
    for f, e, z0, d0 in zip(faults, heaves, z0s, dips):
        if e <= 0:
            continue
        fx = _fault_fx(f, z0, d0, Hd, use_curve)
        new = []
        for z in out:
            # footwall / hangingwall split along the fault plane at this surface's depth
            hw = (f.down * (x - f.x0) > 0) & (z < fx(x))
            fw = ~hw
            xo, s = _shear_subsidence(x[hw], z[hw], fx, e, f.down, alpha)
            zz, o, g = _composite([(x[fw], z[fw]), (xo, z[hw] - s)], x)
            ov += o; gp += g; new.append(zz)
        out = new
    return out, ov / max(len(surfs), 1), gp / max(len(surfs), 1)


def _apply_domino(surfs, x, faults, params):
    """Restore surfaces for rigid domino blocks. params[k] = (phi, a, heave) for the block in the
    hangingwall of fault k (between fault k and the next fault). Faults sorted along x."""
    order = np.argsort([f.x0 for f in faults]); faults = [faults[k] for k in order]; params = [params[k] for k in order]
    ov = gp = 0.0; out = []
    for z in surfs:
        pieces = []
        bounds = [x[0] - 1] + [f.x0 for f in faults] + [x[-1] + 1]
        # block 0 = footwall of first fault, fixed
        m0 = x < bounds[1]; pieces.append((x[m0], z[m0]))
        for k, f in enumerate(faults):
            phi, a, hv = params[k]
            m = (x >= bounds[k + 1]) & (x < bounds[k + 2])
            if not m.any():
                continue
            xp = f.x0; zp = np.interp(xp, x, z)
            dxs, dzs = x[m] - xp, z[m] - zp
            xr = xp + dxs * np.cos(phi) - dzs * np.sin(phi)
            zr = zp + dxs * np.sin(phi) + dzs * np.cos(phi) - a
            xr = xr - f.down * hv
            pieces.append((xr, zr))
        zz, o, g = _composite(pieces, x)
        out.append(zz); ov += o; gp += g
    return out, ov / max(len(surfs), 1), gp / max(len(surfs), 1)


def _growth_target(x, h, f, nxt, win, extra=None):
    """D(x) over the hangingwall block of fault f (to the next fault): stripped thickness minus its value
    at the footwall cutoff, plus optional extra relief. Returns (mask, D)."""
    s = f.down * (x - f.x0)
    lim = abs(nxt - f.x0) - win if nxt is not None else np.inf
    m = (s > win) & (s < lim)
    href = np.interp(f.x0 - f.down * win, x, h)
    D = h - href
    if extra is not None:
        D = D + extra
    return m, D


def _flexural_coupling(x, d, Te, rho_fill=2400.0, rho_m=RHO_M):
    """Flexural-cantilever isostatic response to a kinematic drop d(x) (m, down positive).
    Returns (corr, c): vertical correction to add to the kinematically restored surfaces and the
    kinematic amplitude factor c. See restore_kinematic(flexural=True)."""
    dx = x[1] - x[0]
    P = lambda v: -plate_deflection(-(RHO_C - rho_fill) * G * v, dx, Te, rho_m - rho_fill)   # rebound (down +) of drop v
    m = d > 1.0
    if m.sum() < 3:
        return np.zeros_like(x), 1.0
    Pd = P(d)
    den = np.sum(d[m] * (d[m] - Pd[m]))
    c = float(np.clip(np.sum(d[m] ** 2) / den, 1.0, 20.0)) if den > 0 else 1.0
    net = c * d - c * Pd
    return d - net, c


def restore_kinematic(sec: Section, faults, method="listric", alpha=65.0, Te=1500.0, beta=1.15,
                      rift_end_Ma=155.0, fault_end_Ma=140.0, Hd=10e3, win=1500.0, bathy_relief=True,
                      heaves=np.arange(0.0, 8001.0, 50.0), flexural=False, rho_fill=2400.0, rho_m=RHO_M,
                      relief_mode="climax", relief_fraction=1.0, listric_dips=None, heave_solver="brent"):
    """Growth-strata-driven fault restoration. method='listric' (inclined shear at alpha; 90 = vertical)
    or 'domino' (rigid block rotation on planar faults at Fault.dip). Returns a dict like restore_faulted
    plus per-step parameters and overlap/gap lengths.

    Sequence for age step i (top = unit i): geometry G_i from stripping + decompaction + flexural
    unloading + reverse thermal (as restore_faulted with no faults); then the fault restorations of the
    syn-rift intervals j = kend+1 .. i are applied in order (youngest first). Restoration j is fitted to
    the stripped unit j-1 (decompacted thickness at step j-1, in the coordinates already restored).

    flexural=True couples the kinematics with the flexural-cantilever isostatic response (Kusznir et al.
    1991): the kinematic hangingwall drop d(x) of each step (vertical part of the fitted restoration, top
    surface) is treated as the forward slip load q = (rho_c - rho_fill) g c d on an elastic plate (Te,
    restoring density rho_m - rho_fill). The net forward displacement is c d - P(c d) (P = plate response),
    which includes footwall uplift; the amplitude c >= 1 is fitted by least squares so that the net
    hangingwall drop still matches the growth-strata target, and the restoration then removes the net
    field (hangingwall raised by c d - P(c d), footwall lowered by P(c d)). Approximation: one c per step
    and the top-surface field applied to all surfaces (column-uniform flexure).

    Paleo-bathymetric relief at the end of faulting (fault_end_Ma, e.g. base Cretaceous 140 Ma) is an
    assumption: relief_mode='climax' (default; all relief created in the first restored syn-rift
    interval), 'spread' (shared equally by all syn-rift intervals) or 'none' (relief kept as inherited
    bathymetry, never restored); relief_fraction scales it (0-1). bathy_relief=False == relief_mode='none'.
    listric_dips overrides the near-surface listric dips fitted from the fault sticks (one per fault, x order).

    heave_solver: 'brent' (default; scipy.optimize.minimize_scalar, bounded, on [heaves.min(), heaves.max()],
    started from a coarse 500 m scan so a local minimum is not taken) or 'grid' (exhaustive scan of `heaves`)."""
    if method not in ("listric", "domino"):
        raise ValueError("method must be 'listric' or 'domino' (use restore() for 'cantilever')")
    if method == "listric" and not (20.0 <= alpha < 89.0):
        raise ValueError("listric restoration uses inclined (antithetic) simple shear only: 20 <= alpha < 89 deg; "
                         "vertical shear (alpha = 90) is not offered")
    base = restore_faulted(sec, [], Te=Te, beta=beta, rift_end_Ma=rift_end_Ma, fault_end_Ma=fault_end_Ma)
    x = sec.x; ages = sec.age_top; nu = len(ages)
    kend = int(np.argmin(abs(ages - fault_end_Ma)))
    fs = sorted(faults, key=lambda f: f.x0)
    nxt = [fs[k + 1].x0 if k + 1 < len(fs) else None for k in range(len(fs))]
    # listric shapes from the digitized fault faces of the present-day oldest top
    z0s, dips, xts = [], [], []
    for f in fs:
        if f.stick is not None and len(np.asarray(f.stick)) >= 2:
            xt, z0, d0 = fit_listric_stick(f.stick, f.down, Hd)       # traced / detected / imported fault plane
        else:
            xt, z0, d0 = fit_listric(x, sec.W0 + sec.h[:-1].sum(0), f.x0, f.down, Hd, win)
        z0s.append(z0); dips.append(d0); xts.append(xt)
    # curved faults (Fault.curve) are used directly by the listric method (translated with the restored horizon);
    # an explicit listric_dips overrides them with the exponential listric shape
    use_curve = listric_dips is None
    if listric_dips is not None:
        dips = [float(d) for d in listric_dips]
    if not bathy_relief:
        relief_mode = "none"
    if relief_mode not in ("climax", "spread", "none"):
        raise ValueError("relief_mode must be 'climax', 'spread' or 'none'")
    n_syn = max(nu - 1 - kend, 1)
    steps = []                                            # list of (kind, params) for j = kend+1..
    def transform(surfs, upto):
        ov = gp = 0.0
        for st in steps[:upto]:
            if method == "listric":
                surfs, o, g = _apply_listric(surfs, x, st["faults"], st["heave"], alpha, Hd, st["z0"], dips, use_curve)
            else:
                surfs, o, g = _apply_domino(surfs, x, fs, st["params"])
            if st.get("corr") is not None:
                surfs = [z + st["corr"] for z in surfs]
            ov += o; gp += g
        return surfs, ov, gp
    TOP, BASE, SF, OV, GP = [], [], [], [], []
    for i in range(nu):
        if i > kend:
            # fit step i from stripped unit i-1 at step i-1 (already-restored coordinates)
            j = i - 1
            surfs = [base["top"][j][j], base["base"][j][j], base["seafloor"][j]]
            (zt, zb, zsf), _, _ = transform(surfs, len(steps))
            h = np.maximum(zb - zt, 0.0)
            extra = None
            if relief_mode == "climax" and j == kend:
                extra = relief_fraction * zsf
            elif relief_mode == "spread":
                if j == kend:
                    relief0 = zsf.copy()               # relief at end of faulting, in restored coordinates
                extra = relief_fraction * relief0 / n_syn
            top_now = base["top"][i][i]
            (top_now,), _, _ = transform([top_now], len(steps))
            st = {"faults": fs, "z0": [], "heave": [], "params": []}
            dkin = np.zeros_like(x)                               # fitted forward kinematic drop of this step
            for k, f in enumerate(fs):
                ex = None if extra is None else extra - np.interp(f.x0 - f.down * win, x, extra)
                m, D = _growth_target(x, h, f, nxt[k], win, ex)
                if m.sum() < 3 or np.nanmax(np.abs(D[m])) < 1e-6 or not f.active(i):
                    st["z0"].append(np.interp(f.x0, x, top_now)); st["heave"].append(0.0); st["params"].append((0.0, 0.0, 0.0)); continue
                if method == "listric":
                    z0 = float(np.interp(f.x0 - f.down * win, x, top_now))
                    fx = _fault_fx(f, z0, dips[k], Hd, use_curve)
                    best = _fit_heave(x[m], top_now[m], fx, D[m], f.down, alpha, heaves, heave_solver)
                    if best[1] > 0:
                        dkin[m] += np.maximum(_shear_subsidence(x[m], top_now[m], fx, best[1], f.down, alpha)[1], 0.0)
                    st["z0"].append(z0); st["heave"].append(best[1]); st["params"].append(None)
                else:
                    s_ = f.down * (x[m] - f.x0)
                    b, a = np.polyfit(s_, D[m], 1)
                    a = max(a, 0.0)
                    phi = np.arctan(-b * f.down)
                    dkin[m] += np.maximum(a + b * s_, 0.0)
                    st["params"].append((phi, a, a / np.tan(np.radians(f.dip)))); st["z0"].append(0.0); st["heave"].append(0.0)
            st["corr"] = None; st["c"] = 1.0
            if flexural:
                st["corr"], st["c"] = _flexural_coupling(x, dkin, Te, rho_fill, rho_m)
            steps.append(st)
        # geometry at step i
        surfs = [base["top"][i][k] for k in range(i, nu)] + [base["base"][i][k] for k in range(i, nu)] + [base["seafloor"][i]]
        n_apply = max(i - kend, 0)
        surfs, ov, gp = transform(surfs, n_apply)
        nl = nu - i
        top = np.full((nu, len(x)), np.nan); bot = np.full((nu, len(x)), np.nan)
        top[i:] = np.array(surfs[:nl]); bot[i:] = np.array(surfs[nl:2 * nl])
        TOP.append(top); BASE.append(bot); SF.append(surfs[-1]); OV.append(ov); GP.append(gp)
    return Restoration2D(ages=ages, x=x, units=sec.units, top=np.array(TOP), base=np.array(BASE), seafloor=np.array(SF),
                         method=method, extras=dict(overlap_m=np.array(OV), gap_m=np.array(GP), steps=steps,
                                                    listric_dips=dips, listric_z0=z0s, alpha=alpha))
