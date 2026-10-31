# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Planar faults restored with the flexural cantilever model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..flexure import G
from ..constants import RHO_W, RHO_M, RHO_C, MCK
from .plate import plate_deflection
from .section import Section
from .restoration import backstrip_section
from .result import Restoration2D


# ---------------------------------------------------------------- fault restoration (flexural cantilever)
# Kusznir, Marsden & Egan (1991); Kusznir et al. (1995); Roberts et al. (1998).
# A planar normal fault cuts the elastic upper crust down to a detachment depth Hd.
# Slip t (throw) drops the hangingwall kinematically (vertical simple-shear collapse
# onto the fault plane): v_k(s) = -t for 0 < s < L - e, ramping to 0 at s = L,
# where s = distance into the hangingwall, L = Hd / tan(dip), e = t / tan(dip) (heave).
# The surface mass deficit (rho_c - rho_fill) g v_k loads the plate, which rebounds
# flexurally: w = plate_deflection(q), net uplift u = v_k - w. The footwall rises
# (footwall uplift), the hangingwall subsides less than t, and the offset at the
# fault stays exactly t. For Te -> 0: footwall u = 0, hangingwall u = -t (rho_m - rho_c)/(rho_m - rho_fill).


@dataclass
class Fault:
    x0: float                     # m, fault trace position on the section
    dip: float                    # degrees
    down: int                     # +1: hangingwall on +x side (dips to +x); -1: hangingwall on -x side
    throw: np.ndarray             # (nu,) throw (m) of the TOP horizon of each unit (youngest first)
    label: str = ""
    source: str = ""              # 'digitized: ...' or 'illustrative, estimated - not measured'
    stick: object = None          # optional (n, 2) array of fault-plane points (x, z) in m, from tracing/detection/import
    confidence: float = None      # optional 0-1 score from auto-detection
    tip_unit: int = None          # upper tip: top of this unit (e.g. base post-rift); units younger (index < tip_unit)
    #                               are never cut or slipped. None = no tip (legacy: fault_end_Ma decides)
    z_tip: float = None           # depth (m) of the upper tip on the plane (for drawing)
    cutoffs: object = None        # optional table of footwall / hangingwall cutoffs used to place the plane
    curve: object = None          # optional curved fault trace: (n, 2) array (x, z), z increasing, tip -> detachment
    curve_params: dict = None     # parameters of a fitted curve (faults.fit_fault_curve), for perturbation / refits

    def active(self, i):
        """True if this fault may slip horizon i (i > tip_unit)."""
        return self.tip_unit is None or i > self.tip_unit

    def x_at(self, z):
        """x of the fault at depth z: the curve (linear extrapolation of its end segments), else the plane through
        (x0, depth 0 ... ) at Fault.dip measured from the trace (z_of_x(x0) = 0 for a plain planar fault)."""
        z = np.asarray(z, float)
        if self.curve is None:
            return self.x0 + self.down * z / np.tan(np.radians(self.dip))
        c = np.asarray(self.curve, float); zc, xc = c[:, 1], c[:, 0]
        out = np.interp(z, zc, xc)
        if len(c) >= 2:
            s0 = (xc[1] - xc[0]) / max(zc[1] - zc[0], 1e-9); s1 = (xc[-1] - xc[-2]) / max(zc[-1] - zc[-2], 1e-9)
            out = np.where(z < zc[0], xc[0] + s0 * (z - zc[0]), out)
            out = np.where(z > zc[-1], xc[-1] + s1 * (z - zc[-1]), out)
        return out

    def dip_at(self, z):
        """Local dip (deg) at depth z (Fault.dip for a planar fault)."""
        if self.curve is None:
            return np.full(np.shape(z), float(self.dip)) if np.ndim(z) else float(self.dip)
        z = np.asarray(z, float); h = 25.0
        d = np.degrees(np.arctan2(2 * h, np.abs(self.x_at(z + h) - self.x_at(z - h))))
        return float(d) if d.ndim == 0 else d

    def z_of_x(self, xx):
        """Depth of the curved fault at positions xx (monotone along the dip direction); flat at the curve's
        deepest point beyond it, extrapolated at the top dip above it."""
        c = np.asarray(self.curve, float); s = self.down * (c[:, 0] - c[0, 0]); zc = c[:, 1]
        sx = self.down * (np.asarray(xx, float) - c[0, 0])
        o = np.argsort(s); s, zc = np.maximum.accumulate(s[o]), zc[o]
        out = np.interp(sx, s, zc)
        top_slope = (zc[1] - zc[0]) / max(s[1] - s[0], 1e-9) if len(s) > 1 else 1.0
        return np.where(sx < s[0], zc[0] + top_slope * (sx - s[0]), out)

    def cantilever_geometry(self, Hd):
        """(dip, L) used by the flexural cantilever: a planar fault gives (dip, Hd / tan dip); a curved fault gives
        the local dip where the curve crosses the trace depth (heave and ramp) and the curve's horizontal reach
        from the trace x0 to the detachment Hd as the width L of the collapsing hangingwall."""
        if self.curve is None:
            return float(self.dip), None
        zt = float(self.z_of_x(self.x0))
        L = abs(float(self.x_at(Hd)) - self.x0)
        return float(self.dip_at(zt)), max(L, 250.0)


def cantilever_displacement(x, x0, throw, dip, down=1, Te=1500.0, Hd=15e3, rho_c=RHO_C, rho_fill=RHO_W,
                            rho_m=RHO_M, method="fft", L=None):
    """Net vertical displacement u(x) (m, up positive) from slip `throw` on one planar fault,
    plus the kinematic part v_k and the flexural rebound w (down positive).
    L (m) overrides the width of the collapsing hangingwall (default Hd / tan(dip)); restore_faulted
    uses the distance to the next fault (domino blocks) when block_width=True."""
    x = np.asarray(x, float); dx = x[1] - x[0]
    s = down * (x - x0); tn = np.tan(np.radians(dip)); L = Hd / tn if L is None else L; e = max(abs(throw) / tn, dx)
    vk = np.where(s > 0, -throw * np.clip((L - s) / e, 0.0, 1.0), 0.0)
    q = (rho_c - rho_fill) * G * vk
    w = plate_deflection(q, dx, Te, rho_m - rho_fill, method)
    return vk - w, vk, w


def _block_widths(x, faults, Hd, block_width):
    out = []
    for f in faults:
        if not block_width:
            out.append(None); continue
        nxt = [g.x0 for g in faults if f.down * (g.x0 - f.x0) > 0]
        edge = x[-1] if f.down > 0 else x[0]
        out.append(min([abs(g - f.x0) for g in nxt] + [abs(edge - f.x0)]))
    return out


def _geom(f, Hd, Lblock=None):
    """Cantilever (dip, L) of a fault, planar or curved; L limited by the block width if given."""
    d, L = f.cantilever_geometry(Hd) if hasattr(f, "cantilever_geometry") else (f.dip, None)
    if Lblock is not None:
        L = Lblock if L is None else min(L, Lblock)
    return d, L


def fault_slips(x, faults, i, Te=1500.0, Hd=15e3, rho_fill=2400.0, rho_m=RHO_M, win=1500.0, block_width=False):
    """Fault slips t_j (m) whose combined cantilever displacement reproduces the observed throws
    T_j of horizon i (offset measured `win` either side of each fault): solve A t = T with
    A_jk = offset at fault j produced by unit slip on fault k. Because the isostatic rebound
    partly cancels the kinematic drop, the slip exceeds the observed (net) throw when Te is small."""
    nf = len(faults)
    if i is None or nf == 0:
        return np.zeros(nf)
    T = np.array([f.throw[i] for f in faults]) if np.ndim(i) == 0 else np.asarray(i, float)
    A = np.zeros((nf, nf)); Ls = _block_widths(x, faults, Hd, block_width)
    for k, fk in enumerate(faults):
        dk, Lk = _geom(fk, Hd, Ls[k])
        u = cantilever_displacement(x, fk.x0, 1.0, dk, fk.down, Te, Hd, rho_fill=rho_fill, rho_m=rho_m, L=Lk)[0]
        for j, fj in enumerate(faults):
            A[j, k] = np.interp(fj.x0 - fj.down * win, x, u) - np.interp(fj.x0 + fj.down * win, x, u)
    return np.maximum(np.linalg.solve(A, T), 0.0)


def _heave_shift(arr, x, x0, heave, down):
    """Restore heave: hangingwall columns move back toward the footwall by `heave` (closing the fault gap)."""
    s = down * (x - x0); out = arr.copy()
    hw = s >= 0
    src = x[hw] + down * heave
    out[..., hw] = np.array([np.interp(src, x, a) for a in np.atleast_2d(arr)]).reshape(out[..., hw].shape)
    return out


def restore_faulted(sec: Section, faults, Te=1500.0, beta=1.15, rift_end_Ma=155.0, fault_end_Ma=140.0,
                    Hd=15e3, heave=True, rho_fill=2400.0, win=1500.0, block_width=False, rho_m=RHO_M, rho_w=RHO_W):
    """Sequential restoration with stripping + decompaction + flexural unloading + reverse
    post-rift thermal subsidence (Roberts et al. 1998) + flexural-cantilever removal of fault
    slip (Kusznir et al. 1991, 1995).

    Faults are inactive after fault_end_Ma: the relief of horizons younger than or equal to it is kept
    as paleo-bathymetry (sediment-starved fault scarps, as in Roberts et al. 1998 Fig. 4b). For an older
    (syn-/pre-rift) horizon h the slip younger than h is its whole throw, measured on the stripped, decompacted
    and unloaded geometry at that step, i.e. h is assumed to have been deposited without relief across the
    fault (the usual reverse-modelling assumption). Fault.throw only flags which horizons are offset.
    Returns a :class:`Restoration2D` (extras: u_fault[t,x], slip[t,fault], rest).
    """
    from ..thermal import mckenzie_postrift
    rest = backstrip_section(sec, Te, rho_m=rho_m, rho_w=rho_w)
    drho = rho_m - rho_w; x = sec.x
    beta = np.broadcast_to(np.asarray(beta, float), x.shape)
    ages = sec.age_top
    kend = int(np.argmin(abs(ages - fault_end_Ma)))
    SL0 = -rest.sea_surface[0]
    TOP, BASE, SF, U, SLIP = [], [], [], [], []
    for i, t in enumerate(ages):
        tt = min(t, rift_end_Ma)
        p = dict(MCK, rho_m=rho_m, rho_w=rho_w)
        dth = np.array([mckenzie_postrift(np.array([rift_end_Ma]), b, p)[0] - mckenzie_postrift(np.array([rift_end_Ma - tt]), b, p)[0]
                        for b in beta])
        SLt = -rest.sea_surface[i]
        # layer surfaces measured from the unloaded seafloor
        us = rest.unloaded_seafloor[i] - dth + (SLt - SL0) * rho_w / drho
        top = rest.top[i] - rest.seafloor[i] + us; base = rest.base[i] - rest.seafloor[i] + us
        if i > kend and faults:
            # throws of the current top horizon measured on the stripped, decompacted, unloaded geometry
            ztop = top[i]
            Tcur = np.array([max((np.interp(f.x0 + f.down * win, x, ztop) - np.interp(f.x0 - f.down * win, x, ztop)), 0.0)
                             if f.throw[i] > 0 and f.active(i) else 0.0 for f in faults])
            slips = fault_slips(x, faults, Tcur, Te, Hd, rho_fill, rho_m, win, block_width)
        else:
            slips = np.zeros(len(faults))
        Ls = _block_widths(x, faults, Hd, block_width)
        u = np.zeros_like(x)
        for f, sl, Lf in zip(faults, slips, Ls):
            if sl > 0:
                df_, Lf_ = _geom(f, Hd, Lf)
                u = u + cantilever_displacement(x, f.x0, sl, df_, f.down, Te, Hd, rho_fill=rho_fill, rho_m=rho_m, L=Lf_)[0]
        top = top + u; base = base + u; sf = us + u
        if heave:
            for f, sl in zip(faults, slips):
                if sl > 0:
                    e = sl / np.tan(np.radians(_geom(f, Hd)[0]))
                    top = _heave_shift(top, x, f.x0, e, f.down); base = _heave_shift(base, x, f.x0, e, f.down)
                    sf = _heave_shift(sf, x, f.x0, e, f.down)
        SLIP.append(slips)
        TOP.append(top); BASE.append(base); SF.append(sf); U.append(u)
    return Restoration2D(ages=ages, x=x, units=sec.units, top=np.array(TOP), base=np.array(BASE), seafloor=np.array(SF),
                         method="cantilever", extras=dict(u_fault=np.array(U), slip=np.array(SLIP), rest=rest))


def faults_from_offsets(x, tops, ref=-1, min_jump=250.0, win=1500.0, dip=50.0, source=""):
    """Locate faults as steps in a reference horizon (default oldest top) and read the throw of every
    horizon across each step (depth on downthrown side - depth on upthrown side, `win` m either side).
    Dip is NOT observable from a depth section with vertical exaggeration and must be assumed."""
    x = np.asarray(x); z = np.asarray(tops)[ref]; dx = x[1] - x[0]; n = max(int(win / dx), 1)
    dz = z[n:] - z[:-n]                         # change over win
    cand = np.where(abs(dz) > min_jump)[0]
    faults = []
    while len(cand):
        grp = cand[cand < cand[0] + 2 * n]; j = grp[np.argmax(abs(dz[grp]))]
        xc = 0.5 * (x[j] + x[j + n]); down = 1 if dz[j] > 0 else -1
        i0 = max(j - n // 2, 0); i1 = min(j + n + n // 2, len(x) - 1)
        thr = (np.asarray(tops)[:, i1] - np.asarray(tops)[:, i0]) * down
        faults.append(Fault(x0=xc, dip=dip, down=down, throw=np.maximum(thr, 0.0), source=source))
        cand = cand[cand >= cand[0] + 2 * n]
    return faults


def fault_at(x, tops, x0, down=1, win=1500.0, dip=50.0, label="", source=""):
    """Fault at a chosen trace x0 with throws of every horizon read across it
    (downthrown minus upthrown depth, sampled `win` m either side of x0)."""
    x = np.asarray(x); tops = np.asarray(tops)
    zu = np.array([np.interp(x0 - down * win, x, t) for t in tops]); zd = np.array([np.interp(x0 + down * win, x, t) for t in tops])
    return Fault(x0=float(x0), dip=dip, down=down, throw=np.maximum(zd - zu, 0.0), label=label, source=source)
