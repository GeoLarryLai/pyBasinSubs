# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Curved (listric-type) fault traces: dip-decay curve fits, curved Faults, perturbation for Monte Carlo."""
from __future__ import annotations

import numpy as np
from .sticks import faults_from_cutoffs

# ---------------------------------------------------------------- curved (listric-type) fault traces
# dip(z) = dip_inf + (dip0 - dip_inf) exp(-(z - z_r) / lam)   for z >= z_r (dip0 above z_r)
# x(z)   = x_r + down * integral_{z_r}^{z} cot(dip(z')) dz'
# The dip decreases monotonically with depth (steep near the top, flattening toward the detachment), so the
# trace is monotone in depth and concave toward the hangingwall. Fitted by weighted least squares in x.
def curve_dip(z, z_r, dip0, lam, dip_inf=5.0):
    """Dip (deg) of the curved fault model at depth z."""
    z = np.asarray(z, float)
    return dip_inf + (dip0 - dip_inf) * np.exp(-np.maximum(z - z_r, 0.0) / lam)


def curve_x(z, x_r, z_r, dip0, lam, down=1, dip_inf=5.0, z_max=20e3):
    """x of the curved fault model at depths z (any order)."""
    z = np.asarray(z, float)
    lo = min(float(np.min(z)), z_r); hi = max(float(np.max(z)), z_r, lo + 1.0)
    zg = np.linspace(lo, min(hi, z_max), 1500)
    cot = 1.0 / np.tan(np.radians(curve_dip(zg, z_r, dip0, lam, dip_inf)))
    X = np.r_[0.0, np.cumsum(0.5 * (cot[1:] + cot[:-1]) * np.diff(zg))]
    X = X - np.interp(z_r, zg, X)
    return x_r + down * np.interp(z, zg, X)


def prm_x(z, prm):
    """x of a fitted curve (parameter dict) at depths z. With a footwall-crest anchor and ``dip_up``, the part above
    the anchor is the straight up-dip segment at the fitted syn-rift face dip ``dip_up`` (>= dip0)."""
    z = np.asarray(z, float)
    x = curve_x(z, prm["x_r"], prm["z_r"], prm["dip0"], prm["lam"], prm["down"], prm["dip_inf"])
    if prm.get("anchor") is not None and prm.get("dip_up") is not None:
        xa, za = prm["anchor"]
        x = np.where(z < za, xa + prm["down"] * (z - za) / np.tan(np.radians(prm["dip_up"])), x)
    return x


def fit_fault_curve(points, weights, down, dip_inf=5.0, dip0_bounds=(10.0, 70.0), lam_bounds=(500.0, 60e3),
                    lam_prior=(6000.0, 0.5), prior_weight=1.0, anchor=None):
    """Fit the curved fault model to constraint points (x, z) with weights. Returns dict(x_r, z_r, dip0, lam,
    dip_inf, cov (3x3 for x_r, dip0, lam), rms_x, n). z_r = depth of the shallowest point.

    ``lam_prior`` = (lam0, sd of ln lam): a soft log-normal prior on the flattening depth scale that keeps short,
    shallow constraint sets (a few km of fault) from bending the curve into an unphysical kink; its strength is
    ``prior_weight`` x the rms weighted residual of a straight-line fit (so it is in metres of x misfit).

    ``anchor`` = (x_a, z_a): hard constraint, the curve passes exactly through this point (x_r is then solved from
    dip0 and lam, and only those two are fitted; cov keeps the 3x3 layout with a zero x_r row/column)."""
    from scipy.optimize import least_squares
    P = np.asarray(points, float); w = np.sqrt(np.asarray(weights, float) / np.mean(weights))
    z_r = float(P[:, 1].min()) if anchor is None else float(min(P[:, 1].min(), anchor[1]))
    b, a = np.polyfit(P[:, 1], P[:, 0], 1, w=w)
    d_lin = float(np.clip(np.degrees(np.arctan2(1.0, abs(b))), dip0_bounds[0] + 1, dip0_bounds[1] - 1))
    p0 = [a + b * z_r, min(d_lin + 10.0, dip0_bounds[1] - 1), 4000.0]
    lin_rms = float(np.sqrt(np.average((a + b * P[:, 1] - P[:, 0]) ** 2, weights=w ** 2))) if len(P) > 2 else 300.0
    pw = prior_weight * max(lin_rms, 100.0) * np.sqrt(len(P))

    def xr_of(d0, lam):                      # x_r that puts the curve through the anchor
        return anchor[0] - float(curve_x(anchor[1], 0.0, z_r, d0, lam, down, dip_inf))

    def fun(p):
        if anchor is not None:
            p = [xr_of(p[0], p[1]), p[0], p[1]]
        r = (curve_x(P[:, 1], p[0], z_r, p[1], p[2], down, dip_inf) - P[:, 0]) * w
        return np.r_[r, pw * np.log(p[2] / lam_prior[0]) / lam_prior[1]] if lam_prior else r
    lb = [P[:, 0].min() - 20e3, max(dip0_bounds[0], dip_inf + 0.5), lam_bounds[0]]
    ub = [P[:, 0].max() + 20e3, dip0_bounds[1], lam_bounds[1]]
    if anchor is not None:
        r = least_squares(fun, p0[1:], bounds=(lb[1:], ub[1:]), x_scale=[10.0, 3000.0])
        J2 = r.jac[:len(P)]
        r.x = np.r_[xr_of(*r.x), r.x]; r.jac = np.c_[np.zeros((r.jac.shape[0], 1)), r.jac]
    else:
        r = least_squares(fun, p0, bounds=(lb, ub), x_scale=[1000.0, 10.0, 3000.0])
    res = r.fun[:len(P)] / w; dof = max(len(P) - 3, 1)
    s2 = float(np.sum(r.fun[:len(P)] ** 2) / dof)
    try:
        if anchor is not None:
            cov = np.zeros((3, 3)); cov[1:, 1:] = np.linalg.inv(J2.T @ J2) * s2
        else:
            cov = np.linalg.inv(r.jac.T @ r.jac) * s2
    except np.linalg.LinAlgError:
        cov = np.diag([0.0 if anchor is not None else 500.0 ** 2, 5.0 ** 2, 2000.0 ** 2])
    return dict(anchor=None if anchor is None else (float(anchor[0]), float(anchor[1])), x_r=float(r.x[0]), z_r=z_r, dip0=float(r.x[1]), lam=float(r.x[2]), dip_inf=dip_inf, down=int(down),
                cov=cov, rms_x=float(np.sqrt(np.average(res ** 2, weights=w ** 2))), n=len(P))


def _curve_samples(prm, z_top, Hd, n=120):
    zs = np.linspace(z_top, Hd, n)
    if prm.get("anchor") is not None and z_top < prm["anchor"][1] < Hd:     # sample the anchor exactly
        zs = np.unique(np.r_[zs, prm["anchor"][1]])
    return np.c_[prm_x(zs, prm), zs]


def _tip_on_curve(prm, x, zt_h, z_hint, z_min=0.0):
    """Depth where the curve meets the tip horizon zt_h(x), moved down-dip until the curve stays below it."""
    zz = np.linspace(z_min, max(z_hint, z_min + 10.0) + 3000.0, 1200)
    xx = prm_x(zz, prm)
    above = zz < np.interp(xx, x, zt_h) - 1.0
    if not above.any():
        return float(zz[0])
    k = int(np.where(above)[0].max()) + 1
    return float(zz[min(k, len(zz) - 1)])


def _finish_curved(g, prm, x, tops, tip_unit, Hd, win, ref=-1, seg=None):
    """Set curve, tip, trace x0, throws and effective planar dip of a curved Fault g in place."""
    tops = np.asarray(tops, float)
    zt = _tip_on_curve(prm, x, tops[tip_unit], prm["z_r"]) if tip_unit is not None else prm["z_r"]
    g.curve = _curve_samples(prm, zt, Hd); g.curve_params = dict(prm)
    zr = tops[ref]; x0 = float(g.curve[0, 0])
    if prm.get("anchor") is not None:                          # trace = the anchor (footwall crest of the oldest top)
        x0 = float(prm["anchor"][0])
    else:
        for _ in range(40):                                    # trace on the reference (oldest) top
            x0 = float(np.clip(g.x_at(np.interp(x0, x, zr)), x[0], x[-1]))
    g.x0 = x0
    zu = tops[:, :] ; thr = np.array([np.interp(x0 + g.down * win, x, t) - np.interp(x0 - g.down * win, x, t) for t in zu])
    g.throw = np.maximum(thr, 0.0)
    if tip_unit is not None:
        g.tip_unit = int(tip_unit); g.z_tip = float(zt)
        g.throw = np.where(np.arange(len(g.throw)) <= tip_unit, 0.0, g.throw)
    # block-bounding segment for the domino method: upper tip -> deepest hangingwall cutoff (base of the pre-rift
    # block), else the trace on the oldest top
    z1, z2 = (zt, float(prm.get("z_deep", np.interp(x0, x, zr)))) if seg is None else seg
    if z2 - z1 < 200.0:
        z2 = z1 + 1500.0
    g.dip = float(np.degrees(np.arctan2(z2 - z1, abs(float(g.x_at(z2)) - float(g.x_at(z1))))))
    g.stick = g.curve[:: max(len(g.curve) // 30, 1)]
    return g


def curved_fault_from_stick(f, x, tops, Hd=10e3, tip_unit=None, win=1500.0, ref=-1):
    """User-supplied (imported / traced) curved stick passed through unchanged: Fault.curve = the stick (sorted by
    depth), extended below its deepest point at its last segment's dip down to Hd; tip = shallowest point."""
    import copy
    g = copy.deepcopy(f); st = np.asarray(f.stick, float); st = st[np.argsort(st[:, 1])]
    if st[-1, 1] < Hd and len(st) >= 2:
        g.curve = st
        st = np.r_[st, [[float(g.x_at(Hd)), Hd]]]
    g.curve = st; g.curve_params = None
    if g.z_tip is None:
        g.z_tip = float(st[0, 1])
    if tip_unit is not None:
        g.tip_unit = int(tip_unit)
        g.throw = np.where(np.arange(len(g.throw)) <= tip_unit, 0.0, g.throw)
    zr = np.asarray(tops, float)[ref]; z1 = float(st[0, 1]); z2 = float(np.interp(g.x0, x, zr))
    if z2 - z1 > 200:
        g.dip = float(np.degrees(np.arctan2(z2 - z1, abs(float(g.x_at(z2)) - float(g.x_at(z1))))))
    return g


def faults_from_curves(faults, x, tops, base=None, tip_unit=None, Hd=10e3, win=1500.0, face_weight=1.0,
                       dip_inf=8.0, keep_user_sticks=True, ref=-1, curve_kw=None, trace_anchor="midramp", **cut_kw):
    """Curved fault traces fitted through ALL constraints, ordered by depth (see module notes above):

    1. the fault-face midpoints of the syn-rift horizons below the tip (``Fault.stick`` of auto-detected faults:
       the steep shallow part),
    2. the footwall and hangingwall cutoffs of the syn-rift and pre-rift tops and of ``base``
       (``faults_from_cutoffs``: the yellow block edges at depth),
    3. the model dip(z) = dip_inf + (dip0 - dip_inf) exp(-(z - z_r)/lam), which flattens toward the detachment Hd.

    Weights: cutoffs by their throw; face midpoints by ``face_weight`` x the median cutoff throw. The curve starts at
    the upper tip (top of ``tip_unit``) and ends at Hd. Imported / traced sticks (source starting 'imported' or
    'traced') pass through unchanged when ``keep_user_sticks``. Fault.dip becomes the effective planar dip of the
    block-bounding segment (tip to the deepest hangingwall cutoff), used by the domino method.

    trace_anchor: 'midramp' (default) = no anchor, x0 where the free curve meets the oldest top (in the middle of
    the drawn fault ramp); 'footwall_crest' = hard constraint, the curve passes exactly through the footwall-crest
    cutoff of the oldest top (``ref``, top Triassic on VG1), which is also the trace x0, with the syn-rift face dip
    above it."""
    if trace_anchor not in ("footwall_crest", "midramp"):
        raise ValueError("trace_anchor must be 'footwall_crest' or 'midramp'")
    x = np.asarray(x, float); tops = np.asarray(tops, float)
    fs = sorted(faults, key=lambda f: f.x0)
    user = [keep_user_sticks and f.stick is not None and str(f.source).startswith(("imported", "traced")) for f in fs]
    cut = faults_from_cutoffs(fs, x, tops, base, tip_unit=tip_unit, win=win, ref=ref, **cut_kw)
    out = []
    for f, g, u in zip(fs, cut, user):
        if u:
            out.append(curved_fault_from_stick(f, x, tops, Hd, tip_unit, win, ref)); continue
        if g.cutoffs is None:
            out.append(g); continue
        tab = g.cutoffs["table"]
        # window-picked cutoffs straddle the fault (footwall left of it, hangingwall right of it), so each horizon
        # constrains the fault at the midpoint of its fw / hw cutoff pair; the deepest hangingwall cutoff (base of
        # the pre-rift, the bottom corner of the yellow block) is used as it is: the fault at depth
        fw = tab[["x_fw", "z_fw"]].to_numpy(float); hw = tab[["x_hw", "z_hw"]].to_numpy(float)
        thr = tab.throw.to_numpy(float)
        P = [0.5 * (fw + hw)]; W = [thr.copy()]
        kb = int(np.argmax(hw[:, 1]))
        P.append(hw[kb:kb + 1]); W.append(thr[kb:kb + 1])
        face = np.zeros((0, 2))
        if f.stick is not None:
            st = np.asarray(f.stick, float)
            below = st[:, 1] > np.interp(st[:, 0], x, tops[tip_unit]) + 20.0 if tip_unit is not None else np.ones(len(st), bool)
            face = st[below]
            if len(face):
                P.append(face); W.append(np.full(len(face), face_weight * float(np.median(tab.throw))))
        P = np.vstack(P); W = np.concatenate(W)
        anc = None
        if trace_anchor == "footwall_crest":
            hn = f"top{len(tops) - 1 if ref == -1 else ref}"
            row = tab[tab.horizon == hn]
            if len(row):
                anc = (float(row.x_fw.iloc[0]), float(row.z_fw.iloc[0]))
        if anc is None:
            prm = fit_fault_curve(P, W, f.down, dip_inf=dip_inf, **(curve_kw or {}))
        else:
            # below the anchor: depth-flattening curve through the crest, fitted to the deeper constraints (yellow
            # block edges); above it: straight segment at the syn-rift face dip up to the tip (140 Ma)
            deep = P[:, 1] > anc[1] + 1.0
            if deep.sum() < 2:
                deep = np.ones(len(P), bool)
            prm = fit_fault_curve(P[deep], W[deep], f.down, dip_inf=dip_inf, anchor=anc, **(curve_kw or {}))
            up = face[face[:, 1] < anc[1] - 1.0] if len(face) else face
            if len(up) >= 1:                   # face dip through the anchor (least squares in x)
                dz = up[:, 1] - anc[1]; dx = f.down * (up[:, 0] - anc[0])
                c = float(np.sum(dz * dx) / np.sum(dz * dz)) if np.sum(dz * dz) > 0 else np.inf
                d_up = float(np.degrees(np.arctan2(1.0, c))) if c > 0 else f.dip
            else:
                d_up = f.dip                   # face-midpoint plane dip of the detected stick
            prm["dip_up"] = float(np.clip(d_up, prm["dip0"], 75.0))
        prm["points"] = P; prm["weights"] = W; prm["n_face"] = len(face)
        zbase_hw = float(tab.z_hw.max())
        prm["z_deep"] = zbase_hw
        h = _finish_curved(g, prm, x, tops, tip_unit, Hd, win, ref)
        h.source = (f.source + "; " if f.source else "") + "curve fitted to face midpoints + block cutoffs" + \
            ("; anchored at the footwall crest of the oldest top" if anc else "")
        h.cutoffs = dict(g.cutoffs, curve_rms_x=prm["rms_x"], z_deep=zbase_hw)
        out.append(h)
    return out


def perturb_curved_faults(faults, rng, x, tops, Hd=10e3, scale=1.0, win=1500.0):
    """One random draw of the fitted curves (x_r, dip0, lam from their fit covariance x scale); other faults are
    returned unchanged. Used by flex2d.monte_carlo_section(vary_faults=True)."""
    import copy
    out = []
    for f in faults:
        p = getattr(f, "curve_params", None)
        if not p or "cov" not in p:
            out.append(f); continue
        mu = np.array([p["x_r"], p["dip0"], p["lam"]])
        for _try in range(25):            # anchored draws whose curve would cut the tip horizon below the crest: redraw
            d = rng.multivariate_normal(np.zeros(3), np.asarray(p["cov"]) * scale ** 2)
            q = dict(p); q["x_r"], q["dip0"], q["lam"] = mu + d
            q["dip0"] = float(np.clip(q["dip0"], p["dip_inf"] + 1, 80)); q["lam"] = float(np.clip(q["lam"], 500, 60e3))
            if p.get("dip_up") is not None:
                q["dip_up"] = float(np.clip(p["dip_up"] + d[1], q["dip0"], 80.0))
            if p.get("anchor") is not None:          # anchored curves keep passing through the footwall crest
                xa, za = p["anchor"]
                q["x_r"] = xa - float(curve_x(za, 0.0, q["z_r"], q["dip0"], q["lam"], q["down"], q["dip_inf"]))
            g = _finish_curved(copy.deepcopy(f), q, x, tops, f.tip_unit, Hd, win)
            if q.get("anchor") is None or g.z_tip <= q["anchor"][1] + 1.0:
                break
        else:
            g = copy.deepcopy(f)
        out.append(g)
    return out


def curve_with_dip(f, dip, x, tops, Hd=10e3, win=1500.0):
    """Copy of a curved fault whose effective planar dip is changed to ``dip`` by shifting dip0 (shape kept,
    passing through the same shallowest constraint point). Planar faults: Fault.dip replaced."""
    import copy
    import dataclasses
    p = getattr(f, "curve_params", None)
    if not p:
        return dataclasses.replace(f, dip=float(dip))
    q = dict(p); g = copy.deepcopy(f)
    for _ in range(6):                                     # secant on dip0 to hit the requested effective dip
        q["dip0"] = float(np.clip(q["dip0"] + (dip - g.dip), p["dip_inf"] + 1, 85))
        g = _finish_curved(copy.deepcopy(f), q, x, tops, f.tip_unit, Hd, win)
        if abs(g.dip - dip) < 0.2:
            break
    return g


def curve_shift_dip(f, dip):
    """Copy of a fault with effective dip ``dip``. Curved faults (fitted, with curve_params): dip0 is shifted by the
    change and the curve regenerated through the same upper tip point, so the trace keeps its shape (steep top,
    flattening with depth); trace x0 and throws are kept. Plain faults: Fault.dip replaced."""
    import copy
    import dataclasses
    p = getattr(f, "curve_params", None)
    if f.curve is None or not p:
        return dataclasses.replace(f, dip=float(dip))
    g = copy.deepcopy(f); q = dict(p)
    q["dip0"] = float(np.clip(p["dip0"] + (dip - f.dip), p["dip_inf"] + 1.0, 85.0))
    zt, xt = float(f.curve[0, 1]), float(f.curve[0, 0])
    if p.get("anchor") is not None:              # rotate about the footwall-crest anchor (stays the trace)
        xt, zt = p["anchor"]
    q["x_r"] = 0.0
    q["x_r"] = xt - float(curve_x(zt, 0.0, q["z_r"], q["dip0"], q["lam"], q["down"], q["dip_inf"]))
    zs = f.curve[:, 1]
    if p.get("anchor") is not None:
        zs = np.unique(np.r_[zs, p["anchor"][1]])
    if q.get("dip_up") is not None:
        q["dip_up"] = float(np.clip(p["dip_up"] + (dip - f.dip), q["dip0"], 80.0))
    g.curve = np.c_[prm_x(zs, q), zs]
    g.curve_params = q; g.dip = float(dip); g.stick = g.curve[:: max(len(g.curve) // 30, 1)]
    return g
