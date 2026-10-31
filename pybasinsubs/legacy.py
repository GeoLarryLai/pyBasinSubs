# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Line-by-line Python port of the BasinSubs R scripts (Lai, 2022).

This layer reproduces the published R outputs in ``BasinSubs/output results``
*exactly*, including the R scripts' quirks (documented below). Use the generic
API (:mod:`pybasinsubs.backstrip`) for new work.

Sources ported
--------------
* ``Porosity-depth function.R``  -> :func:`porosity_fit_r`
* ``Age PB dSL-N.R`` / ``-S.R``  -> :func:`age_model_r`, :func:`pb_age_r`,
  :func:`pb_unit_r`, :func:`sl_unit_r`
* ``Subsidence Analysis N.R`` / ``S.R`` -> :func:`subsidence_analysis_r`

Error propagation follows R ``propagate(type="stat", do.sim=FALSE)``:
first-order (delta-method) mean = f(means), sd = sqrt(J diag(s^2) J^T).
Derivatives are taken by the complex-step method (exact to machine precision).

Known quirks reproduced on purpose (flagged for the author)
----------------------------------------------------------
1. ``FL = Litho[1, k+1]``: the R loop uses the lithology fractions of the
   *first row* (unit 0: 50 % mudstone / 50 % sandstone) for **every** unit.
   The per-unit fractions in ``Litho_N/S.csv`` are therefore never used.
2. Decompacted-thickness error is seeded as |T_i - T0| + 2*T0_err before the
   delta method (an ad-hoc inflation, as commented in the R code).
3. Errors of lithology-weighted sums are added linearly (sum of F*sd).
4. ``W0`` for the eroded unit is ``MeanPB[3]/2`` (unit 2) in the N script but
   ``MeanPB[2]/2`` (unit 1) in the S script; the paper states W0 = W1/2.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

from ._paths import data_path

from .constants import RHO_W, RHO_W_ERR, RHO_M as RHO_A, RHO_M_ERR as RHO_A_ERR


# ----------------------------------------------------------------- delta method
def propagate_first_order(f, means, sds):
    """First-order (delta method) propagation, as R ``propagate`` 'Mean.1'/'sd.1'.

    f : callable taking a 1-D array of parameters (complex-safe)
    Returns (f(means), sqrt(sum((df/dx_i * sd_i)^2))).
    """
    x = np.asarray(means, dtype=complex)
    s = np.asarray(sds, dtype=float)
    val = f(x.copy()).real
    h = 1e-30
    grad = np.empty(len(x))
    for i in range(len(x)):
        xi = x.copy()
        xi[i] += 1j * h
        grad[i] = f(xi).imag / h
    return float(val), float(np.sqrt(np.sum((grad * s) ** 2)))


# ------------------------------------------------------------- R functions
def porosity_depth(depth, porosity_0, decayC):
    """R ``porosity_depth``: phi(z) = phi0 * exp(-z/C)  (paper Eq. 5)."""
    return porosity_0 * np.exp(-np.asarray(depth) / decayC)


def decompaction(D0, D0_err, Di, Di_err, T0, T0_err, p0, p0_err, C, C_err):
    """R ``decompaction``: fixed-point solve of paper Eq. 7 for one lithology.

    Returns (Ti, Ti_se) from the delta method, exactly as the R function.
    """
    K1 = -C * p0 * np.exp(-(Di / C))
    K2 = (C * p0 * np.exp(-(D0 / C)) * (np.exp(-(T0 / C)) - 1)) + T0 - K1
    dT = T0
    while True:
        test = K1 * np.exp(-(dT / C)) + K2
        if abs(test - dT) < 1e-6 or np.isnan(test):
            break
        dT = test
    dT_err = abs(dT - T0) + 2 * T0_err

    def expr(v):
        oD, dD, oT, dTv, P, Cc = v
        return ((-1) * Cc * P * np.exp(-(dD / Cc))) * np.exp(-(dTv / Cc)) + (
            (Cc * P * np.exp(-(oD / Cc)) * (np.exp(-(oT / Cc)) - 1)) + oT
            - (-1) * Cc * P * np.exp(-(dD / Cc)))

    return propagate_first_order(expr, [D0, Di, T0, dT, p0, C],
                                 [D0_err, Di_err, T0_err, dT_err, p0_err, C_err])


def mass(Di, Di_err, Ti, Ti_err, rho_w, rho_w_err, rho_g, rho_g_err, p0, p0_err, C, C_err):
    """R ``Mass``: 1-D mass of a unit (paper Eq. 12), delta-method error."""
    def expr(v):
        dD, dT, rW, rG, P, Cc = v
        return rG * dT + (rW - rG) * Cc * P * np.exp(-(dD / Cc)) * (1 - np.exp(-(dT / Cc)))
    return propagate_first_order(expr, [Di, Ti, rho_w, rho_g, p0, C],
                                 [Di_err, Ti_err, rho_w_err, rho_g_err, p0_err, C_err])


def tect_z(S, S_err, W, W_err, dSL, dSL_err, rho_w, rho_w_err, rho_a, rho_a_err, rho_s, rho_s_err):
    """R ``TectZ``: Airy tectonic subsidence (paper Eq. 10), delta-method error."""
    def expr(v):
        s, w, sl, rW, rA, rS = v
        return s * ((rA - rS) / (rA - rW)) + w - sl * (rA / (rA - rW))
    return propagate_first_order(expr, [S, W, dSL, rho_w, rho_a, rho_s],
                                 [S_err, W_err, dSL_err, rho_w_err, rho_a_err, rho_s_err])


def depth_r(T, T_err):
    """R ``Depth``: cumulative depth of each unit top and its error."""
    n = len(T)
    D = np.zeros(n); De = np.zeros(n)
    for i in range(1, n):
        D[i] = D[i - 1] + T[i - 1]
        De[i] = np.sqrt(T_err[i - 1] ** 2 + De[i - 1] ** 2)
    return D, De


# ------------------------------------------------------------ main analysis
MAX_ELEV = {"N": 1334.0, "S": 1682.0}


def load_inputs_r(region="N", root=None):


    """Read the BasinSubs input CSVs (thickness, ages, lithology, density, porosity, sea level) for region N or S, as the R script does."""
    root = root or data_path("coastal_range")
    j = lambda *p: os.path.join(root, *p)
    return dict(
        PoFun=pd.read_csv(j("Backstripping", "PoFun.csv")),
        SL=pd.read_csv(j("Sealevel", f"SL_Unit_{region}.csv")),
        Strat=pd.read_csv(j("Thickness", f"Ages_{region}_new.csv")),
        Litho=pd.read_csv(j("Thickness", f"Litho_{region}.csv"), encoding="utf-8-sig"),
        PB=pd.read_csv(j("Thickness", f"PB_Unit_{region}.csv")),
    )


def subsidence_analysis_r(region="N", root=None, inputs=None):
    """Port of ``Subsidence Analysis N.R``/``S.R``.

    Returns (results_df, rates_df) with the same columns as the R CSV outputs.
    """
    d = inputs or load_inputs_r(region, root)
    PoFun, Strat, Litho = d["PoFun"], d["Strat"], d["Litho"]
    T0 = Strat.oT.to_numpy(float); T0e = Strat.T0_err.to_numpy(float)
    age = Strat.Age.to_numpy(float); agee = Strat.Age_err.to_numpy(float)
    unit = Strat.Units.to_numpy()
    rg, rge = PoFun.rho_g.to_numpy(float), PoFun.rho_g_err.to_numpy(float)
    p0, p0e = PoFun.Porosity_0.to_numpy(float), PoFun.Porosity_0_err.to_numpy(float)
    C, Ce = PoFun.DecayC.to_numpy(float), PoFun.DecayC_err.to_numpy(float)
    dSL, dSLe = d["SL"].dSL.to_numpy(float), d["SL"].dSLerr.to_numpy(float)
    W, We = d["PB"].W.to_numpy(float), d["PB"].Werr.to_numpy(float)
    nlith = Litho.shape[1] - 1
    FLrow = Litho.iloc[0, 1:].to_numpy(float)       # quirk 1: first row only
    N = len(unit)
    D0, D0e = depth_r(T0, T0e)
    maxD0 = D0.max()
    Su = np.array([maxD0 - T0[:i].sum() + W[i] - dSL[i] for i in range(N)])
    Su_e = np.array([np.sqrt(np.sum(T0e[i:] ** 2) + We[i] ** 2 + dSLe[i] ** 2) for i in range(N)])

    St, Ste, Si, Sie, rho, rhoe = [], [], [], [], [], []
    for i in range(N):
        Di = np.full(N, np.nan); Die = np.full(N, np.nan)
        Di[i] = 0.0; Die[i] = 0.0
        Ti = np.full(N, np.nan); Tie = np.full(N, np.nan)
        Mi = np.full(N, np.nan); Mie = np.full(N, np.nan)
        for j in range(i, N):
            tk, tke = [], []
            for k in range(nlith):
                FL = FLrow[k]
                if k >= len(p0) or FL == 0:      # NA lithology (e.g. Ls) or zero weight
                    continue
                t, te = decompaction(D0[j], D0e[j], Di[j], Die[j], T0[j], T0e[j], p0[k], p0e[k], C[k], Ce[k])
                tk.append(t * FL); tke.append(te * FL)
            Ti[j] = np.nansum(tk)
            Tie[j] = T0e[j] if i == 0 else np.nansum(tke)
            if j <= N - 2:
                Di[j + 1] = Di[j] + Ti[j]
                Die[j + 1] = np.sqrt(Die[j] ** 2 + Tie[j] ** 2)
            mk, mke = [], []
            for k in range(nlith):
                FL = FLrow[k]
                if k >= len(p0) or FL == 0:
                    continue
                m, me = mass(Di[j], Die[j], Ti[j], Tie[j], RHO_W, RHO_W_ERR, rg[k], rge[k], p0[k], p0e[k], C[k], Ce[k])
                mk.append(FL * m); mke.append(FL * me)
            Mi[j] = np.sum(mk); Mie[j] = np.sum(mke)
        mD, mDe = np.nanmax(Di), np.nanmax(Die)
        St.append(mD); Ste.append(mDe)
        Si.append(mD + W[i] - dSL[i]); Sie.append(np.sqrt(mDe ** 2 + We[i] ** 2 + dSLe[i] ** 2))
        if i == N - 1:
            rho.append(0.0); rhoe.append(0.0)
        else:
            sM = np.nansum(Mi)
            r = sM / mD
            rho.append(r)
            rhoe.append(r * np.sqrt((np.sqrt(np.nansum(Mie ** 2)) / sM) ** 2 + (mDe / mD) ** 2))
    Z, Ze = zip(*[tect_z(St[i], Ste[i], W[i], We[i], dSL[i], dSLe[i], RHO_W, RHO_W_ERR,
                         RHO_A, RHO_A_ERR, rho[i], rhoe[i]) for i in range(N)])
    res = pd.DataFrame(dict(Unit=unit, Age=age, Age_err=agee, T0=T0, T0_err=T0e, PB=W, PB_err=We,
                            dSL=dSL, dSLerr=dSLe, undecompactedTotalS=Su, undecompactedTotalS_err=Su_e,
                            TotalS=Si, TotalS_err=Sie, TectS=Z, TectS_err=Ze, RhoS=rho, RhoS_err=rhoe))
    # rates (vertical rates_N/S.csv)
    Si = np.array(Si); Sie = np.array(Sie)
    ME = MAX_ELEV[region] / 2; MEe = ME
    MaxU = ((Si[0] + ME) / age[0]) / 1000
    MaxUe = MaxU * np.sqrt((np.sqrt(Sie[0] ** 2 + MEe ** 2) / (Si[0] + ME)) ** 2 + (agee[0] / age[0]) ** 2)
    MinU = ((Si[1] + ME) / age[1]) / 1000
    MinUe = MinU * np.sqrt((np.sqrt(Sie[1] ** 2 + MEe ** 2) / (Si[1] + ME)) ** 2 + (agee[1] / age[1]) ** 2)
    a, b = 1, 14
    Sr = ((Si[a] - Si[b]) / abs(age[a] - age[b])) / 1000
    Sre = Sr * np.sqrt((np.sqrt(Sie[a] ** 2 + Sie[b] ** 2) / (Si[a] - Si[b])) ** 2
                       + (np.sqrt(agee[a] ** 2 + agee[b] ** 2) / (age[a] - age[b])) ** 2)
    rates = pd.DataFrame(dict(SubRate=[Sr], SubRate_err=[Sre], Elev=[ME], Elev_err=[MEe],
                              MinUpRate=[MinU], MinUpRate_err=[MinUe], EstUpRate=[MaxU], EstUpRate_err=[MaxUe]))
    return res, rates


# ----------------------------------------------------- porosity-depth fit
def porosity_fit_r(table, step=50.0):
    """Port of ``Porosity-depth function.R``.

    Samples each published curve phi0*exp(-z/C) on [min_z, max_z] every ``step`` m,
    fits lm(log(phi) ~ z) and returns dict with phi0, phi0_se, C, C_se, z, phi,
    fitted, ci_lower, ci_upper (the R script's CI formula, kept verbatim).
    """
    zs, ps = [], []
    for _, r in table.iterrows():
        z = np.arange(r.min_z, r.max_z + 1e-9, step)
        zs.append(z); ps.append(r.InitialPorosity * np.exp(-z / r.Decay_c))
    z = np.concatenate(zs); phi = np.concatenate(ps)
    y = np.log(phi)
    n = len(z)
    X = np.column_stack([np.ones(n), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    sse = np.sum((y - fit) ** 2); mse = sse / (n - 2)
    cov = mse * np.linalg.inv(X.T @ X)
    se_a, se_b = np.sqrt(np.diag(cov))
    phi0 = np.exp(beta[0]); Cfit = -1 / beta[1]
    t = stats.t.ppf(0.975, n - 2)
    se = np.sqrt(mse * np.sqrt(1 / n + (z - z.mean()) ** 2 / np.sum((z - z.mean()) ** 2)))  # verbatim R
    return dict(phi0=phi0, phi0_se=phi0 * se_a, C=Cfit, C_se=Cfit * np.sqrt((se_b / beta[1]) ** 2),
                z=z, phi=phi, fitted=np.exp(fit), ci_lower=np.exp(fit - t * se),
                ci_upper=np.exp(fit + t * se), r_squared=1 - sse / np.sum((y - y.mean()) ** 2))


# ------------------------------------------------------------ age models
AGE_SEGMENTS = {"N": [("upper ages", 0, 7), ("upper ages 2", 6, 8), ("lower ages 2", 7, 11), ("lower ages", 10, 12)],
                "S": [("upper ages", 0, 8), ("lower ages", 7, 12)]}


class _LM:
    """Ordinary least squares y = a + b x with R-style summaries and predict(se.fit)."""
    def __init__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        self.n = n = len(x)
        X = np.column_stack([np.ones(n), x])
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ self.beta
        df = n - 2
        self.XtXi = np.linalg.inv(X.T @ X)
        if df > 0:
            self.s2 = res @ res / df
            se = np.sqrt(np.diag(self.s2 * self.XtXi))
            tval = self.beta[1] / se[1]
            self.p_slope = 2 * stats.t.sf(abs(tval), df)
        else:
            self.s2 = np.nan; se = np.array([np.nan, np.nan]); self.p_slope = np.nan
        self.se = se
        ss = np.sum((y - y.mean()) ** 2)
        self.r2 = 1 - (res @ res) / ss if ss > 0 else np.nan
        if df == 0:
            self.r2 = 1.0

    @property
    def intercept(self): return self.beta[0]
    @property
    def slope(self): return self.beta[1]

    def predict(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        X = np.column_stack([np.ones(len(x)), x])
        fit = X @ self.beta
        sef = np.sqrt(np.einsum("ij,jk,ik->i", X, self.XtXi, X) * self.s2)
        return fit, sef


def age_model_r(region="N", root=None):
    """Port of the age-model part of ``Age PB dSL-*.R``.

    Returns dict(ages_new=DataFrame like Ages_*_new.csv, regressions (Age~H),
    plot_regressions (H~Age), models=list of (name, lm_age_on_h)).
    """
    root = root or data_path("coastal_range")
    S = pd.read_csv(os.path.join(root, "Thickness", f"Ages_{region}.csv"), encoding="utf-8-sig")
    age_all = S.age.to_numpy(float).copy(); agee_all = S.age_err.to_numpy(float).copy()
    top = S.mean_T.to_numpy(float); tope = S.mean_T_err.to_numpy(float)
    a = age_all[2:14]; t = top[2:14]
    segs = AGE_SEGMENTS[region]
    m_at = [(nm, _LM(t[i0:i1], a[i0:i1])) for nm, i0, i1 in segs]
    m_ta = [(nm, _LM(a[i0:i1], t[i0:i1])) for nm, i0, i1 in segs]
    upper = m_at[0][1]; lower = m_at[-1][1]
    f, se = upper.predict(top[0:2])
    age_all[0:2] = f; agee_all[0:2] = se
    f, se = lower.predict(top[14:15])
    age_all[14] = f[0]
    agee_all[14] = age_all[14] * 0.05 if region == "N" else se[0]
    T0 = np.r_[top[0:14] - top[1:15], 0.0]
    T0e = np.r_[np.sqrt(tope[0:14] ** 2 + tope[1:15] ** 2), tope[14]]
    ages_new = pd.DataFrame(dict(Units=S.unit.to_numpy()[:15], Age=age_all[:15], Age_err=agee_all[:15],
                                 oT=T0, T0_err=T0e))

    def inter(m1, m2):
        return abs((m2.intercept - m1.intercept) / (m2.slope - m1.slope))
    ms = [m for _, m in m_at]; ps = [m for _, m in m_ta]
    ints = [inter(ms[k], ms[k + 1]) for k in range(len(ms) - 1)]
    pints = [inter(ps[k], ps[k + 1]) for k in range(len(ps) - 1)]

    def table(models, lower_col, upper_col):
        return pd.DataFrame(dict(Model=[n for n, _ in models], slope=[m.slope for _, m in models],
                                 slope_err=[m.se[1] for _, m in models], intercept=[m.intercept for _, m in models],
                                 intercept_err=[m.se[0] for _, m in models], r_squared=[m.r2 for _, m in models],
                                 p_slope=[m.p_slope for _, m in models], **{lower_col[0]: lower_col[1], upper_col[0]: upper_col[1]}))
    reg = table(m_at, ("lowerupper", ints + [0.0]), ("upperbound", [top.max()] + ints))
    pname = "lowerupper" if region == "N" else "plotupper"
    preg = table(m_ta, (pname, pints + [np.nanmax(age_all[:15]) if region == "N" else np.nanmax(age_all)]),
                 ("upperbound", [0.0] + pints))
    return dict(ages_new=ages_new, regressions=reg, plot_regressions=preg, models=m_at, intercepts=ints)


def pb_age_r(region="N", root=None, am=None, sort=True):
    """Assign ages (and errors) to paleobathymetry samples (``PB_Age_*.csv``)."""
    root = root or data_path("coastal_range")
    am = am or age_model_r(region, root)
    P = pd.read_csv(os.path.join(root, "Thickness", f"Bathymetry_{region}.csv"), encoding="utf-8-sig")
    H = P.StratH.to_numpy(float)
    ms = [m for _, m in am["models"]]; I = am["intercepts"]
    ages = np.full(len(H), np.nan); agee = np.full(len(H), np.nan)
    for i, h in enumerate(H):
        if np.isnan(h):
            continue
        if region == "N":
            if h >= I[0]: k, pct = 0, False
            elif I[1] < h < I[0]: k, pct = 1, True
            elif I[2] <= h <= I[1]: k, pct = 2, False
            elif h < I[2]: k, pct = 3, True
            else: continue
        else:
            k, pct = (0, False) if h >= I[0] else (1, False)
        f, se = ms[k].predict([h])
        ages[i] = f[0]; agee[i] = f[0] * 0.05 if pct else se[0]
    out = pd.DataFrame(dict(Age=ages, Age_err=agee, W=P.W.to_numpy(float), Werr=P.Werr.to_numpy(float),
                            stratH=H, section=P.section, ID=P.id, source=P.Source))
    if sort:   # R writes the table ordered by age (stable for ties)
        out = out.sort_values("Age", kind="stable", na_position="last").reset_index(drop=True)
    return out


def pb_unit_r(region="N", root=None, pb_age=None, ages_new=None):
    """Average paleobathymetry within +-0.2 Myr of each unit-top age (``PB_Unit_*.csv``)."""
    root = root or data_path("coastal_range")
    A = pb_age if pb_age is not None else pb_age_r(region, root)
    U = ages_new if ages_new is not None else pd.read_csv(os.path.join(root, "Thickness", f"Ages_{region}_new.csv"))
    age = U.Age.to_numpy(float)
    W = np.full(len(age), np.nan); We = np.full(len(age), np.nan)
    for i, a in enumerate(age):
        m = (np.abs(A.Age - a) <= 0.2) & A.Age.notna() & A.W.notna() & (A.Werr <= 4500)
        if m.any():
            W[i] = A.W[m].mean(); We[i] = np.sqrt(np.sum(A.Werr[m] ** 2)) / m.sum()
    ref = 2 if region == "N" else 1   # N script uses MeanPB[3], S script MeanPB[2]
    W[0] = W[ref] / 2; We[0] = W[ref] / 2
    return pd.DataFrame(dict(Unit=U.Units.to_numpy(), W=W, Werr=We))


def sl_unit_r(region="N", root=None, ages_new=None):
    """Average Miller et al. (2020) sea level within age +- (err + 0.01) Myr (``SL_Unit_*.csv``)."""
    root = root or data_path("coastal_range")
    from .sealevel import miller2020_path
    local = os.path.join(root, "Sealevel", "SeaLevel_Miller2020.csv")
    SLc = pd.read_csv(local if os.path.exists(local) else miller2020_path(), encoding="utf-8-sig")
    SL = SLc.iloc[:, 0].to_numpy(float); SA = SLc.iloc[:, 1].to_numpy(float)
    U = ages_new if ages_new is not None else pd.read_csv(os.path.join(root, "Thickness", f"Ages_{region}_new.csv"))
    out_m, out_s = [], []
    for a, e in zip(U.Age, U.Age_err):
        m = (SA >= a - e - 0.01) & (SA <= a + e + 0.01)
        x = SL[m]
        out_m.append(x.mean()); out_s.append(x.std(ddof=1) / np.sqrt(len(x)))
    return pd.DataFrame(dict(Unit=U.Units.to_numpy(), dSL=out_m, dSLerr=out_s))
