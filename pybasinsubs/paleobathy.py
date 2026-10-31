# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Paleo-water depth from foraminifera (Lai et al. 2022, Eqs. 1-4).

Eq. 1  planktic percentage   %Ps = 100 P / (P + B - S - O)
       (P planktic, B benthic, S stress-marker benthics, O other excluded counts)
Eq. 2  Hayward et al. (2001) / Hayward & Triggs (2016) logistic transfer
       W = exp[ a + (1/b) ln( %Ps / (100 - %Ps) ) ] - 0.1
       with a = 5.23 +- 0.042 and b = 1.45 +- 0.08.

       NOTE: the paper prints "alpha = 1.45" as the additive term and
       "beta = 5.23" as the divisor. The published W values (Figshare Supp. Data 3,
       BasinSubs Bathymetry_*.csv) are reproduced only with a = 5.23 additive and
       b = 1.45 divisor, i.e. the two constants are swapped in the printed equation.
Eq. 3  Hohenegger (2005) presence/absence estimate from the depth ranges of the
       benthic taxa present:  l_n = sqrt(w_min w_max),  d_n = w_max - w_min
       W = sum(l_n / d_n) / sum(1 / d_n)
Eq. 4  sigma_W = sqrt( sum((l_n - W)^2 / d_n) / sum(1 / d_n) )

Ordinal abundance codes (Chang 1967-1970) -> counts (Supp. Data 3):
       ? 1+-1, R 3+-2, F 7+-3, C 15+-5, A 35+-15, V 75+-25, D (reworked) 0.
"""
from __future__ import annotations
import numpy as np
from .legacy import propagate_first_order

ABUNDANCE = {"?": (1, 1), "R": (3, 2), "F": (7, 3), "C": (15, 5), "A": (35, 15), "V": (75, 25), "D": (0, 0)}
A_HAY, A_HAY_ERR = 5.23, 0.042
B_HAY, B_HAY_ERR = 1.45, 0.08


def abundance_to_count(code):
    """Ordinal abundance letter -> (count, 1-sigma)."""
    return ABUNDANCE[str(code).strip().upper()]


def planktic_percent(P, B, S=0.0, O=0.0):
    """Eq. 1, in percent."""
    return 100.0 * np.asarray(P, float) / (np.asarray(P, float) + np.asarray(B, float) - S - O)


def hayward_depth(pct, a=A_HAY, b=B_HAY):
    """Eq. 2 (constants in the order that reproduces the published values)."""
    pct = np.asarray(pct, float)
    return np.exp(a + np.log(pct / (100 - pct)) / b) - 0.1


def hayward_depth_from_counts(P, Perr, B, Berr, S=0.0, Serr=0.0, a=A_HAY, a_err=A_HAY_ERR, b=B_HAY, b_err=B_HAY_ERR):
    """W and its 1-sigma error (delta method over P, B, S, a, b), plus %Ps and its error."""
    def W(v):
        p, bb, s, aa, bh = v
        f = p / (p + bb - s)
        return np.exp(aa + np.log(f / (1 - f)) / bh) - 0.1

    def pct(v):
        p, bb, s = v
        return p / (p + bb - s)
    arrs = np.broadcast_arrays(*[np.asarray(v, float) for v in (P, Perr, B, Berr, S, Serr)])
    P, Perr, B, Berr, S, Serr = [x.ravel() for x in arrs]
    out = np.array([propagate_first_order(W, [P[i], B[i], S[i], a, b], [Perr[i], Berr[i], Serr[i], a_err, b_err])
                    + propagate_first_order(pct, [P[i], B[i], S[i]], [Perr[i], Berr[i], Serr[i]])
                    for i in range(len(P))], float)
    shp = arrs[0].shape
    return tuple(out[:, j].reshape(shp) for j in range(4))


def hohenegger_depth(w_min, w_max):
    """Eqs. 3-4: weighted geometric-mean depth of the taxa ranges and its sd."""
    w_min = np.asarray(w_min, float); w_max = np.asarray(w_max, float)
    l = np.sqrt(np.maximum(w_min, 1e-9) * w_max)
    d = np.maximum(w_max - w_min, 1e-9)
    W = np.sum(l / d) / np.sum(1 / d)
    s = np.sqrt(np.sum((l - W) ** 2 / d) / np.sum(1 / d))
    return W, s


def unit_mean(ages, W, Werr, unit_age, window=0.2, max_err=4500.0):
    """BasinSubs rule: mean of W within |age - unit_age| <= window (Werr <= max_err);
    standard error = sqrt(sum Werr^2) / n."""
    ages = np.asarray(ages, float); W = np.asarray(W, float); Werr = np.asarray(Werr, float)
    out = []
    for a in np.atleast_1d(unit_age):
        m = (np.abs(ages - a) <= window) & ~np.isnan(ages) & ~np.isnan(W) & (Werr <= max_err)
        out.append((W[m].mean(), np.sqrt(np.sum(Werr[m] ** 2)) / m.sum()) if m.any() else (np.nan, np.nan))
    return np.array(out)
