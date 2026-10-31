# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fit porosity-depth curves phi(z) = phi0 exp(-z/C) (paper Eq. 5).

* :func:`fit_exponential` - least squares on ln(phi) (as BasinSubs) or nonlinear
  least squares on phi, with a *correct* 95 % confidence band of the mean curve:
  se(z) = sqrt(mse (1/n + (z - zbar)^2 / Sxx)).
  (The R script uses sqrt(mse * sqrt(...)), which makes the band too wide; the
  verbatim version is kept in :func:`pybasinsubs.legacy.porosity_fit_r`.)
* :func:`sample_curves` - sample published curves every ``step`` m (BasinSubs method).
"""
from __future__ import annotations
import numpy as np
from scipy import stats, optimize


def sample_curves(table, step=50.0):


    """Sample each published curve phi0*exp(-z/c) every ``step`` m over [min_z, max_z]; returns (z, phi) arrays."""
    zs, ps = [], []
    for _, r in table.iterrows():
        z = np.arange(r.min_z, r.max_z + 1e-9, step)
        zs.append(z); ps.append(r.InitialPorosity * np.exp(-z / r.Decay_c))
    return np.concatenate(zs), np.concatenate(ps)


def fit_exponential(z, phi, method="loglinear", zgrid=None, level=0.95):
    """Return dict(phi0, phi0_se, C, C_se, r2, zgrid, mean, lo, hi)."""
    z = np.asarray(z, float); phi = np.asarray(phi, float)
    n = len(z)
    zgrid = np.linspace(0, max(6000, z.max()), 301) if zgrid is None else zgrid
    t = stats.t.ppf(0.5 + level / 2, n - 2)
    if method == "loglinear":
        y = np.log(phi)
        X = np.column_stack([np.ones(n), z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        mse = res @ res / (n - 2)
        cov = mse * np.linalg.inv(X.T @ X)
        Xg = np.column_stack([np.ones(len(zgrid)), zgrid])
        mean = Xg @ beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov, Xg))
        phi0 = np.exp(beta[0]); C = -1 / beta[1]
        se_a, se_b = np.sqrt(np.diag(cov))
        r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
        return dict(phi0=phi0, phi0_se=phi0 * se_a, C=C, C_se=C * abs(se_b / beta[1]), r2=r2,
                    zgrid=zgrid, mean=np.exp(mean), lo=np.exp(mean - t * se), hi=np.exp(mean + t * se))
    f = lambda zz, p0, c: p0 * np.exp(-zz / c)
    p, cov = optimize.curve_fit(f, z, phi, p0=[0.5, 2000.0])
    J = np.column_stack([np.exp(-zgrid / p[1]), p[0] * zgrid / p[1] ** 2 * np.exp(-zgrid / p[1])])
    se = np.sqrt(np.einsum("ij,jk,ik->i", J, cov, J))
    mean = f(zgrid, *p)
    r2 = 1 - np.sum((phi - f(z, *p)) ** 2) / np.sum((phi - phi.mean()) ** 2)
    return dict(phi0=p[0], phi0_se=np.sqrt(cov[0, 0]), C=p[1], C_se=np.sqrt(cov[1, 1]), r2=r2,
                zgrid=zgrid, mean=mean, lo=mean - t * se, hi=mean + t * se)
