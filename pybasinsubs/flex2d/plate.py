# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Thin elastic plate solutions used by the 2-D restorations."""
from __future__ import annotations
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from ..flexure import rigidity, deflection_fft, G
from ..constants import RHO_W, RHO_M


def flexural_parameter(Te, drho):
    """alpha = (4D / (drho g))^(1/4) (m)."""
    return (4 * rigidity(Te) / (drho * G)) ** 0.25


def line_load_analytic(x, V0, Te, drho, broken=False):
    """Turcotte & Schubert (2014, s.3.17) line load V0 (N/m) at x=0.
    continuous: w = V0 a^3/(8D) e^{-|x|/a}(cos|x|/a + sin|x|/a)
    broken (load at the free end): w = V0 a^3/(4D) e^{-x/a} cos(x/a), x>=0."""
    D = rigidity(Te); a = flexural_parameter(Te, drho); s = np.abs(x) / a
    if broken:
        return V0 * a ** 3 / (4 * D) * np.exp(-s) * np.cos(s)
    return V0 * a ** 3 / (8 * D) * np.exp(-s) * (np.cos(s) + np.sin(s))


def _fd_plate(q, dx, D, drho):
    """Finite-element-like FD for (D w'')'' + drho g w = q with free ends (M = V = 0).

    Energy form: minimise  sum_i D_i (w''_i)^2 dx / 2 + sum_i drho g w_i^2 c_i dx / 2 - sum_i q_i w_i c_i dx,
    with w'' evaluated at interior nodes and trapezoid weights c (1/2 at the ends);
    the natural boundary conditions of this functional are zero moment and shear.
    """
    n = len(q); c = np.ones(n); c[0] = c[-1] = 0.5
    L = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format="csr") / dx ** 2
    Dm = np.broadcast_to(D, (n,))[1:-1]
    K = (L.T @ diags(Dm) @ L) * dx + diags(drho * G * c * dx)
    return spsolve(K.tocsc(), q * c * dx)


def plate_deflection(q, dx, Te, drho=RHO_M - RHO_W, method="fft", break_index=None):
    """Deflection (m, + down) of a thin elastic plate under load q (Pa).

    Te = 0 -> local Airy w = q/(drho g). method 'fft' (continuous, constant Te)
    or 'fd' (variable Te allowed). break_index: plate broken at that grid node
    (two independent plates with free ends; forces 'fd').
    """
    q = np.asarray(q, float)
    if np.all(np.asarray(Te) <= 0):
        return q / (drho * G)
    if break_index is not None:
        i = int(break_index); D = rigidity(np.broadcast_to(np.maximum(Te, 1.0), q.shape))
        return np.r_[_fd_plate(q[:i], dx, D[:i], drho), _fd_plate(q[i:], dx, D[i:], drho)]
    if method == "fft" and np.ndim(Te) == 0:
        return deflection_fft(q, dx, Te, drho)
    return _fd_plate(q, dx, rigidity(np.broadcast_to(np.maximum(Te, 1.0), q.shape)), drho)
