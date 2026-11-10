# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Vertical stress, hydrostatic pore pressure, effective stress and K0 horizontal stress.

Vertical (overburden) stress at depth z below the seafloor, with a water column of
thickness W:

    σ_v(z) = ρ_w g W + g ∫_0^z ρ_b(z') dz'                                 (1)

Hydrostatic pore pressure (phase 1; no overpressure) measured from the contemporaneous
sea surface:

    P_p(z) = ρ_w g (W + z)                                                  (2)

Terzaghi effective stress:

    σ' = σ_v - P_p                                                          (3)

Under hydrostatic conditions this reduces to σ'(z) = g ∫_0^z (ρ_b - ρ_w) dz'.

Horizontal stress, two options:

* Jáky (1944) at-rest coefficient, in the widely used simplified form
  K_0 = 1 - sin φ' (φ' = drained friction angle). Jáky's original 1944 expression is
  slightly different; the 1 - sin φ' form is the common geotechnical approximation
  (see Zoback 2007, Reservoir Geomechanics). Then σ_h = K_0 σ' + P_p.
* Uniaxial poroelastic (Zoback 2007): K_0 = ν / (1 - ν) and
  σ_h = K_0 (σ_v - α P_p) + α P_p. With Biot α = 1 this is σ_h' = [ν/(1-ν)] σ'.

Phase 1 does not compute overpressure (disequilibrium compaction, hydrocarbon
generation). Pore pressure is hydrostatic.
"""
from __future__ import annotations
import numpy as np

from .constants import G, RHO_W, NU


def hydrostatic_pressure(z_bsf_m, water_depth_m=0.0, rho_w=RHO_W, g=G):
    """Hydrostatic pore pressure (Pa) at depth z_bsf_m below the seafloor."""
    z = np.asarray(z_bsf_m, float)
    W = np.asarray(water_depth_m, float)
    return rho_w * g * (np.maximum(W, 0.0) + np.maximum(z, 0.0))


def water_load(water_depth_m, rho_w=RHO_W, g=G):
    """Vertical stress (Pa) at the seafloor from the overlying water column."""
    return rho_w * g * np.maximum(np.asarray(water_depth_m, float), 0.0)


def overburden_increment(thickness_m, phi, rho_g, rho_w=RHO_W, g=G):
    """Vertical-stress increment (Pa) through a layer of given thickness and porosity."""
    rho_b = bulk_density(phi, rho_g, rho_w)
    return rho_b * g * np.maximum(np.asarray(thickness_m, float), 0.0)


def bulk_density(phi, rho_g, rho_w=RHO_W):
    """Bulk density ρ_b = φ ρ_w + (1 - φ) ρ_g (kg/m^3)."""
    phi = np.clip(np.asarray(phi, float), 0.0, 0.95)
    return phi * rho_w + (1.0 - phi) * np.asarray(rho_g, float)


def effective_stress(sigma_v, pore_pressure):
    """Terzaghi effective stress σ' = σ_v - P_p (Pa)."""
    return np.asarray(sigma_v, float) - np.asarray(pore_pressure, float)


def k0_jaky(friction_deg):
    """Simplified Jáky (1944) K_0 = 1 - sin φ'."""
    phi = np.deg2rad(np.asarray(friction_deg, float))
    return 1.0 - np.sin(phi)


def k0_poroelastic(nu=NU):
    """Uniaxial-strain poroelastic K_0 = ν / (1 - ν) (Zoback 2007)."""
    nu = float(nu)
    if not 0.0 < nu < 0.5:
        raise ValueError("Poisson's ratio must be between 0 and 0.5")
    return nu / (1.0 - nu)


def horizontal_stress(sigma_v, pore_pressure, friction_deg=30.0, method="jaky", nu=NU, alpha=1.0):
    """Total horizontal stress (Pa). method is 'jaky' or 'poroelastic'."""
    sigma_v = np.asarray(sigma_v, float)
    Pp = np.asarray(pore_pressure, float)
    if method == "jaky":
        K0 = k0_jaky(friction_deg)
        return K0 * (sigma_v - Pp) + Pp
    if method == "poroelastic":
        K0 = k0_poroelastic(nu)
        return K0 * (sigma_v - alpha * Pp) + alpha * Pp
    raise ValueError("method must be 'jaky' or 'poroelastic'")


def column_stresses(tops_bsf, thick, phi, rho_g, water_depth_m, rho_w=RHO_W, g=G,
                    friction_deg=30.0, method="jaky", nu=NU):
    """Stresses at the midpoint of each unit in a stacked column.

    tops_bsf, thick, phi, rho_g, friction_deg are length-n arrays (NaN = unit absent).
    Returns dict of arrays (Pa): sigma_v, pore_pressure, sigma_eff, sigma_h, and
    mid-point depths z_mid (m below seafloor).
    """
    tops = np.asarray(tops_bsf, float)
    th = np.asarray(thick, float)
    n = len(tops)
    z_mid = tops + 0.5 * th
    sigma_v = np.full(n, np.nan)
    Pp = np.full(n, np.nan)
    load = water_load(water_depth_m, rho_w, g)
    # Integrate from the top of the column downward, skipping absent units.
    order = np.lexsort((np.arange(n), np.where(np.isfinite(tops), tops, np.inf)))
    running = load
    for j in order:
        if not np.isfinite(tops[j]) or not np.isfinite(th[j]) or th[j] <= 0:
            continue
        dsig = overburden_increment(th[j], phi[j], rho_g[j], rho_w, g)
        sigma_v[j] = running + 0.5 * dsig
        Pp[j] = hydrostatic_pressure(z_mid[j], water_depth_m, rho_w, g)
        running = running + dsig
    seff = effective_stress(sigma_v, Pp)
    sh = horizontal_stress(sigma_v, Pp, friction_deg, method=method, nu=nu)
    return dict(z_mid=z_mid, sigma_v=sigma_v, pore_pressure=Pp, sigma_eff=seff, sigma_h=sh)
