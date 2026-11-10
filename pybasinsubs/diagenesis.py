# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Phase 1 mechanical compaction, quartz cementation and property evolution.

This module is opt-in: nothing here changes the existing Athy-law backstrip
(:mod:`pybasinsubs.backstrip`) or the existing thermal history. Burial geometry
(unit tops and thicknesses versus time) and temperature come from those 0.2.0
results; the porosities computed here are a diagnostic overlay.

Mechanical compaction
---------------------
Schneider, Potdevin, Wolf & Faille (1996, Tectonophysics 263, 307-317) couple
porosity to vertical effective stress and take the plastic limit as the maximum
effective stress reached so far. The widely used exponential form of that
stress-based rheology, written with a residual (minimum) intergranular volume
as in Lander & Walderhaug (1999, AAPG Bulletin 83, 433-449), is

    φ_mech(σ'_max) = φ_r + (φ_0 - φ_r) exp(-β σ'_max)                       (1)

Unloading (uplift, erosion, or a drop in effective stress) does not restore
porosity: σ'_max is remembered (Bowers 1995, SPE Drilling & Completion 10,
89-95; Schneider et al. 1996 plastic limit). This is the same idea as the
existing irreversible maximum-depth rule in :mod:`pybasinsubs.backstrip`, but
expressed in effective stress rather than burial depth.

Equation (1) is not the full Schneider elastoplastic-viscoplastic model (that
paper also has an elastic unload/reload loop and a temperature-dependent
viscoplastic term for pressure solution). Phase 1 uses only the irreversible
exponential compaction law.

Quartz cement
-------------
Walderhaug (1996, AAPG Bulletin 80, 731-745) gives the precipitation rate per
unit quartz surface area as an empirical logarithmic function of temperature:

    r = a  10^{b T}     (mol cm^{-2} s^{-1})                                (2)

with a = 1.98e-22 mol cm^{-2} s^{-1} and b = 0.022 °C^{-1}. The volume of
quartz cement precipitated in a time step is

    ΔV_q = (M_q / ρ_q) A r Δt                                               (3)

with M_q = 60.08 g mol^{-1} and ρ_q = 2.65 g cm^{-3}. The quartz surface area
per bulk volume (cm^2 cm^{-3}), including a grain-coating factor C
(Walderhaug 1996; Lander & Walderhaug 1999), is

    A = [6 f_q (1 - φ_0) (1 - C) / D] (φ / φ_ref)                           (4)

where D is grain diameter, f_q is the detrital quartz fraction of the solid,
and φ_ref is the porosity at the onset of significant cementation (first time
the unit reaches 80 °C) or φ_0 before that. Quartz cementation becomes
quantitatively important around 70-80 °C (Walderhaug 1996; Bjørlykke & Jahren
2012, AAPG Bulletin 96, 2193-2214). The rate law (2) is applied at all T; the
80 °C threshold is used only to set φ_ref and to mark onset on plots.

Porosity after cementation:

    φ = max(φ_mech - V_q, φ_irr)                                            (5)

V_q is stored as a fraction of current bulk volume. This matches Walderhaug's
statement that mechanical porosity loss is typically minor after quartz
cementation has started; a depositional-volume reference frame (as in the
full Lander & Walderhaug Exemplar model) is not used in phase 1.

Temperature history is the Easy%Ro time-temperature path already computed by
:func:`pybasinsubs.thermal.burial_temperatures` (Sweeney & Burnham 1990
integral, reused here only as T(t); maturity itself is unchanged).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd

from ._paths import data_path
from . import stress as _stress
from . import petrophysics as _petra
from .thermal import burial_temperatures, MYR, column_profile, bht_correct, heat_flow_from_temperature

WALDERHAUG_A = 1.98e-22          # mol / cm^2 / s
WALDERHAUG_B = 0.022             # 1 / °C
M_QUARTZ = 60.08                 # g / mol
RHO_QUARTZ = 2.65                # g / cm^3
T_CEMENT_ONSET_C = 80.0          # marker / surface-area reference (not a hard rate cutoff)
PHI_IRR = 0.01
DT_MYR_DEFAULT = 0.5

# Built-in copies of the shipped CSV, so the functions work if the file is missing.
_PRIOR_ROWS = None   # filled from CSV when available; see default_priors()


def mechanical_porosity(sigma_eff_pa, phi0, phi_r, beta_1_mpa):
    """Schneider-style irreversible mechanical porosity (equation 1). σ' in Pa."""
    sig_mpa = np.maximum(np.asarray(sigma_eff_pa, float), 0.0) / 1e6
    phi0 = np.asarray(phi0, float)
    phi_r = np.asarray(phi_r, float)
    beta = np.maximum(np.asarray(beta_1_mpa, float), 0.0)
    return phi_r + (phi0 - phi_r) * np.exp(-beta * sig_mpa)


def quartz_rate(temp_C):
    """Walderhaug (1996) precipitation rate r (mol cm^{-2} s^{-1})."""
    T = np.asarray(temp_C, float)
    return WALDERHAUG_A * 10.0 ** (WALDERHAUG_B * T)


def quartz_surface_area(f_quartz, grain_size_m, coat, phi, phi_ref, phi0):
    """Quartz surface area A (cm^2 per cm^3 bulk); equation (4)."""
    D_cm = np.maximum(np.asarray(grain_size_m, float) * 100.0, 1e-8)
    Vqdet = np.clip(np.asarray(f_quartz, float), 0.0, 1.0) * (1.0 - np.asarray(phi0, float))
    A0 = 6.0 * Vqdet * (1.0 - np.clip(np.asarray(coat, float), 0.0, 1.0)) / D_cm
    pref = np.maximum(np.asarray(phi_ref, float), 1e-6)
    return A0 * np.clip(np.asarray(phi, float) / pref, 0.0, 1.0)


def quartz_cement_increment(temp_C, area_cm2_cm3, dt_Myr):
    """Cement volume fraction precipitated over dt_Myr at temperature temp_C."""
    r = quartz_rate(temp_C)
    return (M_QUARTZ / RHO_QUARTZ) * area_cm2_cm3 * r * np.asarray(dt_Myr, float) * MYR


def default_priors():
    """Lithology parameter table (grain size, mineralogy, compaction, k params).

    Each physical column has a matching ``_err`` and ``_source`` column. Sources
    are labelled measured, literature or assumed, with a short citation.
    Nothing in this table is a measurement from well 30/4-1.
    """
    path = data_path("north_sea", "lithology_priors.csv")
    if os.path.exists(path):
        return pd.read_csv(path).set_index("name")
    raise FileNotFoundError(f"lithology priors not found at {path}")


def _numeric_prior_cols():
    return ["phi0", "phi_r", "beta_1_MPa", "grain_size_m", "f_quartz", "f_clay",
            "coat", "friction_deg", "kc_factor"]


def mix_unit_priors(column, priors=None):
    """Fraction-weighted mean of end-member priors for each column unit."""
    priors = default_priors() if priors is None else priors
    names = column.lith_names
    F = column.fractions()
    rows = []
    lith = column.lithologies.loc[names]
    for j, unit in enumerate(column.units.unit.astype(str)):
        rec = {"unit": unit}
        f = F[j]
        for col in _numeric_prior_cols():
            vals = []
            for name in names:
                key = name if name in priors.index else _alias(name)
                vals.append(float(priors.loc[key, col]))
            rec[col] = float(np.dot(f, vals))
        rec["rho_g"] = float(np.dot(f, lith.rho_g.to_numpy(float)))
        rec["f_shale"] = float(sum(fj * _lith_endmember_weights(n).get("shale", 0.0)
                                   for fj, n in zip(f, names)))
        rec["f_sandstone"] = float(sum(fj * _lith_endmember_weights(n).get("sandstone", 0.0)
                                       for fj, n in zip(f, names)))
        rec["f_chalk"] = float(sum(fj * _lith_endmember_weights(n).get("chalk", 0.0)
                                   for fj, n in zip(f, names)))
        rows.append(rec)
    return pd.DataFrame(rows)


def _alias(name):
    n = str(name).lower()
    if n in {"shale", "mudstone", "mud", "mudstone_tw"}:
        return "shale"
    if n in {"sandstone", "sandstone_tw", "sand", "shaly_sandstone"}:
        return "sandstone"
    if n in {"chalk", "carbonate", "limestone"}:
        return "chalk"
    if n in {"marine_avg"}:
        return "shale"
    return name


ENDMEMBERS = ("shale", "sandstone", "chalk")


def _lith_endmember_weights(name):
    """Map a lithology name to shale/sandstone/chalk weights (sum to 1).

    ``shaly_sandstone`` is split half-and-half so Walderhaug (1996) quartz
    cement is not applied to the entire mixed lithology as if it were clean
    sand. The VG1 template codes Triassic–Lower Jurassic as shaly_sandstone.
    Other names use :func:`_alias`.
    """
    n = str(name).lower().strip()
    if n in {"shaly_sandstone", "shaly sand", "shaly-sandstone"}:
        return {"sandstone": 0.5, "shale": 0.5}
    key = name if name in ENDMEMBERS else _alias(name)
    if key in ENDMEMBERS:
        return {key: 1.0}
    return {}


def _unit_fractions(column):
    """(n_units, 3) sandstone/shale/chalk fractions aligned with ENDMEMBERS."""
    names = column.lith_names
    F = column.fractions()
    out = np.zeros((len(column), len(ENDMEMBERS)))
    for i, name in enumerate(names):
        for em, wt in _lith_endmember_weights(name).items():
            out[:, ENDMEMBERS.index(em)] += F[:, i] * wt
    s = out.sum(1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1), 0.0)


def unit_permeability_mix(phi_end, frac, priors):
    """Geometric mean of end-member permeabilities, each at its own porosity."""
    logs, wts = [], []
    for k, name in enumerate(ENDMEMBERS):
        f = float(frac[k])
        if f <= 0 or not np.isfinite(phi_end[k]):
            continue
        kk = _petra.permeability(
            phi_end[k],
            float(priors.loc[name, "grain_size_m"]),
            float(priors.loc[name, "f_clay"]),
            name,
            kc_factor=float(priors.loc[name, "kc_factor"]),
        )
        logs.append(np.log(max(kk, 1e-32)))
        wts.append(f)
    if not wts:
        return np.nan
    return float(np.exp(np.dot(wts, logs) / sum(wts)))


def _sample_endmember_priors(priors, rng):
    """Draw one Monte Carlo realisation of the end-member prior table."""
    drawn = priors.copy()
    for col in _numeric_prior_cols():
        err = f"{col}_err"
        if err not in drawn.columns:
            continue
        sd = np.maximum(drawn[err].to_numpy(float), 0.0)
        drawn[col] = drawn[col].to_numpy(float) + sd * rng.standard_normal(len(drawn))
    drawn["phi0"] = np.clip(drawn["phi0"], 0.05, 0.85)
    drawn["phi_r"] = np.clip(drawn["phi_r"], 0.0, drawn["phi0"] - 0.02)
    drawn["beta_1_MPa"] = np.maximum(drawn["beta_1_MPa"], 0.005)
    drawn["grain_size_m"] = np.maximum(drawn["grain_size_m"], 1e-7)
    drawn["f_quartz"] = np.clip(drawn["f_quartz"], 0.0, 1.0)
    drawn["f_clay"] = np.clip(drawn["f_clay"], 0.01, 0.99)
    drawn["coat"] = np.clip(drawn["coat"], 0.0, 0.99)
    drawn["friction_deg"] = np.clip(drawn["friction_deg"], 10.0, 45.0)
    drawn["kc_factor"] = np.maximum(drawn["kc_factor"], 50.0)
    return drawn


def _interp_on_nodes(ages_nodes, values, t):
    """Interpolate a (n_nodes,) series onto time t (Ma) without extrapolating.

    A unit that has been eroded has NaN tops at younger nodes; those ages are
    outside the finite range and return NaN rather than the last buried value.
    """
    a = np.asarray(ages_nodes, float)
    v = np.asarray(values, float)
    m = np.isfinite(a) & np.isfinite(v)
    if m.sum() == 0:
        return np.nan
    amin, amax = float(a[m].min()), float(a[m].max())
    if t < amin - 1e-9 or t > amax + 1e-9:
        return np.nan
    if m.sum() == 1:
        return float(v[m][0])
    oo = np.argsort(a[m])
    return float(np.interp(t, a[m][oo], v[m][oo]))



def _thickness_at_time(j, t, u, ages_n, thick_n):
    """Decompacted thickness of unit j at time t, including a linear deposition wedge.

    The backstrip engine adds a unit at its top age. Between age_base and age_top
    we grow thickness from 0 to the value recorded at age_top so burial, depth
    and effective stress stay consistent.
    """
    age_top = float(u.age_top_Ma[j])
    age_base = float(u.age_base_Ma[j])
    eroded = bool(u.eroded[j])
    if u.thickness_m[j] <= 0 and not eroded:
        return np.nan
    if t > age_base + 1e-9:
        return np.nan
    eroded_at = float(u.eroded_at_Ma[j]) if "eroded_at_Ma" in u.columns else np.nan
    if eroded and np.isfinite(eroded_at) and t < eroded_at - 1e-9:
        return np.nan
    if t <= age_top + 1e-9:
        th = _interp_on_nodes(ages_n, thick_n[:, j], t)
        return max(th, 0.0) if np.isfinite(th) else np.nan
    th_end = _interp_on_nodes(ages_n, thick_n[:, j], age_top)
    if not np.isfinite(th_end):
        vals = np.asarray(thick_n[:, j], float)
        m = np.isfinite(vals)
        th_end = float(vals[m][-1]) if m.any() else float(u.thickness_m[j])
    span = max(age_base - age_top, 1e-6)
    grown = (age_base - t) / span * th_end
    return grown if grown > 1e-6 else np.nan


def _forward_path(result, thermal_hist, mixed, priors, t_grid, T_s=4.0, k0_method="jaky"):
    """Integrate mechanical compaction + sandstone quartz cement along one realisation.

    End-members are evolved separately. Walderhaug (1996) quartz cement is applied
    only to the sandstone fraction (the lithology that law was calibrated on).
    Shale and chalk get mechanical compaction only in phase 1.
    """
    b = result.best
    ages_n = b["age"][0]
    tops_n = b["tops"][0]
    thick_n = b["thick"][0]
    W_n = b["W"][0]
    u = result.column.units
    nU = len(u)
    nT = len(t_grid)
    frac = _unit_fractions(result.column)
    phi = np.full((nT, nU), np.nan)
    phi_mech = np.full((nT, nU), np.nan)
    vq = np.full((nT, nU), np.nan)
    k = np.full((nT, nU), np.nan)
    seff = np.full((nT, nU), np.nan)
    sv = np.full((nT, nU), np.nan)
    pp = np.full((nT, nU), np.nan)
    sh = np.full((nT, nU), np.nan)
    temp = np.full((nT, nU), np.nan)
    z_bsf = np.full((nT, nU), np.nan)
    z_bsl = np.full((nT, nU), np.nan)
    thick_t = np.full((nT, nU), np.nan)
    top_bsf = np.full((nT, nU), np.nan)
    top_bsl = np.full((nT, nU), np.nan)
    onset = np.full(nU, np.nan)
    sigma_max = np.zeros(nU)
    # Per-unit, per-end-member state.
    Vq_e = np.zeros((nU, 3))
    phi_e = np.zeros((nU, 3))
    pref_e = np.zeros((nU, 3))
    for ik, name in enumerate(ENDMEMBERS):
        phi_e[:, ik] = float(priors.loc[name, "phi0"])
        pref_e[:, ik] = float(priors.loc[name, "phi0"])
    phi_now = (frac * phi_e).sum(1)
    T_hist = thermal_hist["T"]
    T_ages = thermal_hist["age"]

    for it, t in enumerate(t_grid):
        W = _interp_on_nodes(ages_n, W_n, t)
        if not np.isfinite(W):
            W = 0.0
        SL = _interp_on_nodes(ages_n, b["SL"][0], t)
        if not np.isfinite(SL):
            SL = 0.0
        tops = np.full(nU, np.nan)
        th = np.full(nU, np.nan)
        Tmid = np.full(nU, np.nan)
        present = np.zeros(nU, dtype=bool)
        # Grow depositing units and stack from the seafloor so depth and σ' match.
        for j in range(nU):
            hj = _thickness_at_time(j, t, u, ages_n, thick_n)
            if not np.isfinite(hj):
                continue
            th[j] = hj
            present[j] = True
            Tmid[j] = _interp_on_nodes(T_ages, T_hist[:, j], t)
            if not np.isfinite(Tmid[j]):
                Tmid[j] = T_s
        D = 0.0
        for j in range(nU):
            if present[j]:
                tops[j] = D
                D += th[j]
        for _ in range(4):
            rec = _stress.column_stresses(
                tops, th, phi_now, mixed.rho_g.to_numpy(float), W,
                friction_deg=mixed.friction_deg.to_numpy(float), method=k0_method,
            )
            for j in np.where(present)[0]:
                if not np.isfinite(rec["sigma_eff"][j]):
                    continue
                sigma_max[j] = max(sigma_max[j], max(rec["sigma_eff"][j], 0.0))
                for ik, name in enumerate(ENDMEMBERS):
                    if frac[j, ik] <= 0:
                        continue
                    pm = mechanical_porosity(
                        sigma_max[j], float(priors.loc[name, "phi0"]),
                        float(priors.loc[name, "phi_r"]), float(priors.loc[name, "beta_1_MPa"]),
                    )
                    phi_e[j, ik] = max(pm - Vq_e[j, ik], PHI_IRR)
                phi_now[j] = float(np.dot(frac[j], phi_e[j]))
        rec = _stress.column_stresses(
            tops, th, phi_now, mixed.rho_g.to_numpy(float), W,
            friction_deg=mixed.friction_deg.to_numpy(float), method=k0_method,
        )
        if it == 0:
            dt = 0.0
            T_prev = Tmid.copy()
        else:
            dt = max(t_grid[it - 1] - t, 0.0)
        for j in np.where(present)[0]:
            Tj = Tmid[j] if np.isfinite(Tmid[j]) else T_s
            Tuse = Tj
            if dt > 0 and np.isfinite(T_prev[j]):
                Tuse = 0.5 * (Tj + T_prev[j])
            pm_e = np.zeros(3)
            for ik, name in enumerate(ENDMEMBERS):
                if frac[j, ik] <= 0:
                    continue
                pm_e[ik] = mechanical_porosity(
                    sigma_max[j], float(priors.loc[name, "phi0"]),
                    float(priors.loc[name, "phi_r"]), float(priors.loc[name, "beta_1_MPa"]),
                )
                # Quartz cement only in sandstone (Walderhaug 1996 calibration lithology).
                if name == "sandstone" and dt > 0:
                    if Tuse >= T_CEMENT_ONSET_C and not np.isfinite(onset[j]):
                        onset[j] = t
                        pref_e[j, ik] = max(pm_e[ik], PHI_IRR)
                    A = quartz_surface_area(
                        float(priors.loc[name, "f_quartz"]),
                        float(priors.loc[name, "grain_size_m"]),
                        float(priors.loc[name, "coat"]),
                        max(pm_e[ik] - Vq_e[j, ik], PHI_IRR),
                        pref_e[j, ik],
                        float(priors.loc[name, "phi0"]),
                    )
                    Vq_e[j, ik] = min(
                        Vq_e[j, ik] + quartz_cement_increment(Tuse, A, dt),
                        max(pm_e[ik] - PHI_IRR, 0.0),
                    )
                phi_e[j, ik] = max(pm_e[ik] - Vq_e[j, ik], PHI_IRR)
            phi_now[j] = float(np.dot(frac[j], phi_e[j]))
            pm = float(np.dot(frac[j], pm_e))
            Vq = float(np.dot(frac[j], Vq_e[j]))
            phi[it, j] = phi_now[j]
            phi_mech[it, j] = pm
            vq[it, j] = Vq
            k[it, j] = unit_permeability_mix(phi_e[j], frac[j], priors)
            seff[it, j] = rec["sigma_eff"][j]
            sv[it, j] = rec["sigma_v"][j]
            pp[it, j] = rec["pore_pressure"][j]
            sh[it, j] = rec["sigma_h"][j]
            temp[it, j] = Tj
            z_bsf[it, j] = rec["z_mid"][j]
            z_bsl[it, j] = rec["z_mid"][j] + W - SL
            thick_t[it, j] = th[j]
            top_bsf[it, j] = tops[j]
            top_bsl[it, j] = tops[j] + W - SL
        T_prev = Tmid.copy()
    return dict(phi=phi, phi_mechanical=phi_mech, quartz_cement=vq, permeability_m2=k,
                effective_stress_pa=seff, vertical_stress_pa=sv, pore_pressure_pa=pp,
                horizontal_stress_pa=sh, temperature_C=temp, depth_bsf_m=z_bsf,
                depth_bsl_m=z_bsl, thickness_m=thick_t,
                top_bsf_m=top_bsf, top_bsl_m=top_bsl,
                cement_onset_Ma=onset, sigma_max_pa=sigma_max)


VARIABLES = (
    "porosity", "phi_mechanical", "quartz_cement", "permeability_m2",
    "effective_stress_mpa", "pore_pressure_mpa", "vertical_stress_mpa",
    "horizontal_stress_mpa", "temperature_C", "depth_bsf_m", "depth_bsl_m",
)


def _pack_variables(path):
    """Convert internal arrays to the public variable set (MPa where noted)."""
    out = {
        "porosity": path["phi"],
        "phi_mechanical": path["phi_mechanical"],
        "quartz_cement": path["quartz_cement"],
        "permeability_m2": path["permeability_m2"],
        "effective_stress_mpa": path["effective_stress_pa"] / 1e6,
        "pore_pressure_mpa": path["pore_pressure_pa"] / 1e6,
        "vertical_stress_mpa": path["vertical_stress_pa"] / 1e6,
        "horizontal_stress_mpa": path["horizontal_stress_pa"] / 1e6,
        "temperature_C": path["temperature_C"],
        "depth_bsf_m": path["depth_bsf_m"],
        "depth_bsl_m": path["depth_bsl_m"],
        "top_bsf_m": path["top_bsf_m"],
        "top_bsl_m": path["top_bsl_m"],
        "thickness_m": path["thickness_m"],
    }
    return out


def _long_table(t_grid, units, best_vars, p10=None, p90=None):
    rows = []
    nT, nU = best_vars["porosity"].shape
    for it in range(nT):
        for j, name in enumerate(units):
            if not np.isfinite(best_vars["porosity"][it, j]):
                continue
            for var in VARIABLES:
                val = best_vars[var][it, j]
                lo = p10[var][it, j] if p10 is not None else val
                hi = p90[var][it, j] if p90 is not None else val
                rows.append(dict(time_ma=t_grid[it], unit=name, variable=var,
                                 p10=lo, p50=val, p90=hi))
    return pd.DataFrame(rows)


def _present_summary(t_grid, units, mixed, best_vars, path, p10=None, p90=None):
    # last finite sample per unit
    rows = []
    for j, name in enumerate(units):
        col = best_vars["porosity"][:, j]
        ok = np.where(np.isfinite(col))[0]
        if len(ok) == 0:
            continue
        it = ok[-1]
        rec = dict(unit=name, time_ma=t_grid[it],
                   phi0=float(mixed.phi0[j]),
                   f_sandstone=float(mixed.f_sandstone[j]),
                   f_shale=float(mixed.f_shale[j]),
                   f_chalk=float(mixed.f_chalk[j]),
                   grain_size_m=float(mixed.grain_size_m[j]),
                   f_quartz=float(mixed.f_quartz[j]),
                   coat=float(mixed.coat[j]),
                   cement_onset_Ma=float(path["cement_onset_Ma"][j]))
        for var in VARIABLES:
            rec[var] = float(best_vars[var][it, j])
            if p10 is not None:
                rec[var + "_p10"] = float(p10[var][it, j])
                rec[var + "_p90"] = float(p90[var][it, j])
        rec["permeability_mD"] = float(_petra.m2_to_md(rec["permeability_m2"]))
        rec["mech_loss"] = rec["phi0"] - rec["phi_mechanical"]
        rec["remaining_porosity"] = rec["porosity"]
        rows.append(rec)
    return pd.DataFrame(rows)


ASSUMPTIONS = [
    "New porosities are a diagnostic overlay: they do not feed back into decompaction, isostasy or conductivity.",
    "Pore pressure is hydrostatic (no overpressure). Effective stress is therefore a maximum for a given overburden.",
    "Mechanical compaction uses the exponential Schneider / Lander–Walderhaug form with irreversible σ'_max (Bowers 1995), not the full Schneider viscoplastic pressure-solution term.",
    "Walderhaug (1996) quartz cement is applied only to the sandstone end-member (the calibration lithology), using a single d50. Shale and chalk receive no quartz cement in phase 1. Panda & Lake (1994) grain-size distribution effects are not included.",
    "No carbonate pressure solution, no smectite-to-illite, no other cements (illite, calcite, pyrite).",
    "Grain size, quartz fraction, clay fraction and grain coatings are literature / assumed priors, not measurements from the well.",
    "Burial geometry and temperature come from the existing Athy backstrip and steady geotherm (constant present-day heat flow).",
    "Cement volume is stored as a fraction of current bulk volume, not in a depositional-volume reference frame.",
    "Mixed-lithology permeability is a geometric mean of end-member laws, each evaluated at that end-member's own porosity.",
    "Chalk/carbonate permeability uses the Kozeny form k = c φ³/S² (Mortensen, Engstrøm & Lind 1998; Fabricius 2007) with c = 0.25 and a literature-prior BET surface of 2.0 m²/g, not a well measurement and not a fit of Mallon et al. (2005).",
    "Monte Carlo (when requested) perturbs the diagenesis priors and, if a backstrip Monte Carlo is present, scales temperature with the draw's burial depth. It is not a fully coupled redraw of the geotherm.",
]


@dataclass
class PropertyResult:
    column: object
    backstrip: object
    ages: np.ndarray
    units: list
    best: dict
    table: pd.DataFrame
    present: pd.DataFrame
    mixed: pd.DataFrame
    priors: pd.DataFrame
    cement_onset_Ma: np.ndarray
    mc_used: bool
    n_mc: int
    heat_flow: float
    assumptions: list = field(default_factory=lambda: list(ASSUMPTIONS))
    p10: dict | None = None
    p90: dict | None = None

    def long_table(self):
        """Long-format table (time_ma, unit, variable, p10, p50, p90)."""
        return self.table



def _emit_progress(progress, done, total, message=""):
    """Optional ``progress(done, total, message)`` hook; ignored when unset."""
    if progress is None:
        return
    progress(int(done), int(max(total, 1)), message)


def evolve_properties(result, thermal_hist=None, heat_flow=None, T_s=4.0,
                      n_mc=200, seed=0, dt_Myr=DT_MYR_DEFAULT, priors=None,
                      k0_method="jaky", progress=None):
    """Walk a backstripped column through time and update φ, cement, k and stress.

    Parameters
    ----------
    result : BackstripResult
        Existing (opt-in caller) 1-D backstrip. Not modified.
    thermal_hist : dict or None
        Output of :func:`pybasinsubs.thermal.burial_temperatures`. Computed
        from ``heat_flow`` if omitted.
    heat_flow : float
        Surface heat flow (W/m^2) used only if ``thermal_hist`` is omitted.
    n_mc : int
        Monte Carlo draws of the diagenesis priors (0 or 1 = deterministic).
        Percentiles are p10 / p50 / p90. p50 is the deterministic (mean-prior)
        run, not the median of the draws, so the published central line stays
        tied to the documented priors.
    progress : callable or None
        Optional ``progress(done, total, message)`` invoked after the
        deterministic run and after each Monte Carlo draw. ``done``/``total``
        count those steps (1 when ``n_mc`` is 0 or 1). The callback must not
        modify the result.
    """
    if result is None or not hasattr(result, "best") or not hasattr(result, "column"):
        raise TypeError("result must be a BackstripResult")
    n_mc = int(n_mc)
    if n_mc < 0:
        raise ValueError("n_mc must be >= 0")
    dt_Myr = float(dt_Myr)
    if not np.isfinite(dt_Myr) or dt_Myr <= 0:
        raise ValueError("dt_Myr must be a positive finite number")
    if k0_method not in {"jaky", "poroelastic"}:
        raise ValueError("k0_method must be 'jaky' or 'poroelastic'")
    if heat_flow is not None:
        heat_flow = float(heat_flow)
        if not np.isfinite(heat_flow) or heat_flow <= 0:
            raise ValueError("heat_flow must be a positive finite number (W/m^2)")
    priors = default_priors() if priors is None else priors.copy()
    q = 0.06 if heat_flow is None else float(heat_flow)
    if thermal_hist is None:
        thermal_hist = burial_temperatures(result, heat_flow=q, T_s=T_s, depth="mid")
    mixed = mix_unit_priors(result.column, priors)
    ages_n = result.best["age"][0]
    t0 = float(np.nanmax(np.r_[ages_n, result.column.units.age_base_Ma.to_numpy(float)]))
    t_grid = np.arange(t0, -dt_Myr * 0.25, -dt_Myr)
    if t_grid[-1] > 0:
        t_grid = np.r_[t_grid, 0.0]
    path = _forward_path(result, thermal_hist, mixed, priors, t_grid, T_s=T_s, k0_method=k0_method)
    best = _pack_variables(path)
    p10 = p90 = None
    mc_used = False
    n_draw = 0
    if n_mc and n_mc > 1:
        rng = np.random.default_rng(seed)
        draws = {var: [] for var in VARIABLES}
        T_best = thermal_hist["T"]
        z_best = thermal_hist["z_bsf"]
        # Optional burial-depth scaling from the backstrip Monte Carlo.
        use_mc_burial = result.mc is not None and "tops" in result.mc
        S_bs = result.mc["tops"].shape[0] if use_mc_burial else 0
        mc_total = int(n_mc) + 1
        _emit_progress(progress, 1, mc_total, f"Monte Carlo 0/{int(n_mc)}")
        for s in range(int(n_mc)):
            priors_s = _sample_endmember_priors(priors, rng)
            mixed_s = mix_unit_priors(result.column, priors_s)
            th_s = dict(thermal_hist)
            if use_mc_burial and S_bs > 0:
                i = s % S_bs
                z_s = result.mc["tops"][i] + 0.5 * result.mc["thick"][i]
                scale = np.ones_like(z_best)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sc = np.where(np.abs(z_best) > 1.0, z_s / np.where(z_best == 0, np.nan, z_best), 1.0)
                sc = np.clip(np.where(np.isfinite(sc), sc, 1.0), 0.3, 3.0)
                th_s = dict(age=thermal_hist["age"],
                            T=T_s + (T_best - T_s) * sc,
                            z_bsf=z_s)
            path_s = _forward_path(result, th_s, mixed_s, priors_s, t_grid, T_s=T_s, k0_method=k0_method)
            packed = _pack_variables(path_s)
            for var in VARIABLES:
                draws[var].append(packed[var])
            _emit_progress(progress, s + 2, mc_total, f"Monte Carlo {s + 1}/{int(n_mc)}")
        p10 = {var: np.nanpercentile(np.stack(draws[var], 0), 10, axis=0) for var in VARIABLES}
        p90 = {var: np.nanpercentile(np.stack(draws[var], 0), 90, axis=0) for var in VARIABLES}
        mc_used = True
        n_draw = int(n_mc)
    else:
        _emit_progress(progress, 1, 1, "Computing properties…")
    units = result.column.units.unit.astype(str).tolist()
    table = _long_table(t_grid, units, best, p10, p90)
    present = _present_summary(t_grid, units, mixed, best, path, p10, p90)
    return PropertyResult(
        column=result.column, backstrip=result, ages=t_grid, units=units,
        best=best, table=table, present=present, mixed=mixed, priors=priors,
        cement_onset_Ma=path["cement_onset_Ma"], mc_used=mc_used, n_mc=n_draw,
        heat_flow=q, p10=p10, p90=p90,
    )


def heat_flow_from_column_bht(result, T_s=4.0, default_tsc_h=12.0):
    """BHT-derived present-day heat flow (W/m^2) using the existing thermal helpers.

    Uses ``column.meta['bht_degC']`` and ``td_bsf_m`` when present (Sodir wells).
    Time since circulation is assumed 12 h if not supplied, as in notebook 6.
    """
    meta = result.column.meta or {}
    if "bht_degC" not in meta or "td_bsf_m" not in meta:
        return None
    z = float(meta["td_bsf_m"])
    Tc, _, _ = bht_correct(meta["bht_degC"], z, None, T_s=T_s, default_tsc_h=default_tsc_h)
    zg, phi, k, _ = column_profile(result)
    return float(heat_flow_from_temperature(float(Tc[0]), z, zg, k, T_s=T_s))


def _section_unit_fractions(section):
    """(n_units, n_x, 3) end-member fractions from a flex2d Section."""
    nU, nX, _ = section.F.shape
    out = np.zeros((nU, nX, len(ENDMEMBERS)))
    for i, name in enumerate(section.lith):
        for em, wt in _lith_endmember_weights(name).items():
            out[:, :, ENDMEMBERS.index(em)] += section.F[:, :, i] * wt
    s = out.sum(-1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1.0), 0.0)


def _section_stage_temperatures(out, section, heat_flow=0.059, T_s=4.0):
    """Independent-column steady geotherms at each restoration stage (unit mid-points).

    Same conductivity model as :func:`pybasinsubs.section_thermal.section_thermal_data`
    (Athy porosity from the Section, lithology matrix conductivity, no 2-D heat
    conduction). Only mid-point temperature is returned; Easy%Ro is not computed.
    """
    from . import thermal
    from .lithology import default_lithologies

    r = out.restoration
    km = default_lithologies().loc[list(section.lith), "k_m"].to_numpy(float)
    nS, nU, nX = r.top.shape
    temp = np.full((nS, nU, nX), np.nan)
    for s in range(nS):
        youngest = min(s, nU - 1)
        for ix in range(nX):
            surface = r.top[s, youngest, ix]
            if not np.isfinite(surface):
                for u0 in range(youngest, nU):
                    if np.isfinite(r.top[s, u0, ix]):
                        surface = r.top[s, u0, ix]
                        break
                else:
                    continue
            z_nodes, k_nodes, units = [0.0], [], []
            for unit in range(youngest, nU):
                top, base = r.top[s, unit, ix], r.base[s, unit, ix]
                if not np.isfinite(top + base) or base - top <= 1e-8:
                    continue
                z_mid = max((top + base) * 0.5 - surface, 0.0)
                phi = (section.F[unit, ix] * section.phi0 * np.exp(-z_mid / section.C)).sum()
                matrix = np.exp(np.dot(section.F[unit, ix], np.log(km)))
                z_nodes.append(float(z_mid))
                k_nodes.append(thermal.bulk_conductivity(phi, matrix))
                units.append(unit)
            if not k_nodes:
                continue
            first = units[0]
            phi_s = np.dot(section.F[first, ix], section.phi0)
            matrix = np.exp(np.dot(section.F[first, ix], np.log(km)))
            kk = np.r_[thermal.bulk_conductivity(phi_s, matrix), k_nodes]
            tt = thermal.steady_geotherm(z_nodes, kk, heat_flow, T_s)
            for unit, tmid in zip(units, tt[1:]):
                temp[s, unit, ix] = tmid
    return temp


def _interp_along_stages(stage_ages, field, t):
    """Interpolate a (n_stages, ...) array onto geologic age t (Ma)."""
    sa = np.asarray(stage_ages, float)
    order = np.argsort(sa)
    sa = sa[order]
    arr = np.asarray(field, float)[order]
    t = float(t)
    if t < sa[0] - 1e-9 or t > sa[-1] + 1e-9:
        return np.full(arr.shape[1:], np.nan)
    if abs(t - sa[0]) <= 1e-9:
        return arr[0].copy()
    if abs(t - sa[-1]) <= 1e-9:
        return arr[-1].copy()
    i = int(np.searchsorted(sa, t, side="right"))
    i = min(max(i, 1), len(sa) - 1)
    t0, t1 = sa[i - 1], sa[i]
    w = (t - t0) / max(t1 - t0, 1e-12)
    a, b = arr[i - 1], arr[i]
    both = np.isfinite(a) & np.isfinite(b)
    out = np.where(both, (1.0 - w) * a + w * b, np.nan)
    out = np.where(np.isfinite(a) & ~np.isfinite(b) & (abs(t - t0) <= 1e-9), a, out)
    out = np.where(np.isfinite(b) & ~np.isfinite(a) & (abs(t - t1) <= 1e-9), b, out)
    return out


def _stack_column_stresses(thick, phi, rho_g, water_depth, friction_deg, method="jaky"):
    """Vectorized over x: thick/phi/rho_g/friction are (nU, nX); water_depth (nX,)."""
    nU, nX = thick.shape
    load = _stress.water_load(water_depth)
    sigma_v = np.full((nU, nX), np.nan)
    pp = np.full((nU, nX), np.nan)
    tops = np.zeros((nU, nX))
    z = np.zeros(nX)
    present_any = np.zeros(nX, dtype=bool)
    for j in range(nU):
        th = thick[j]
        present = np.isfinite(th) & (th > 1e-6)
        present_any |= present
        tops[j] = np.where(present, z, np.nan)
        dsig = np.where(present, _stress.overburden_increment(th, phi[j], rho_g[j]), 0.0)
        sigma_v[j] = np.where(present, load + 0.5 * dsig, np.nan)
        z_mid = np.where(present, z + 0.5 * np.nan_to_num(th), np.nan)
        pp[j] = np.where(present, _stress.hydrostatic_pressure(z_mid, water_depth), np.nan)
        load = load + dsig
        z = z + np.where(present, np.nan_to_num(th), 0.0)
    seff = _stress.effective_stress(sigma_v, pp)
    sh = _stress.horizontal_stress(sigma_v, pp, friction_deg, method=method)
    return dict(top_bsf=tops, z_mid=tops + 0.5 * thick, sigma_v=sigma_v,
                pore_pressure=pp, sigma_eff=seff, sigma_h=sh)


@dataclass
class SectionPropertyResult:
    """Per-column phase-1 properties on a restored 2-D section (no Monte Carlo)."""
    section: object
    restoration_out: object
    ages: np.ndarray
    stage_ages: np.ndarray
    units: list
    x: np.ndarray
    best: dict
    heat_flow: float
    frac: np.ndarray
    assumptions: list = field(default_factory=lambda: list(ASSUMPTIONS))
    temperature_source: str = ""

    def at_age(self, age_Ma, variable):
        """Return an (n_units, n_x) field linearly interpolated to ``age_Ma``."""
        return _interp_along_stages(self.ages, self.best[variable], float(age_Ma))

    def at_stage(self, stage, variable):
        """Field at a restoration stage (uses that stage's stored age)."""
        return self.at_age(float(self.stage_ages[stage]), variable)


def evolve_section_properties(out, section, heat_flow=0.059, T_s=4.0, dt_Myr=1.0,
                              thermal_data=None, priors=None, k0_method="jaky",
                              progress=None):
    """Apply the 1-D per-unit diagenesis model at every section x.

    Burial geometry is taken from the existing spatial restoration (Athy
    decompaction already done there). Units appear at their top age, matching
    the restoration frames; no extra deposition wedge is added. Temperature is
    taken from ``thermal_data`` (mean of the sub-samples from
    :func:`section_thermal_data`) when supplied, otherwise from independent
    steady columns at each restoration stage. New porosities do not feed back
    into the restoration or the geotherm. No Monte Carlo.
    progress : callable or None
        Optional ``progress(done, total, message)`` invoked after each
        time-integration step (every column advances together).
    """
    if out is None or not hasattr(out, "restoration"):
        raise TypeError("out must be a spatial restoration result")
    if section is None:
        raise TypeError("section is required")
    dt_Myr = float(dt_Myr)
    if not np.isfinite(dt_Myr) or dt_Myr <= 0:
        raise ValueError("dt_Myr must be a positive finite number")
    heat_flow = float(heat_flow)
    if not np.isfinite(heat_flow) or heat_flow <= 0:
        raise ValueError("heat_flow must be a positive finite number (W/m^2)")
    if k0_method not in {"jaky", "poroelastic"}:
        raise ValueError("k0_method must be 'jaky' or 'poroelastic'")
    priors = default_priors() if priors is None else priors.copy()
    r = out.restoration
    if list(section.units) != list(r.units) or not np.array_equal(section.x, r.x):
        raise ValueError("Section must match the restoration's units and input x")
    nS, nU, nX = r.top.shape
    stage_ages = np.asarray(r.ages, float)
    frac = _section_unit_fractions(section)
    rho_g = np.dot(section.F, np.asarray(section.rho_g, float))
    friction = np.zeros((nU, nX))
    for ik, name in enumerate(ENDMEMBERS):
        friction += frac[:, :, ik] * float(priors.loc[name, "friction_deg"])

    if thermal_data is not None:
        raw = np.asarray(thermal_data["temperature"], float)
        T_stage = np.nanmean(raw, axis=-1) if raw.ndim == 4 else raw
        if T_stage.shape != (nS, nU, nX):
            raise ValueError("thermal_data temperature shape does not match the restoration")
        t_source = ("section_thermal_data (mean of sub-samples; independent steady "
                    "columns + causal Easy%Ro temperatures, not 2-D heat conduction)")
    else:
        T_stage = _section_stage_temperatures(out, section, heat_flow=heat_flow, T_s=T_s)
        t_source = ("independent steady columns at each restoration stage "
                    "(Athy porosity from the Section, no 2-D heat conduction)")

    thick_stage = np.where(
        (r.base - r.top) > 1e-6, np.asarray(r.base - r.top, float), np.nan
    )
    for s in range(nS):
        if s > 0:
            thick_stage[s, :s] = np.nan
    W_stage = np.full((nS, nX), np.nan)
    for s in range(nS):
        youngest = min(s, nU - 1)
        surf = r.top[s, youngest]
        W_stage[s] = np.where(np.isfinite(surf), np.maximum(surf, 0.0), 0.0)

    t0 = float(np.nanmax(np.r_[stage_ages, np.asarray(section.age_base, float)]))
    t_grid = np.arange(t0, -dt_Myr * 0.25, -dt_Myr)
    if t_grid[-1] > 0:
        t_grid = np.r_[t_grid, 0.0]
    nT = len(t_grid)

    phi = np.full((nT, nU, nX), np.nan)
    phi_mech = np.full((nT, nU, nX), np.nan)
    vq = np.full((nT, nU, nX), np.nan)
    perm = np.full((nT, nU, nX), np.nan)
    seff = np.full((nT, nU, nX), np.nan)
    sv = np.full((nT, nU, nX), np.nan)
    pp = np.full((nT, nU, nX), np.nan)
    temp = np.full((nT, nU, nX), np.nan)
    thick_t = np.full((nT, nU, nX), np.nan)
    top_bsf = np.full((nT, nU, nX), np.nan)

    Vq_e = np.zeros((nU, 3, nX))
    phi_e = np.zeros((nU, 3, nX))
    pref_e = np.zeros((nU, 3, nX))
    for ik, name in enumerate(ENDMEMBERS):
        phi_e[:, ik] = float(priors.loc[name, "phi0"])
        pref_e[:, ik] = float(priors.loc[name, "phi0"])
    sigma_max = np.zeros((nU, nX))
    phi_now = (frac * np.moveaxis(phi_e, 1, -1)).sum(-1)

    T_prev = np.full((nU, nX), np.nan)
    i_sand = ENDMEMBERS.index("sandstone")

    _emit_progress(progress, 0, nT, f"Column evolution 0/{nT}")
    for it, t in enumerate(t_grid):
        th = _interp_along_stages(stage_ages, thick_stage, t)
        W = _interp_along_stages(stage_ages, W_stage, t)
        W = np.where(np.isfinite(W), W, 0.0)
        Tmid = _interp_along_stages(stage_ages, T_stage, t)
        Tmid = np.where(np.isfinite(th) & ~np.isfinite(Tmid), T_s, Tmid)
        rec = _stack_column_stresses(th, phi_now, rho_g, W, friction, method=k0_method)
        present = np.isfinite(th) & (th > 1e-6) & np.isfinite(rec["sigma_eff"])
        se = np.maximum(rec["sigma_eff"], 0.0)
        sigma_max = np.where(present, np.maximum(sigma_max, se), sigma_max)

        dt = 0.0 if it == 0 else max(t_grid[it - 1] - t, 0.0)
        Tuse = Tmid.copy()
        if dt > 0:
            both = np.isfinite(Tmid) & np.isfinite(T_prev)
            Tuse = np.where(both, 0.5 * (Tmid + T_prev), Tmid)

        pm_e = np.zeros((nU, 3, nX))
        for ik, name in enumerate(ENDMEMBERS):
            pm_e[:, ik] = mechanical_porosity(
                sigma_max, float(priors.loc[name, "phi0"]),
                float(priors.loc[name, "phi_r"]), float(priors.loc[name, "beta_1_MPa"]),
            )
        if dt > 0:
            sand_frac = frac[:, :, i_sand]
            do = present & (sand_frac > 0)
            hot = do & np.isfinite(Tuse) & (Tuse >= T_CEMENT_ONSET_C)
            # φ_ref is set the first time a column of this unit reaches 80 °C.
            pref_e[:, i_sand] = np.where(
                hot & (pref_e[:, i_sand] == float(priors.loc["sandstone", "phi0"])),
                np.maximum(pm_e[:, i_sand], PHI_IRR),
                pref_e[:, i_sand],
            )
            A = quartz_surface_area(
                float(priors.loc["sandstone", "f_quartz"]),
                float(priors.loc["sandstone", "grain_size_m"]),
                float(priors.loc["sandstone", "coat"]),
                np.maximum(pm_e[:, i_sand] - Vq_e[:, i_sand], PHI_IRR),
                pref_e[:, i_sand],
                float(priors.loc["sandstone", "phi0"]),
            )
            increment = quartz_cement_increment(np.where(do, Tuse, T_s), A, dt)
            increment = np.where(do, increment, 0.0)
            cap = np.maximum(pm_e[:, i_sand] - PHI_IRR, 0.0)
            Vq_e[:, i_sand] = np.minimum(Vq_e[:, i_sand] + increment, cap)

        for ik, name in enumerate(ENDMEMBERS):
            if name == "sandstone":
                phi_e[:, ik] = np.maximum(pm_e[:, ik] - Vq_e[:, ik], PHI_IRR)
            else:
                phi_e[:, ik] = np.maximum(pm_e[:, ik], PHI_IRR)
            # Absent end-members stay unused via frac weights.
        phi_now = (frac * np.moveaxis(phi_e, 1, -1)).sum(-1)
        pm = (frac * np.moveaxis(pm_e, 1, -1)).sum(-1)
        Vq = (frac * np.moveaxis(Vq_e, 1, -1)).sum(-1)

        # Permeability: geometric mean of end-members at each x.
        logs = np.full((nU, nX), 0.0)
        wts = np.zeros((nU, nX))
        for ik, name in enumerate(ENDMEMBERS):
            f = frac[:, :, ik]
            use = present & (f > 0) & np.isfinite(phi_e[:, ik])
            if not use.any():
                continue
            kk = _petra.permeability(
                phi_e[:, ik],
                float(priors.loc[name, "grain_size_m"]),
                float(priors.loc[name, "f_clay"]),
                name,
                kc_factor=float(priors.loc[name, "kc_factor"]),
            )
            logs = np.where(use, logs + f * np.log(np.maximum(kk, 1e-32)), logs)
            wts = np.where(use, wts + f, wts)
        k_mix = np.where(wts > 0, np.exp(logs / np.where(wts > 0, wts, 1.0)), np.nan)

        mask = present
        phi[it] = np.where(mask, phi_now, np.nan)
        phi_mech[it] = np.where(mask, pm, np.nan)
        vq[it] = np.where(mask, Vq, np.nan)
        perm[it] = np.where(mask, k_mix, np.nan)
        seff[it] = np.where(mask, rec["sigma_eff"] / 1e6, np.nan)
        sv[it] = np.where(mask, rec["sigma_v"] / 1e6, np.nan)
        pp[it] = np.where(mask, rec["pore_pressure"] / 1e6, np.nan)
        temp[it] = np.where(mask, Tmid, np.nan)
        thick_t[it] = np.where(mask, th, np.nan)
        top_bsf[it] = np.where(mask, rec["top_bsf"], np.nan)
        T_prev = Tmid.copy()
        phi_now = np.where(mask, phi_now, phi_now)  # keep last phi for stacking continuity
        _emit_progress(progress, it + 1, nT, f"Column evolution {it + 1}/{nT}")

    best = dict(
        porosity=phi, phi_mechanical=phi_mech, quartz_cement=vq,
        permeability_m2=perm, effective_stress_mpa=seff,
        vertical_stress_mpa=sv, pore_pressure_mpa=pp,
        temperature_C=temp, thickness_m=thick_t, top_bsf_m=top_bsf,
    )
    extra = list(ASSUMPTIONS) + [
        "2-D geometry is the existing restoration (Athy decompaction, vertical-shear "
        "fault displacements). Overlay porosities do not change those thicknesses.",
        "Each x is evolved as an independent 1-D column. There is no lateral flow, "
        "no 2-D heat conduction and no bending-stress term in phase 1.",
        "Temperature: " + t_source + ".",
        "Units appear at their restoration top age (no extra deposition wedge).",
        "VG1 shaly_sandstone (Triassic–Lower Jurassic) is split 50/50 sandstone/shale "
        "so quartz cement is not applied as if the unit were clean sand.",
        "No Monte Carlo on the section; reported fields are the deterministic mean-prior run.",
    ]
    return SectionPropertyResult(
        section=section, restoration_out=out, ages=t_grid, stage_ages=stage_ages,
        units=list(r.units), x=np.asarray(r.x, float), best=best,
        heat_flow=float(heat_flow), frac=frac, assumptions=extra,
        temperature_source=t_source,
    )
