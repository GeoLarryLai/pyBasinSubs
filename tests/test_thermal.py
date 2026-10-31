# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Thermal benchmarks: analytic geotherms, McKenzie (1978) closed forms, Easy%Ro (Sweeney & Burnham 1990)."""
import numpy as np
import pytest
from scipy.integrate import quad

from pybasinsubs import thermal as T
from pybasinsubs.constants import MCK

MCK78 = dict(MCK, rho_m=3330.0, rho_w=1000.0, kappa=8e-7)      # McKenzie (1978) parameter set


def test_steady_geotherm_constant_A_analytic():
    z = np.linspace(0, 5000, 2001); k = 2.5; qs = 0.065; A = 2e-6
    Ta = 4 + qs * z / k - A * z ** 2 / (2 * k)
    assert np.allclose(T.steady_geotherm(z, np.full_like(z, k), qs, 4.0, A), Ta, atol=1e-6)


def test_steady_geotherm_exponential_A_analytic():
    z = np.linspace(0, 8000, 4001); k = 2.5; qs = 0.065; A0 = 2.5e-6; hr = 3000.0
    Ta = 4 + (qs - A0 * hr) * z / k + A0 * hr ** 2 * (1 - np.exp(-z / hr)) / k
    assert np.allclose(T.steady_geotherm(z, np.full_like(z, k), qs, 4.0, A0, hr=hr), Ta, atol=1e-6)
    # array A(z) on the grid gives the same answer (trapezoid integration)
    Az = T.radiogenic_heat(z, A0, hr)
    assert np.allclose(T.steady_geotherm(z, np.full_like(z, k), qs, 4.0, Az), Ta, atol=0.01)


def test_transient_converges_to_steady_with_exponential_A():
    z = np.linspace(0, 6000, 301); k = np.full_like(z, 2.5); Az = T.radiogenic_heat(z, 2e-6, 4000.0)
    qb = 0.04
    Tt = T.transient_geotherm(z, lambda t: k, lambda t: qb, np.array([200.0, 0.0]), 4.0, Az, dt_Myr=5.0)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility
    qs = qb + trapz(Az, z)
    assert np.allclose(Tt[-1], T.steady_geotherm(z, k, qs, 4.0, Az), atol=0.05)


def test_mckenzie_E0_and_tau_match_mckenzie_1978():
    a, al, T1 = MCK78["a"], MCK78["alpha"], MCK78["T1"]
    E0 = 4 * a * 3330 * al * T1 / (np.pi ** 2 * 2330)
    assert E0 == pytest.approx(3170, rel=0.01)                      # ~3.17 km (McKenzie 1978)
    tau = a ** 2 / (np.pi ** 2 * MCK78["kappa"]) / T.MYR
    assert tau == pytest.approx(62.8, abs=0.2)                       # 62.8 Myr (McKenzie 1978)
    beta = 2.0
    S1 = T.mckenzie_postrift(np.array([tau, 1e5]), beta, MCK78)
    assert S1[0] / S1[1] == pytest.approx(1 - np.exp(-1), rel=1e-6)
    assert S1[1] == pytest.approx(E0 * beta / np.pi * np.sin(np.pi / beta), rel=1e-6)


@pytest.mark.parametrize("beta", [1.2, 2.0, 4.0])
def test_mckenzie_series_equilibrium_limit(beta):
    p = MCK78
    S_inf = p["a"] * p["rho_m"] * p["alpha"] * p["T1"] * (1 - 1 / beta) / (2 * (p["rho_m"] - p["rho_w"]))
    assert T.mckenzie_postrift(np.array([1e5]), beta, p, nterms=500)[0] == pytest.approx(S_inf, rel=2e-3)
    # first-term approximation vs series after >= 1 time constant: ~11 % low at beta 1.2, ~5 % at beta 2-4
    t = np.array([63.0, 150.0])
    gap = abs(1 - T.mckenzie_postrift(t, beta, p) / T.mckenzie_postrift(t, beta, p, nterms=200))
    assert np.all(gap < {1.2: 0.12, 2.0: 0.06, 4.0: 0.05}[beta])


def test_mckenzie_heat_flow_limits():
    q_eq = MCK["k"] * MCK["T1"] / MCK["a"]
    assert T.mckenzie_heat_flow(np.array([1e5]), 2.0)[0] == pytest.approx(q_eq, rel=1e-9)
    assert T.mckenzie_heat_flow(np.array([0.0]), 2.0, nterms=5000)[0] / q_eq == pytest.approx(2.0, rel=2e-3)   # q(0) = beta q_eq
    assert T.mckenzie_synrift(1.0) == pytest.approx(0.0)


def test_easy_ro_published_constants():
    k = T.EASY_RO
    assert len(k["f"]) == 20 and k["f"].sum() == pytest.approx(0.85)
    assert k["E_kcal"][0] == 34 and k["E_kcal"][-1] == 72 and k["A"] == 1e13
    assert np.exp(k["a"]) == pytest.approx(0.20, abs=0.003)                          # initial %Ro
    assert np.exp(k["a"] + k["b"] * 0.85) == pytest.approx(4.69, abs=0.01)           # maximum %Ro (S&B 1990: ~4.7)


def test_easy_ro_isothermal_exact():
    t = np.array([10.0, 0.0]); Tc = np.array([150.0, 150.0]); k = T.EASY_RO
    kk = k["A"] * np.exp(-k["E_kcal"] / (T.R_KCAL * 423.15)) * 10 * T.MYR
    F = np.sum(k["f"] * (1 - np.exp(-kk)))
    assert T.kinetic_conversion(t, Tc)[-1] == pytest.approx(F, rel=1e-6)


@pytest.mark.parametrize("rate", [1.0, 5.0, 10.0])
def test_easy_ro_constant_heating_rate_vs_quadrature(rate):
    """Constant heating rate H (C/Myr) from 20 C: F = sum f_i [1 - exp(-A/H int exp(-E_i/RT) dT)]."""
    k = T.EASY_RO; T0, T1_ = 20.0, 200.0
    times = np.linspace((T1_ - T0) / rate, 0, 400); temps = T0 + rate * (times[0] - times)
    H = rate / T.MYR
    I = np.array([quad(lambda TK: np.exp(-E / (T.R_KCAL * TK)), T0 + 273.15, T1_ + 273.15, epsabs=0, epsrel=1e-10)[0]
                  for E in k["E_kcal"]])
    ro = np.exp(-1.6 + 3.7 * np.sum(k["f"] * (1 - np.exp(-k["A"] / H * I))))
    assert T.easy_ro(times, temps)[-1] == pytest.approx(ro, rel=2e-3)


def test_easy_rodl_is_plugin_only():
    assert not hasattr(T, "EASY_RO_DL")
    assert "PLUG-IN ONLY" in T.__doc__


def test_bht_per_measurement_tsc():
    Tm = np.array([90.0, 110.0, 120.0]); z = np.array([2500.0, 3200.0, 3800.0]); tsc = [6.0, np.nan, 24.0]
    Tc, used, assumed = T.bht_correct(Tm, z, tsc)
    assert list(assumed) == [False, True, False] and used[1] == 12.0
    for i in range(3):
        assert Tc[i] == pytest.approx(T.bht_waples_ramly(Tm[i], z[i], used[i]))
    assert Tc[0] > T.bht_waples_ramly(Tm[0], z[0], 24.0)      # shorter wait -> larger correction
    Td, _, _ = T.bht_correct(Tm, z, 10.0, method="waples_danish")
    assert np.allclose(Td, T.bht_waples_danish(Tm, z, 10.0))
