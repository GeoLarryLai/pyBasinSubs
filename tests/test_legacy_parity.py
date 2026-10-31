# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pandas as pd
import pytest
from pybasinsubs import data_path, legacy

rd = lambda *p: pd.read_csv(data_path("coastal_range", *p))


def num(df):
    return df.select_dtypes("number").to_numpy(float)


@pytest.mark.parametrize("reg", ["N", "S"])
def test_subsidence_and_rates(reg):
    res, rates = legacy.subsidence_analysis_r(reg)
    assert np.nanmax(abs(num(res) - num(rd("r_outputs", f"Subsidence results_{reg}.csv")))) < 1e-9
    assert np.nanmax(abs(num(rates) - num(rd("r_outputs", f"vertical rates_{reg}.csv")))) < 1e-9


@pytest.mark.real_miller
@pytest.mark.parametrize("reg", ["N", "S"])
def test_age_model_and_intermediates(reg):
    am = legacy.age_model_r(reg)
    assert np.nanmax(abs(num(am["regressions"]) - num(rd("r_outputs", f"Age_DepthModel_{reg}.csv")))) < 1e-8
    assert np.nanmax(abs(num(am["plot_regressions"]) - num(rd("r_outputs", f"Depth_AgeModel_{reg}.csv")))) < 1e-8
    assert np.nanmax(abs(num(am["ages_new"]) - num(rd("Thickness", f"Ages_{reg}_new.csv")))) < 1e-9
    assert np.nanmax(abs(num(legacy.pb_age_r(reg)) - num(rd("Thickness", f"PB_Age_{reg}.csv")))) < 1e-9
    assert np.nanmax(abs(num(legacy.pb_unit_r(reg)) - num(rd("Thickness", f"PB_Unit_{reg}.csv")))) < 1e-9
    assert np.nanmax(abs(num(legacy.sl_unit_r(reg)) - num(rd("Sealevel", f"SL_Unit_{reg}.csv")))) < 1e-9


def test_porosity_fit_matches_pofun():
    pofun = rd("Backstripping", "PoFun.csv")
    for f, row in [("Porosity_mud.csv", 0), ("Porosity_sand.csv", 1)]:
        tab = rd("Backstripping", f).dropna(subset=["InitialPorosity"])
        r = legacy.porosity_fit_r(tab)
        assert r["phi0"] == pytest.approx(pofun.Porosity_0[row], abs=1e-6)
        assert r["C"] == pytest.approx(pofun.DecayC[row], abs=1e-3)


def test_complex_step_matches_finite_difference():
    f = lambda v: v[0] * np.exp(-v[1] / v[2])
    m, s = legacy.propagate_first_order(f, [0.5, 1000.0, 2000.0], [0.01, 50.0, 100.0])
    h = 1e-6; x = np.array([0.5, 1000.0, 2000.0]); sd = np.array([0.01, 50.0, 100.0])
    J = [(f(x + h * np.eye(3)[i] * x[i]) - f(x - h * np.eye(3)[i] * x[i])) / (2 * h * x[i]) for i in range(3)]
    assert s == pytest.approx(np.sqrt(np.sum((np.array(J) * sd) ** 2)), rel=1e-6)
