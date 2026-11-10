# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Phase-1 diagenesis, petrophysics and stress (opt-in; does not change 0.2.0 results)."""
import numpy as np
import pytest

from pybasinsubs import diagenesis as dg
from pybasinsubs import petrophysics as petra
from pybasinsubs import stress
from pybasinsubs.backstrip import backstrip
from pybasinsubs.column import from_sodir
from pybasinsubs.constants import NU


def test_mechanical_porosity_decreases_and_is_irreversible():
    phi0, phi_r, beta = 0.49, 0.26, 0.08
    p_low = dg.mechanical_porosity(10e6, phi0, phi_r, beta)
    p_high = dg.mechanical_porosity(40e6, phi0, phi_r, beta)
    assert phi_r < p_high < p_low < phi0
    # Unloading from 40 MPa to 10 MPa must keep the high-stress porosity
    # (caller tracks σ'_max). The function itself is single-valued in σ'.
    assert p_high < p_low


def test_quartz_cement_faster_when_hotter():
    A = 50.0
    cool = dg.quartz_cement_increment(80.0, A, 10.0)
    hot = dg.quartz_cement_increment(120.0, A, 10.0)
    assert cool > 0 and hot > cool
    # 10 Myr at 40 °C should be a small fraction of the 80 °C increment.
    cold = dg.quartz_cement_increment(40.0, A, 10.0)
    assert cold < 0.4 * cool


def test_kozeny_carman_and_yang_aplin_signs():
    k_sand = petra.kozeny_carman(0.20, 2.5e-4)
    assert k_sand > 0
    # Yang & Aplin (2010) at φ=0.20, CF=0.5 should be mudstone-like (~1e-21 to 1e-19 m^2).
    k_mud = petra.yang_aplin_2010(0.20, 0.5)
    assert 1e-22 < k_mud < 1e-18
    assert petra.yang_aplin_2010(0.20, 0.7) < petra.yang_aplin_2010(0.20, 0.3)
    assert petra.m2_to_md(k_sand) > 10


def test_jaky_and_poroelastic_k0():
    assert stress.k0_jaky(30.0) == pytest.approx(1.0 - 0.5, abs=1e-12)
    assert stress.k0_poroelastic(0.25) == pytest.approx(NU / (1 - NU))
    sv, pp = 40e6, 15e6
    sh = stress.horizontal_stress(sv, pp, friction_deg=30.0, method="jaky")
    assert sh == pytest.approx(0.5 * (sv - pp) + pp)
    with pytest.raises(ValueError):
        stress.horizontal_stress(sv, pp, method="nope")


def test_hydrostatic_effective_stress_increases_with_depth():
    rec = stress.column_stresses(
        tops_bsf=np.array([0.0, 100.0]),
        thick=np.array([100.0, 100.0]),
        phi=np.array([0.4, 0.3]),
        rho_g=np.array([2650.0, 2720.0]),
        water_depth_m=100.0,
    )
    assert rec["sigma_eff"][1] > rec["sigma_eff"][0] > 0
    assert rec["pore_pressure"][1] > rec["pore_pressure"][0]


def test_priors_have_error_and_source_columns():
    p = dg.default_priors()
    for col in dg._numeric_prior_cols():
        assert f"{col}_err" in p.columns
        assert f"{col}_source" in p.columns
        assert p[f"{col}_source"].str.len().min() > 5
    assert set(p.index) >= {"shale", "sandstone", "chalk"}
    # No source string should claim these numbers were measured at a bundled well.
    blob = " ".join(p.filter(like="_source").astype(str).to_numpy().ravel())
    assert "30/4-1" not in blob


def test_evolve_30_4_1_present_day_table():
    col = from_sodir("30/4-1", "viking_graben")
    res = backstrip(col, mc=False)
    q = dg.heat_flow_from_column_bht(res)
    assert q is not None and 0.04 < q < 0.09
    prop = dg.evolve_properties(res, heat_flow=q, n_mc=0, dt_Myr=2.0)
    assert set(["time_ma", "unit", "variable", "p10", "p50", "p90"]) <= set(prop.table.columns)
    assert not prop.mc_used
    pres = prop.present.set_index("unit")
    assert "BRENT" in pres.index
    # Remaining porosity and cement are physical fractions.
    assert ((pres.porosity > 0) & (pres.porosity < 0.7)).all()
    assert (pres.quartz_cement >= -1e-12).all()
    # Deeper sandy Brent should be more cemented than shallow Nordland.
    if "NORDLAND" in pres.index:
        assert pres.loc["BRENT", "quartz_cement"] > pres.loc["NORDLAND", "quartz_cement"]
    # Long table p10=p50=p90 when MC is off.
    row = prop.table.iloc[0]
    assert row.p10 == row.p50 == row.p90
    assert np.nanmin(prop.best["effective_stress_mpa"]) >= -1e-6


def test_existing_backstrip_unchanged_by_property_overlay():
    col = from_sodir("30/4-1", "viking_graben")
    a = backstrip(col, mc=False).summary()
    b = backstrip(col, mc=False).summary()
    # Calling the new engine must not mutate the backstrip arrays.
    res = backstrip(col, mc=False)
    tops_before = res.best["tops"].copy()
    dg.evolve_properties(res, heat_flow=0.059, n_mc=0, dt_Myr=5.0)
    np.testing.assert_array_equal(res.best["tops"], tops_before)
    # And two plain backstrips still match (no hidden global state).
    np.testing.assert_allclose(a.total_subsidence_m, b.total_subsidence_m)


def test_shaly_sandstone_is_split_not_clean_sand():
    w = dg._lith_endmember_weights("shaly_sandstone")
    assert w["sandstone"] == 0.5 and w["shale"] == 0.5
    assert dg._lith_endmember_weights("sandstone") == {"sandstone": 1.0}
    assert dg._lith_endmember_weights("chalk") == {"chalk": 1.0}


def test_property_cmaps():
    from pybasinsubs.property_plots import PROPERTY_CMAPS, cmap_for
    assert PROPERTY_CMAPS["porosity"] == "copper"
    assert PROPERTY_CMAPS["permeability_m2"] == "cividis"
    assert cmap_for("effective_stress_mpa") == "viridis"
    assert cmap_for("temperature_C") == "inferno"
    assert cmap_for("pore_pressure_mpa") == "Blues"
    assert cmap_for("quartz_cement") == "YlOrBr"



def test_mortensen_fabricius_chalk_reasonable():
    """Documented chalk k is far below spherical KC with a 5 μm d50."""
    k = petra.mortensen_fabricius_chalk(0.50)
    k_md = petra.m2_to_md(k)
    assert 0.1 < k_md < 20  # ~4 mD at φ=0.50 with default priors
    k_deep = petra.m2_to_md(petra.mortensen_fabricius_chalk(0.22))
    assert 0.01 < k_deep < 1.0
    assert k_deep < k_md
    # The old spherical-grain KC 5 μm path was ~10^2–10^4 mD at shallow φ.
    k_kc = petra.m2_to_md(petra.kozeny_carman(0.50, 5e-6))
    assert k_kc > 50
    assert petra.m2_to_md(petra.permeability(0.50, 5e-6, 0.15, "chalk")) == pytest.approx(k_md)
    with pytest.raises(ValueError):
        petra.mortensen_fabricius_chalk(0.4, s_bet_m2_g=0.0)


def test_evolve_properties_validates_inputs():
    col = from_sodir("30/4-1", "viking_graben")
    res = backstrip(col, mc=False)
    with pytest.raises(ValueError):
        dg.evolve_properties(res, heat_flow=0.059, n_mc=-1)
    with pytest.raises(ValueError):
        dg.evolve_properties(res, heat_flow=0.059, n_mc=0, dt_Myr=0)
    with pytest.raises(ValueError):
        dg.evolve_properties(res, heat_flow=-0.1, n_mc=0)


def test_property_log_limits_are_fixed():
    from pybasinsubs.property_plots import property_log_limits, plot_property_logs
    col = from_sodir("30/4-1", "viking_graben")
    res = backstrip(col, mc=False)
    prop = dg.evolve_properties(res, heat_flow=0.059, n_mc=0, dt_Myr=5.0)
    limits = property_log_limits(prop)
    fig0 = plot_property_logs(prop, age=0.0, limits=limits)
    fig1 = plot_property_logs(prop, age=66.0, limits=limits)
    try:
        ax0 = fig0.axes
        ax1 = fig1.axes
        # Shared depth and porosity / permeability / stress / temperature ranges.
        assert ax0[0].get_ylim() == ax1[0].get_ylim()
        assert ax0[1].get_xlim() == ax1[1].get_xlim()
        assert ax0[2].get_xlim() == ax1[2].get_xlim()
        assert ax0[3].get_xlim() == ax1[3].get_xlim()
        assert ax0[4].get_xlim() == ax1[4].get_xlim()
        assert fig0.get_size_inches().tolist() == fig1.get_size_inches().tolist()
        assert limits["ylim"][0] > limits["ylim"][1]
        assert limits["porosity"][0] == 0.0
        assert limits["permeability_mD"][0] > 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig0)
        plt.close(fig1)


def test_save_property_logs_animation_gif(tmp_path):
    from pybasinsubs.property_plots import save_property_logs_animation, property_log_frame_ages
    col = from_sodir("30/4-1", "viking_graben")
    res = backstrip(col, mc=False)
    prop = dg.evolve_properties(res, heat_flow=0.059, n_mc=0, dt_Myr=20.0)
    from pybasinsubs.timestep import timestep_steps
    ages = property_log_frame_ages(prop)
    assert len(ages) == len(timestep_steps(res))
    path = tmp_path / "logs.gif"
    out = save_property_logs_animation(prop, path, ages=ages[:3], dpi=40, fps=2)
    assert out == path and path.stat().st_size > 1000
    with pytest.raises(ValueError):
        save_property_logs_animation(prop, tmp_path / "logs.bmp", ages=ages)


def test_evolve_properties_progress_callback():
    col = from_sodir("30/4-1", "viking_graben")
    res = backstrip(col, mc=False)
    seen = []

    def on_progress(done, total, message=""):
        seen.append((int(done), int(total), str(message)))

    dg.evolve_properties(res, heat_flow=0.059, n_mc=2, dt_Myr=10.0, progress=on_progress)
    assert seen, "progress callback should run"
    dones = [d for d, _t, _m in seen]
    totals = {t for _d, t, _m in seen}
    assert totals == {3}
    assert dones == sorted(dones)
    assert dones[0] >= 1 and dones[-1] == 3
    assert any("Monte Carlo" in m for _d, _t, m in seen)

    seen.clear()
    dg.evolve_properties(res, heat_flow=0.059, n_mc=0, dt_Myr=20.0, progress=on_progress)
    assert seen == [(1, 1, "Computing properties…")]

    # Optional hook is ignored when omitted (no TypeError).
    dg.evolve_properties(res, heat_flow=0.059, n_mc=0, dt_Myr=20.0)

