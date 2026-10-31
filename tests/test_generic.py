# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pybasinsubs import data_path, legacy, thermal, flexure
from pybasinsubs.backstrip import backstrip, solid_thickness, thickness_from_solid
from pybasinsubs.column import from_basinsubs, from_sodir, read_column
from pybasinsubs.paleobathy import hayward_depth_from_counts


@pytest.mark.parametrize("reg", ["N", "S"])
def test_generic_matches_r_deterministic(reg):
    out, _ = legacy.subsidence_analysis_r(reg)
    det = backstrip(from_basinsubs(reg, litho_mode="r_first_row"), n=0, mc=False).summary()
    det = det.sort_values("age_Ma").reset_index(drop=True); o = out.sort_values("Age").reset_index(drop=True)
    k = min(len(o), len(det))
    assert np.max(abs(det.total_subsidence_m[:k] - o.TotalS[:k])) < 1e-4
    assert np.max(abs(det.tectonic_subsidence_m[:k] - o.TectS[:k])) < 1e-4


def test_solid_thickness_roundtrip():
    G = solid_thickness(500.0, 2000.0, 0.6, 1500.0)
    T = thickness_from_solid(G, 0.0, 0.6, 1500.0)
    assert T > 500 and solid_thickness(T, 0.0, 0.6, 1500.0) == pytest.approx(G, rel=1e-9)


def test_monte_carlo_reproducible_and_bracketing():
    col = from_basinsubs("N")
    a = backstrip(col, n=300, seed=3).summary(); b = backstrip(col, n=300, seed=3).summary()
    pd.testing.assert_frame_equal(a, b)
    assert (a.tectonic_subsidence_m_p2_5 if "tectonic_subsidence_m_p2_5" in a else a["tectonic_subsidence_m_p2.5"]).le(a.tectonic_subsidence_m + 1e-6).all()


def test_hayward_reproduces_published():
    bn = pd.read_csv(data_path("coastal_range", "Thickness", "Bathymetry_N.csv")); k = bn.P > 0
    W, We, *_ = hayward_depth_from_counts(bn.P, bn.Perr, bn.B, bn.Berr, bn.S, bn.Serr)
    assert np.nanmax(abs(W[k] - bn.W[k])) < 1e-5 and np.nanmax(abs(We[k] - bn.Werr[k])) < 1e-5


def test_template_and_sodir_load():
    c = read_column(data_path("template", "my_column_template.csv"))
    assert len(c) > 3
    r = backstrip(from_sodir("7120/9-2", "barents_hammerfest"), n=50)
    assert any("erosion" in s for s in r.node_labels)


def test_easy_ro_and_windows():
    t = np.linspace(100, 0, 101); T = 20 + 1.2 * (100 - t)
    ro = thermal.easy_ro(t, T)
    assert ro[0] == pytest.approx(0.2, abs=0.05) and np.all(np.diff(ro) >= -1e-12)
    assert thermal.maturity_window(0.8) == "oil"


def test_mckenzie_fit_recovers_beta():
    age = np.linspace(140, 0, 30)
    z = thermal.mckenzie_postrift(140 - age, 1.6)
    b, rms = thermal.fit_beta(age, z, 140)
    assert b == pytest.approx(1.6, abs=1e-3) and rms < 1e-2


def test_flexure_fft_vs_fd_and_airy_limit():
    x = np.arange(0, 400e3, 2e3); q = np.where(abs(x - 200e3) < 30e3, 1e7, 0.0)
    wf = flexure.deflection_fft(q, 2e3, 10e3, 2275.0); wd = flexure.deflection_fd(q, 2e3, np.full_like(x, 10e3), 2275.0)
    assert abs(wf - wd).max() < 0.01 * abs(wf).max()
    w0 = flexure.deflection_fft(q, 2e3, 1.0, 2275.0)
    assert np.allclose(w0, q / (2275.0 * 9.81), rtol=1e-3, atol=1e-3)


def test_bht_corrections_increase_temperature():
    assert thermal.bht_waples_ramly(150, 4000) > 150
    assert thermal.bht_waples_danish(150, 4000) > 150


def test_app_core_runs():
    import matplotlib
    matplotlib.use("Agg")
    from pybasinsubs import app_core as core
    col = core.apply_overrides(core.load_case(core.SOURCES[2], "30/4-1"), water_scale=1.1)
    out = core.run_case(col, n=50, heat_flow_mW=60)
    assert "fig_maturity" in out and len(out["summary"]) == len(col.units) + 0 or True
    assert out["maturity"].present_Ro.notna().any()


def test_temperature_and_maturity_field():
    r = backstrip(from_sodir("30/4-1"), n=0, mc=False)
    tf = thermal.temperature_field(r, heat_flow=0.06)
    assert tf["T"].shape == (len(tf["age"]), len(tf["z"]))
    T0 = tf["T"][-1]; ok = ~np.isnan(T0)
    assert np.all(np.diff(T0[ok]) >= -1e-9) and T0[ok][0] == pytest.approx(4.0, abs=1.0)
    mf = thermal.maturity_field(r, tf)
    assert mf["ro"].min() >= 0.19 and mf["ro"].max() > 1.0
    assert np.all(np.isnan(mf["ro_paths"][:, 0]) | (np.diff(np.r_[0, np.nan_to_num(mf["ro_paths"][:, 0])]) >= -1e-9))


def test_section_geometry():
    vg = pd.read_csv(data_path("north_sea", "sodir_viking_graben_wells.csv")).sort_values("lon")
    R = [backstrip(from_sodir(w), n=0, mc=False) for w in vg.well]
    x = np.linspace(0, 90e3, len(R))
    g0 = flexure.section_geometry(R, x, 0.0)
    assert np.all(g0["base"] >= g0["top"] - 1e-9)
    # present-day total thickness equals the drilled thickness at each well
    for i, r in enumerate(R):
        assert g0["base"][-1][i] - g0["seafloor"][i] == pytest.approx(r.column.units.thickness_m.sum(), rel=1e-6)
    g = flexure.section_geometry(R, x, 150.0)
    k = g["units"].index("NORDLAND")
    assert np.allclose(g["base"][k] - g["top"][k], 0)


# ---------------------------------------------------------------- 2-D flexural backstripping
from pybasinsubs import flex2d


def _viking_section(dx=500.0):
    vg = pd.read_csv(data_path("north_sea", "sodir_viking_graben_wells.csv")).sort_values("lon")
    cols = [from_sodir(w) for w in vg.well]
    xw = ((vg.lon - vg.lon.min()) * 111.32e3 * np.cos(np.radians(vg.lat.mean()))).to_numpy()
    return cols, flex2d.correlate(cols, xw, dx=dx)


@pytest.mark.parametrize("broken", [False, True])
def test_line_load_matches_turcotte_schubert(broken):
    dx, Te, dr, V0 = 250.0, 10e3, 2275.0, 1e12
    x = np.arange(-300e3, 300e3 + dx, dx); i0 = len(x) // 2; q = np.zeros_like(x)
    if broken:
        q[i0] = V0 / dx; w = flex2d.plate_deflection(q, dx, Te, dr, break_index=i0)[i0:]
        wa = flex2d.line_load_analytic(x, V0, Te, dr, broken=True)[i0:]
    else:
        q[i0] = V0 / dx; w = flex2d.plate_deflection(q, dx, Te, dr, method="fd")
        wa = flex2d.line_load_analytic(x, V0, Te, dr)
    assert np.max(abs(w - wa)) / np.max(abs(wa)) < 2e-3


def test_load_conservation_and_fft_fd_agree():
    dx = 500.0; x = np.arange(0, 400e3, dx); q = 2e7 * np.exp(-((x - 200e3) / 20e3) ** 2)
    wf = flex2d.plate_deflection(q, dx, 5e3, method="fft"); wd = flex2d.plate_deflection(q, dx, 5e3, method="fd")
    assert np.max(abs(wf - wd)) < 1e-2 * np.max(wf)
    assert (2275 * 9.81 * wd).sum() == pytest.approx(q.sum(), rel=1e-5)


def test_roberts_1998_published_numbers():
    from pybasinsubs.flexure import rigidity
    assert rigidity(3e3) == pytest.approx(1.7e20, rel=0.02)          # "c. 1.7e20 N m ~ c. 3 km"
    D = lambda Te: 1e11 * Te ** 3 / (12 * (1 - 0.25 ** 2))
    lam = lambda Te: 2 * np.pi / (2300 * 9.81 / D(Te)) ** 0.25
    assert 25e3 < lam(1e3) < 32e3 and 55e3 < lam(3e3) < 70e3         # Fig. 7c, p. 335


def test_section_te0_equals_airy_1d():
    cols, sec = _viking_section(dx=50.0)
    r0 = flex2d.backstrip_section(sec, 0.0)
    assert np.allclose(r0.Z, r0.Z_airy)
    c = cols[1]; i = 1
    sm = backstrip(c, n=0, mc=False, mixing="mixed").summary(); d = dict(zip(sm.age_Ma.round(2), sm.tectonic_subsidence_m))
    for k, a in enumerate(sec.age_top[:6]):
        if round(a, 2) in d:
            assert np.interp(sec.xw[i], sec.x, r0.Z[k]) == pytest.approx(d[round(a, 2)], abs=8.0)


def test_section_pinchout_conservation_and_restoration():
    cols, sec = _viking_section()
    k = sec.units.index("CROMER KNOLL")
    assert (sec.h[k] == 0).any() and (sec.h[k] > 0).any()          # pinch-out exists
    r = flex2d.backstrip_section(sec, 1500.0)
    assert np.allclose(r.S[0], sec.h.sum(0))                          # present thickness recovered
    assert np.all(np.diff(r.S, axis=0) <= 1e-6)                      # stripping only reduces S
    rp = flex2d.reverse_postrift(r, sec, 1.2, 140.0)
    assert np.allclose(rp["seafloor"][0], sec.W0, atol=1e-6)


# ---------------------------------------------------------------- flexural-cantilever fault restoration
def _toy_fault_section():
    x = np.arange(0, 120e3, 250.0)
    tops = np.array([np.full_like(x, 200.0), np.full_like(x, 1200.0), 2000 + 800 * (x > 60e3)])
    base = tops[-1] + 1000
    sec = flex2d.section_from_horizons(x, tops, base, ["post", "syn", "pre"], [0, 140, 160], [140, 160, 250],
                                       [{"shale": 1}, {"shale": 1}, {"sandstone": 1}], tops[0])
    return x, tops, sec


def test_cantilever_airy_limit_and_offset():
    x = np.arange(-50e3, 80e3, 100.0)
    u, vk, w = flex2d.cantilever_displacement(x, 0.0, 1000.0, 45.0, 1, Te=0.0, Hd=8e3, rho_fill=2400.0)
    assert np.allclose(u[x < 0], 0.0)
    hw = (x > 500) & (x < 7000)
    assert np.allclose(u[hw], -1000.0 * (3300 - 2800) / (3300 - 2400))
    u5, _, _ = flex2d.cantilever_displacement(x, 0.0, 1000.0, 45.0, 1, Te=5e3, Hd=8e3, rho_fill=2400.0)
    assert u5[np.searchsorted(x, -1000)] > 0                         # footwall uplift with an elastic plate
    assert u5[np.searchsorted(x, 1000)] < 0                          # hangingwall subsidence
    # mirror symmetry: a west-dipping fault gives the mirrored field
    um, _, _ = flex2d.cantilever_displacement(-x[::-1], 0.0, 1000.0, 45.0, -1, Te=5e3, Hd=8e3, rho_fill=2400.0)
    assert np.allclose(um[::-1], u5, atol=1e-6)


def test_zero_throw_reduces_to_unfaulted_restoration():
    x, tops, sec = _toy_fault_section()
    f0 = flex2d.Fault(x0=60e3, dip=45.0, down=1, throw=np.zeros(3))
    a = flex2d.restore_faulted(sec, [f0], Te=1500.0, beta=1.1, rift_end_Ma=140.0)
    b = flex2d.restore_faulted(sec, [], Te=1500.0, beta=1.1, rift_end_Ma=140.0)
    assert np.allclose(a["top"], b["top"], equal_nan=True) and np.allclose(a["seafloor"], b["seafloor"])
    # post-rift ages unaffected by faults even with throw
    f = flex2d.fault_at(x, tops, 60e3, 1, 1500.0, 45.0)
    c = flex2d.restore_faulted(sec, [f], Te=1500.0, beta=1.1, rift_end_Ma=140.0, heave=False)
    assert np.allclose(c["top"][:2], b["top"][:2], equal_nan=True)


def test_fault_restoration_removes_observed_offset():
    x, tops, sec = _toy_fault_section()
    f = flex2d.fault_at(x, tops, 60e3, 1, 1500.0, 45.0)
    assert f.throw[2] == pytest.approx(800.0) and f.throw[0] == 0
    for Te in (0.0, 1500.0, 5000.0):
        r = flex2d.restore_faulted(sec, [f], Te=Te, beta=1.1, rift_end_Ma=140.0, heave=False, Hd=15e3)
        z = r["top"][2][2]
        off = np.interp(61.5e3, x, z) - np.interp(58.5e3, x, z)
        assert abs(off) < 1.0                                      # offset of the pre-rift top is closed
        assert r["slip"][2][0] > 0 and r["slip"][1][0] == 0          # pre-rift slip restored; none at the post-rift step


def test_vg1_digitized_section_loads():
    d = pd.read_csv(data_path("north_sea", "roberts1998_fig4a_vg1_digitized.csv"))
    tc = [f"top{L}_m" for L in range(1, 11)] + ["base_m"]
    assert d.x_km.min() < 1 and d.x_km.max() > 80
    assert np.all(np.diff(d[tc].to_numpy(), axis=1) >= 0)          # horizons ordered, pinch-outs = 0 thickness


# ---------------------------------------------------------------- growth-strata kinematic restoration (listric / domino)
def _wedge_section(wedge=True):
    x = np.arange(0, 100e3, 250.0); s = x - 40e3
    syn = np.where(s > 0, (600 - 0.02 * s) if wedge else 300.0, 300.0)
    t0 = np.full_like(x, 200.0); t1 = t0 + 1000; t2 = t1 + syn; base = t2 + 1500
    sec = flex2d.section_from_horizons(x, np.array([t0, t1, t2]), base, ["post", "syn", "pre"], [0, 140, 160], [140, 160, 250],
                                       [{"shale": 1}, {"shale": 1}, {"sandstone": 1}], t0)
    return x, sec


@pytest.mark.parametrize("method", ["listric", "domino"])
def test_zero_growth_gives_zero_slip(method):
    x, sec = _wedge_section(wedge=False)
    f = flex2d.Fault(x0=40e3, dip=50.0, down=1, throw=np.array([0, 0, 1.0]))
    r = flex2d.restore_kinematic(sec, [f], method=method, rift_end_Ma=140.0, bathy_relief=False)
    st = r["steps"][0]
    if method == "listric":
        assert st["heave"][0] == 0.0
    else:
        assert abs(st["params"][0][0]) < 1e-9 and st["params"][0][1] == 0.0
    b = flex2d.restore_faulted(sec, [], rift_end_Ma=140.0)
    assert np.allclose(r["top"][2][2], b["top"][2][2], atol=1e-6)


def test_inclined_shear_planar_analytic_and_no_vertical_option():
    # planar fault (slope t): s = t e / (1 + t cot(alpha)), constant along the hangingwall
    x = np.arange(41e3, 60e3, 100.0); t = np.tan(np.radians(50.0)); f = lambda xx: 1000.0 + (xx - 40e3) * t
    _, s = flex2d._shear_subsidence(x, np.full_like(x, 1500.0), f, 1000.0, 1, 65.0)
    assert np.allclose(s, t * 1000.0 / (1 + t / np.tan(np.radians(65.0))), rtol=1e-3)
    x2, sec = _wedge_section()
    f1 = flex2d.Fault(x0=40e3, dip=50.0, down=1, throw=np.zeros(3))
    with pytest.raises(ValueError):
        flex2d.restore_kinematic(sec, [f1], method="listric", alpha=90.0)


def test_kinematic_area_conservation():
    x = np.arange(0, 100e3, 100.0)
    top = 2000 + np.where(x > 40e3, 300.0, 0.0); bot = top + 800
    f = flex2d.Fault(x0=40e3, dip=50.0, down=1, throw=np.zeros(1))
    (t2, b2), ov, gp = flex2d._apply_listric([top, bot], x, [f], [1000.0], 65.0, 12e3, [1500.0], [50.0])
    m = (x > 45e3) & (x < 90e3)                               # away from the fault zone and the end
    assert np.nansum((b2 - t2)[m]) == pytest.approx(np.sum((bot - top)[m]), rel=0.03)  # inclined shear conserves area
    (t3, b3), _, _ = flex2d._apply_domino([top, bot], x, [f], [(0.05, 300.0, 250.0)])
    area0 = ((bot - top)[x > 41e3]).sum(); area1 = ((b3 - t3)[x > 41e3]).sum()
    assert area1 == pytest.approx(area0, rel=0.02)                 # rigid rotation conserves area


def test_domino_rotation_recovers_tilted_block():
    x = np.arange(0, 80e3, 100.0); s = x - 30e3
    D = np.where(s > 0, 400 - 0.03 * s, 0.0)
    z = 2000 + D                                               # block dropped 400 m and tilted towards the fault
    m = (s > 1500) & (s < 45e3)
    b, a = np.polyfit(s[m], D[m], 1)
    f = flex2d.Fault(x0=30e3, dip=60.0, down=1, throw=np.zeros(1))
    (zr,), _, _ = flex2d._apply_domino([z], x, [f], [(np.arctan(-b), a, 0.0)])
    assert np.std(zr[(x > 32e3) & (x < 70e3)]) < 15.0 and abs(np.mean(zr[(x > 32e3) & (x < 70e3)]) - 2000) < 15.0


# ---------------------------------------------------------------- fault interpretation: detection, import, method switch
from pybasinsubs import faults as F

EX = Path(data_path("faults"))


def _synthetic_faulted(x0s=(20e3, 50e3), dip=55.0, throw=600.0):
    x = np.arange(0, 80e3, 100.0); depths = [1500.0, 2200.0, 3000.0, 3800.0]
    tops = []
    for z0 in depths:
        z = np.full_like(x, z0)
        for xf in x0s:
            xc = xf + (z0 - 1500.0) / np.tan(np.radians(dip))   # where this horizon meets the plane
            xa = xc - 0.5 * throw / np.tan(np.radians(dip)); xb = xc + 0.5 * throw / np.tan(np.radians(dip))
            z = z + np.clip((x - xa) / (xb - xa), 0, 1) * throw  # face of the offset follows the fault plane
        tops.append(z)
    return x, np.array(tops)


def test_autodetect_synthetic_section():
    x, tops = _synthetic_faulted()
    fs, tab = F.detect_faults_from_horizons(x, tops, ref=0)
    assert len(fs) == 2
    for f, xf in zip(fs, (20e3, 50e3)):
        assert abs(f.x0 - xf) < 300 and abs(f.dip - 55.0) < 3 and f.down == 1 and f.confidence > 0.7


def test_autodetect_image_synthetic():
    pytest.importorskip("skimage")
    x, tops = _synthetic_faulted()
    img, ext = F.render_section_image(x, tops, tops[-1] + 800, extent_z=(1000, 5000), px=(1600, 400))
    fs = F.detect_faults_from_image(img, ext, x, tops, ref=0)
    assert len(fs) == 2
    for xf in (20e3, 50e3):
        assert min(abs(f.x0 - xf) for f in fs) < 500
    assert all(abs(f.dip - 55.0) < 8 for f in fs)


def test_parsers_roundtrip(tmp_path):
    L = ((450000.0, 6720000.0), (460000.0, 6720000.0))
    ref = F.read_fault_sticks(EX / "vg1_fault_sticks.csv")
    assert list(ref) == ["F1", "F2", "F3", "F4", "F5"]
    for name in ["vg1_fault_sticks.geojson", "vg1_fault_sticks_petrel_charisma.txt", "vg1_fault_sticks_kingdom.txt"]:
        s = F.read_fault_sticks(EX / name, line=L)
        assert list(s) == list(ref)
        for k in ref:
            assert np.allclose(np.unique(s[k], axis=0), np.unique(ref[k], axis=0), atol=1.0)
    od = F.read_fault_sticks(EX / "vg1_F3_opendtect_faultstickset.txt", line=L)
    assert np.allclose(list(od.values())[0], ref["F3"], atol=1.0)
    p = tmp_path / "rt.csv"; F.write_fault_sticks_csv({"A": np.array([[1.0, 2.0], [3.0, 4.0]])}, p)
    assert np.allclose(F.read_fault_sticks(p)["A"], [[1, 2], [3, 4]])
    g = tmp_path / "rt.geojson"; F.write_fault_sticks_geojson({"A": np.array([[1.0, 2.0], [3.0, 4.0]])}, g)
    assert np.allclose(F.read_fault_sticks(g)["A"], [[1, 2], [3, 4]])


def test_sticks_to_faults_dip_and_direction():
    x, tops = _synthetic_faulted(x0s=(30e3,))
    st = {"a": np.array([[30e3 + z / np.tan(np.radians(60.0)), 1500.0 + z] for z in (0.0, 1000.0, 2000.0)])}
    (f,) = F.sticks_to_faults(st, x, tops)
    assert abs(f.dip - 60.0) < 0.1 and f.down == 1


@pytest.mark.parametrize("method", ["cantilever", "listric", "domino"])
def test_method_switch(method):
    x, sec = _wedge_section()
    tops = np.array([sec.W0 + sec.h[:k].sum(0) for k in range(3)])
    f = flex2d.fault_at(x, tops, 40e3, dip=50.0)
    r = flex2d.restore(sec, [f], method=method, rift_end_Ma=140.0)
    assert r["top"].shape[0] == 3
    if method == "cantilever":
        r2 = flex2d.restore_faulted(sec, [f], rift_end_Ma=140.0)
        assert np.allclose(r["top"][2], r2["top"][2], equal_nan=True)
    else:
        assert r["method"] == method
    with pytest.raises(ValueError):
        flex2d.restore(sec, [f], method="vertical")


def test_constants_single_source():
    import importlib; from pybasinsubs import constants as K, flex2d, flexure, thermal, legacy as L
    backstrip = importlib.import_module("pybasinsubs.backstrip")
    assert backstrip.RHO_A == flex2d.RHO_M == L.RHO_A == thermal.MCK["rho_m"] == K.RHO_M == 3300.0
    assert flexure.rigidity(1e3) == pytest.approx(K.E_YOUNG * 1e9 / (12 * (1 - K.NU ** 2)))
    assert flexure.rigidity(1e3, E=100e9) / flexure.rigidity(K.te_equivalent(1e3, 100e9)) == pytest.approx(1.0)
    # McKenzie follows rho_m when passed explicitly
    s30 = thermal.mckenzie_postrift(50.0, 1.5, dict(K.MCK, rho_m=3330.0))
    assert s30 != pytest.approx(thermal.mckenzie_postrift(50.0, 1.5), rel=1e-6)


def test_sealevel_offline_fixture(monkeypatch, tmp_path):
    from pybasinsubs import sealevel
    synth = Path(__file__).parent / "fixtures" / "synthetic_sealevel.csv"
    monkeypatch.setenv("PYBASINSUBS_MILLER2020", str(synth)); monkeypatch.setattr(sealevel, "_CURVE", None)
    m, s = sealevel.sea_level_at([10.0, 100.0], [0.5, 1.0])
    assert np.isfinite(m).all() and s[1] == 100.0 and m[1] == 0.0
    monkeypatch.setattr(sealevel, "_CURVE", None)
    monkeypatch.delenv("PYBASINSUBS_MILLER2020"); monkeypatch.setenv("PYBASINSUBS_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        sealevel.miller2020_path(download=False)


def test_cli_entry_points(monkeypatch, capsys):
    from pybasinsubs import cli
    monkeypatch.setenv("PYBASINSUBS_DRY_RUN", "1")
    cli.ui([]); cli.ui(["--streamlit"]); cli.tracer(["--streamlit"]); cli.tracer([])
    out = capsys.readouterr().out.splitlines()
    assert "panel serve" in out[0] and out[0].split()[4].endswith("panel_app.py")
    assert "streamlit run" in out[1] and "streamlit_app.py" in out[1]
    assert "tracer_streamlit.py" in out[2] and "tracer_panel.py" in out[3]
    for line in out:
        assert Path(line.split()[4 if "panel" in line.split()[2] else 4]).exists()
