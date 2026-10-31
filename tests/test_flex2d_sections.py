# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Section CSV import, 2-D Monte Carlo, fault tips/cutoffs, apps, Restoration2D, notebooks."""
import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybasinsubs import flex2d, app_core as core, faults as F, column
from pybasinsubs._paths import data_path

ROOT = Path(__file__).resolve().parents[1]
B = importlib.import_module("pybasinsubs.backstrip")


def _vg1(dx=250.0):
    dv = pd.read_csv(data_path("north_sea", "roberts1998_fig4a_vg1_digitized.csv"))
    xg = np.arange(1000.0, 81001.0, dx)
    vt = np.array([np.interp(xg, dv.x_km * 1e3, dv[f"top{k}_m"]) for k in range(1, 11)])
    vb = np.interp(xg, dv.x_km * 1e3, dv.base_m)
    A = [0, 5.3, 23.0, 56.0, 66.0, 100.5, 140.0, 157.3, 166.1, 170.3]
    L = [{"shale": 1}] * 4 + [{"chalk": 1}, {"shale": 1}, {"shale": 1}, {"shale": 1}, {"sandstone": 1}, {"shaly_sandstone": 1}]
    sec = flex2d.section_from_horizons(xg, vt, vb, [f"u{k}" for k in range(10)], A, A[1:] + [252.0], L, vt[0])
    return xg, vt, vb, sec


# ---------------------------------------------------------------- section CSV
def test_section_csv_round_trip(tmp_path):
    t = flex2d.read_section_template()
    sec = flex2d.section_from_csv(t["horizons"], t["units"])
    assert sec.h.shape == (10, 161) and sec.wells == ["34/7-demo"] and sec.W[0, 0] == 100.0
    H, U = flex2d.section_to_csv(sec, tmp_path / "h.csv", tmp_path / "u.csv")
    sec2 = flex2d.section_from_csv(tmp_path / "h.csv", tmp_path / "u.csv")
    for a in ("h", "F", "W", "SL", "W0", "phi0", "C", "rho_g", "age_top", "age_base"):
        assert np.allclose(getattr(sec, a), getattr(sec2, a)), a
    assert sec2.units == sec.units and sec2.lith == sec.lith


def test_section_csv_pinchout_and_errors():
    H = pd.DataFrame(dict(x_m=[0, 1000, 2000] * 3 + [0, 2000], unit=["A"] * 3 + ["B"] * 3 + ["base"] * 3 + ["C", "C"],
                          depth_m=[100, 100, 100, 500, 400, 300, 900, 900, 900, 700, 700]))
    H = H[~((H.unit == "C") & (H.x_m == 2000))]                       # C picked only at x = 0
    U = pd.DataFrame(dict(unit=["A", "B", "C"], age_top_Ma=[0, 10, 20], f_shale=[1, 0.5, 0], f_sandstone=[0, 0.5, 1]))
    sec = flex2d.section_from_csv(H, U, dx=1000.0)
    assert sec.h[2, 0] == 200.0 and sec.h[2, -1] == 0.0               # C pinches out away from its pick
    assert np.allclose(sec.h.sum(0), 800.0) and sec.age_base[-1] == 20.0
    with pytest.raises(ValueError, match="base"):
        flex2d.section_from_csv(H[H.unit != "base"], U)
    with pytest.raises(ValueError, match="lithology"):
        flex2d.section_from_csv(H, U.drop(columns=["f_shale", "f_sandstone"]))


# ---------------------------------------------------------------- Monte Carlo
def test_mc_seed_reproducible_and_envelopes():
    sec, fs = core.load_section()
    a = flex2d.monte_carlo_section(sec, n=12, seed=7, Te_sd=500.0)
    b = flex2d.monte_carlo_section(sec, n=12, seed=7, Te_sd=500.0)
    c = flex2d.monte_carlo_section(sec, n=12, seed=8, Te_sd=500.0)
    assert np.array_equal(a.top, b.top, equal_nan=True) and np.array_equal(a.Z, b.Z, equal_nan=True)
    assert a.params.equals(b.params) and not np.array_equal(a.Z, c.Z, equal_nan=True)
    lo, mid, hi = a.envelope(3, 5)
    ok = ~np.isnan(mid)
    assert ok.any() and np.all(lo[ok] <= mid[ok] + 1e-9) and np.all(mid[ok] <= hi[ok] + 1e-9)
    assert np.all(a.width("Z")[1:] > 0)
    f = flex2d.monte_carlo_section(sec, fs, "cantilever", n=4, seed=1, Hd=8e3)
    assert f.method == "cantilever" and f.top.shape[1:] == f.best_top.shape


def test_delta_mode_matches_mc():
    col = column.read_column(data_path("template", "my_column_template.csv"))
    d = B.backstrip(col, mode="delta").summary()
    m = B.backstrip(col, n=3000, seed=0, vary={"thickness", "water", "sealevel", "lithology", "rho_w"}).summary()
    sd_d, sd_m = d.tectonic_subsidence_m_sd.to_numpy(), m.tectonic_subsidence_m_sd.to_numpy()
    assert np.allclose(sd_d, sd_m, rtol=0.15, atol=10.0)
    with pytest.raises(ValueError):
        B.backstrip(col, mode="bogus")


# ---------------------------------------------------------------- physics
@pytest.mark.parametrize("seed", range(6))
def test_cantilever_area_conservation(seed):
    """Load balance: the integral of u = v_k - w is (rho_m - rho_c)/(rho_m - rho_f) of the kinematic area, for any Te/dip/throw."""
    rng = np.random.default_rng(seed)
    t, dip, te, hd = rng.uniform(100, 2000), rng.uniform(20, 70), rng.uniform(0, 8000), rng.uniform(5e3, 20e3)
    x = np.arange(-300e3, 300e3, 100.0)
    u, vk, w = flex2d.cantilever_displacement(x, 0.0, t, dip, 1, te, Hd=hd, rho_fill=2400.0)
    rm, rc, rf = flex2d.cantilever.RHO_M, flex2d.cantilever.RHO_C, 2400.0
    assert u.sum() == pytest.approx(vk.sum() * (rm - rc) / (rm - rf), rel=2e-4)
    L, e = hd / np.tan(np.radians(dip)), t / np.tan(np.radians(dip))
    if e < L:                                                            # kinematic wedge area t (L - e/2)
        assert -vk.sum() * 100.0 == pytest.approx(t * (L - e / 2), abs=t * 100.0)


def test_reverse_postrift_vs_roberts1998():
    """Roberts et al. (1998, Fig. 4b vs 4c): with Te 1.5 km, beta 1.15 the Brent crest at 140 Ma restores much
    shallower than with Airy isostasy (they get ~sea level; we get ~200 m vs ~380 m, see notebook 8)."""
    xg, vt, vb, sec = _vg1()
    crest = {}
    for te in (1500.0, 0.0):
        rp = flex2d.reverse_postrift(flex2d.backstrip_section(sec, te), sec, 1.15, 155.0)
        i = list(rp["ages"]).index(140.0); m = (xg > 55e3) & (xg < 66e3)
        crest[te] = rp["seafloor"][i][m].min()
    assert 120 < crest[1500.0] < 280 and 320 < crest[0.0] < 450 and crest[0.0] - crest[1500.0] > 120


# ---------------------------------------------------------------- fault tips / cutoffs
def test_faults_from_cutoffs_tip_and_gating():
    xg, vt, vb, sec = _vg1()
    det, _ = F.detect_faults_from_horizons(xg, vt)
    cut = F.faults_from_cutoffs(det, xg, vt, vb, tip_unit=6)
    assert len(cut) == 5
    for f in cut:
        assert f.tip_unit == 6 and f.active(7) and not f.active(6) and not f.active(0)
        assert np.all(f.throw[:7] == 0)
        assert 15 < f.dip < 40                                            # gentler than the steep old picks
        ztip_horizon = np.interp(f.x0 if f.stick is None else np.asarray(f.stick)[0, 0], xg, vt[6])
        assert f.z_tip >= ztip_horizon - 50                              # tip not above the base post-rift
    r = flex2d.restore(sec, cut, "cantilever", Hd=8e3)
    slip = np.asarray(r["slip"])
    assert np.allclose(slip[:7], 0.0)                                    # no slip on post-rift horizons


def test_restoration2d_dict_access_and_ref():
    sec, fs = core.load_section()
    r = flex2d.restore(sec, fs, "listric", Hd=10e3)
    assert isinstance(r, flex2d.Restoration2D) and "top" in r and r["top"] is r.top and "steps" in r.keys()
    m0 = flex2d.restoration_metrics(r, 9, sec.x, fs)
    m1 = flex2d.restoration_metrics(r, 9, sec.x, fs, ref=sec.units[9])
    assert m0["std"] == pytest.approx(m1["std"])
    m2 = flex2d.restoration_metrics(r, 8, sec.x, [], ref=9)
    assert m2["mean_fault_offset"] == 0


# ---------------------------------------------------------------- apps
def test_streamlit_app_smoke():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pybasinsubs" / "apps" / "streamlit_app.py"), default_timeout=240).run()
    assert not at.exception and len(at.tabs) == 2
    at.selectbox(key="sec_method").set_value("domino").run()
    assert not at.exception and not at.error
    assert any("method = domino" in m.value for m in at.markdown)


def test_panel_layout_builds():
    pn = pytest.importorskip("panel")
    from pybasinsubs.ui import app, section_app
    a = app()
    assert isinstance(a, pn.Tabs) and len(a) == 2
    s = section_app(run_now=True)
    assert "5 faults" in s[1][0].object


# ---------------------------------------------------------------- notebooks (slow)
@pytest.mark.slow
@pytest.mark.parametrize("nb", sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb")))
def test_notebook_executes(nb, tmp_path):
    nbformat = pytest.importorskip("nbformat"); nbclient = pytest.importorskip("nbclient")
    book = nbformat.read(ROOT / "notebooks" / nb, as_version=4)
    nbclient.NotebookClient(book, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    _ = os
