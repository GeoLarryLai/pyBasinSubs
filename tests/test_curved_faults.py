# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Curved fault traces (faults.faults_from_curves) and their use by the restorations."""
import numpy as np
import pandas as pd
import pytest

from pybasinsubs import faults as F, faultfit as FF, flex2d
from pybasinsubs._paths import data_path

TIP = 6


@pytest.fixture(scope="module")
def vg1():
    dv = pd.read_csv(data_path("north_sea", "roberts1998_fig4a_vg1_digitized.csv"))
    xg = np.arange(1000.0, 81001.0, 250.0)
    Iv = lambda c: np.interp(xg, dv.x_km * 1e3, dv[c])
    vt = np.array([Iv(f"top{L}_m") for L in range(1, 11)]); vb = Iv("base_m")
    raw, _ = F.detect_faults_from_horizons(xg, vt)
    cv = F.faults_from_curves(raw, xg, vt, vb, tip_unit=TIP, Hd=10e3, trace_anchor="footwall_crest")
    units = [f"u{k}" for k in range(10)]; ages = [0, 5.3, 23.0, 56.0, 66.0, 100.5, 140.0, 157.3, 166.1, 170.3]
    lith = [{"shale": 1}] * 8 + [{"sandstone": 1}, {"shaly_sandstone": 1}]
    sec = flex2d.section_from_horizons(xg, vt, vb, units, ages, ages[1:] + [252.0], lith, vt[0])
    return xg, vt, vb, raw, cv, sec


def test_curve_model_monotone_and_flattening():
    z = np.linspace(1000, 10000, 200)
    x = F.curve_x(z, 5000.0, 1000.0, 40.0, 5000.0, down=1, dip_inf=8.0)
    assert np.all(np.diff(x) > 0)
    d = F.curve_dip(z, 1000.0, 40.0, 5000.0, 8.0)
    assert d[0] == pytest.approx(40.0) and np.all(np.diff(d) < 0) and d[-1] > 8.0


def test_fit_recovers_synthetic_curve():
    z = np.linspace(2000, 7000, 12)
    x = F.curve_x(z, 10e3, 2000.0, 45.0, 6000.0, down=-1, dip_inf=8.0)
    p = F.fit_fault_curve(np.c_[x, z], np.ones_like(z), -1, dip_inf=8.0)
    assert p["dip0"] == pytest.approx(45.0, abs=1.5) and p["rms_x"] < 20.0


def test_vg1_curves(vg1):
    xg, vt, vb, raw, cv, _ = vg1
    assert len(cv) == 5
    for f in cv:
        c = f.curve
        assert c is not None and np.all(np.diff(c[:, 1]) > 0)                     # depth increases monotonically
        assert np.all(f.down * np.diff(c[:, 0]) >= 0)                              # one-way (no overturn)
        assert c[-1, 1] == pytest.approx(10e3)
        zt = np.interp(c[:, 0], xg, vt[TIP])
        assert np.all(c[:, 1] >= zt - 5.0)                                         # never above the 140 Ma horizon
        assert f.z_tip == pytest.approx(np.interp(c[0, 0], xg, vt[TIP]), abs=30.0)  # tip on the top syn-rift
        assert f.dip_at(f.z_tip) > f.dip_at(8000.0)                                # flattens with depth
        assert np.all(f.throw[:TIP + 1] == 0)
        tab = f.cutoffs["table"]                                                   # passes between / through the cutoffs
        mid = 0.5 * (tab[["x_fw", "z_fw"]].to_numpy() + tab[["x_hw", "z_hw"]].to_numpy())
        assert np.median(np.abs(f.x_at(mid[:, 1]) - mid[:, 0])) < 2000.0
        zb = tab.z_hw.max(); xb = tab.x_hw[tab.z_hw.idxmax()]                       # deepest yellow block corner
        assert abs(float(f.x_at(zb)) - xb) < 2000.0
        assert f.curve_params["rms_x"] < 1100.0


def test_footwall_crest_anchor(vg1):
    """trace_anchor='footwall_crest': the curve passes exactly through the footwall-crest cutoff of top
    Triassic, which is the trace x0; above it the syn-rift face dip, below it a flattening curve."""
    xg, vt, vb, raw, cv, _ = vg1
    for f in cv:
        tab = f.cutoffs["table"]; row = tab[tab.horizon == "top9"].iloc[0]
        assert f.x0 == pytest.approx(row.x_fw)
        assert float(f.x_at(row.z_fw)) == pytest.approx(row.x_fw, abs=1.0)
        p = f.curve_params
        assert p["anchor"] == pytest.approx((row.x_fw, row.z_fw))
        assert p["dip_up"] >= p["dip0"] - 1e-9
        if f.z_tip < row.z_fw - 100:                       # an upper segment exists: it has the face dip
            assert f.dip_at(0.5 * (f.z_tip + row.z_fw)) == pytest.approx(p["dip_up"], abs=0.5)
    # perturbation and dip shifts keep the anchor
    for g in F.perturb_curved_faults(cv, np.random.default_rng(3), xg, vt) + FF._with_dips(cv, [f.dip - 3 for f in cv]):
        xa, za = g.curve_params["anchor"]
        assert float(g.x_at(za)) == pytest.approx(xa, abs=1.0) and g.x0 == pytest.approx(xa)


def test_midramp_option(vg1):
    xg, vt, vb, raw, cv, _ = vg1
    mr = F.faults_from_curves(raw, xg, vt, vb, tip_unit=TIP, Hd=10e3)  # default = midramp
    assert all(f.curve_params["anchor"] is None for f in mr)
    # without the anchor the two western faults are steep at shallow depth (face picks ~38-39 deg)
    assert 33.0 < mr[0].dip_at(mr[0].z_tip) < 50.0 and 33.0 < mr[1].dip_at(mr[1].z_tip) < 50.0
    assert mr[0].x0 > cv[0].x0 + 1000.0                    # mid-ramp trace lies down-dip of the crest
    with pytest.raises(ValueError):
        F.faults_from_curves(raw, xg, vt, vb, tip_unit=TIP, trace_anchor="nope")


def test_user_stick_passes_through_unchanged(vg1):
    xg, vt, vb, raw, cv, _ = vg1
    st = np.array([[20e3, 3000.0], [21e3, 4200.0], [22.5e3, 5200.0]])
    f = flex2d.fault_at(xg, vt, 20.5e3, down=1, dip=45.0)
    f.stick = st; f.source = "imported: test"
    out = F.faults_from_curves([f], xg, vt, vb, tip_unit=TIP, Hd=10e3)[0]
    np.testing.assert_allclose(out.curve[:3], st)
    assert out.curve[-1, 1] == pytest.approx(10e3)
    np.testing.assert_allclose(out.x_at(st[:, 1]), st[:, 0])


@pytest.mark.parametrize("method,kw", [("cantilever", dict(Hd=8e3)), ("listric", dict(alpha=65.0)), ("domino", {})])
def test_restorations_use_curves(vg1, method, kw):
    xg, vt, vb, raw, cv, sec = vg1
    KW = dict(Te=1500.0, beta=1.15, rift_end_Ma=155.0, fault_end_Ma=140.0)
    r = flex2d.restore(sec, cv, method=method, **KW, **kw)
    m = flex2d.restoration_metrics(r, 9, xg, cv)
    assert np.isfinite(m["std"]) and m["std"] < 2000
    import dataclasses
    rp = flex2d.restore(sec, [dataclasses.replace(f, curve=None) for f in cv], method=method, **KW, **kw)
    if method == "domino":                     # domino uses only the effective planar dip
        np.testing.assert_allclose(r["top"][9][9], rp["top"][9][9])
    else:                                      # cantilever (local dip, curve reach) and listric (the curve) differ
        assert not np.allclose(r["top"][9][9], rp["top"][9][9], equal_nan=True)


def test_shift_dip_and_perturb(vg1):
    xg, vt, vb, raw, cv, _ = vg1
    g = FF._with_dips(cv, [f.dip + 5 for f in cv])
    for a, b in zip(cv, g):
        xa, za = a.curve_params["anchor"]
        assert b.dip == pytest.approx(a.dip + 5) and float(b.x_at(za)) == pytest.approx(xa, abs=1.0)
        assert b.curve_params["dip0"] > a.curve_params["dip0"]
    p = F.perturb_curved_faults(cv, np.random.default_rng(0), xg, vt)
    assert any(abs(a.curve_params["dip0"] - b.curve_params["dip0"]) > 0.1 for a, b in zip(cv, p))
    xs, zs = FF.fault_plane(cv[0], 10e3, "listric", xg, vt)
    np.testing.assert_allclose(xs, cv[0].x_at(zs))
