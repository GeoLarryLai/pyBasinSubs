# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Full-length fault planes, regularized dip fit, flexural coupling of kinematic restorations."""
import numpy as np
import pytest

from pybasinsubs import faultfit as FF
from pybasinsubs import fault_tracer, flex2d
from pybasinsubs import faults as F

KW = dict(Te=1500.0, beta=1.15, rift_end_Ma=155.0, fault_end_Ma=140.0)


@pytest.fixture(scope="module")
def vg1():
    sec, tops, base = fault_tracer.vg1_example()
    det, _ = F.detect_faults_from_horizons(sec.x, tops)
    return sec, tops, base, det


def test_planar_plane_reaches_detachment(vg1):
    sec, tops, base, det = vg1
    f = det[0]
    xs, zs = FF.fault_plane(f, 10e3, "planar", sec.x, tops)
    assert zs[0] == 0 and zs[-1] == pytest.approx(10e3)
    x0, z0 = FF._anchor(f, sec.x, tops)
    assert np.interp(z0, zs, xs) == pytest.approx(x0, abs=1.0)
    assert np.degrees(np.arctan(np.ptp(zs) / abs(np.ptp(xs)))) == pytest.approx(f.dip, abs=0.01)
    assert zs[-1] > base.max()                                  # extends below the deepest (Triassic) unit


def test_listric_plane_flattens_into_detachment(vg1):
    sec, tops, base, det = vg1
    xs, zs = FF.fault_plane(det[2], 10e3, "listric", sec.x, tops)
    assert zs.max() < 10e3 and zs.max() > 0.98 * 10e3 - 1e3
    dz = np.diff(zs) / np.abs(np.diff(xs))
    assert dz[-1] < 0.1 * dz[len(dz) // 3]                     # near-horizontal at depth


def test_flexural_coupling_footwall_lowered_on_restoration():
    x = np.arange(0, 80e3, 250.0)
    d = np.where((x > 30e3) & (x < 45e3), 800.0 * (45e3 - x) / 15e3, 0.0)
    corr, c = flex2d._flexural_coupling(x, d, 1500.0)
    assert c >= 1.0
    assert corr[(x > 25e3) & (x < 29e3)].mean() > 0             # footwall goes down when slip is removed
    corr_stiff, c_stiff = flex2d._flexural_coupling(x, d, 1e6)  # very stiff plate: little isostatic response
    assert abs(c_stiff - 1) < abs(c - 1) and np.abs(corr_stiff).max() < np.abs(corr).max()


def test_relief_modes(vg1):
    sec, tops, base, det = vg1
    a = flex2d.restore(sec, det, method="domino", relief_mode="none", **KW)
    b = flex2d.restore(sec, det, method="domino", bathy_relief=False, **KW)
    assert np.allclose(a["top"][9][9], b["top"][9][9], equal_nan=True)
    c = flex2d.restore(sec, det, method="domino", relief_mode="climax", relief_fraction=0.0, **KW)
    assert np.allclose(a["top"][9][9], c["top"][9][9], equal_nan=True)
    with pytest.raises(ValueError):
        flex2d.restore(sec, det, method="domino", relief_mode="bogus", **KW)


def test_domino_flexural_runs(vg1):
    sec, tops, base, det = vg1
    r = flex2d.restore(sec, det, method="domino", flexural=True, **KW)
    assert all(s["c"] >= 1.0 for s in r["steps"]) and np.isfinite(r["top"][9][9]).all()


def test_fit_fault_geometry_improves_and_reports_intervals(vg1):
    sec, tops, base, det = vg1
    fit = FF.fit_fault_geometry(sec, det, "cantilever", Hd=8e3, bounds=(30.0, 70.0), grid_step=10.0, sweeps=1, **KW)
    assert fit["std_after"] <= fit["std_before"] + 1e-6 or fit["prior_penalty"] > 0
    assert fit["n_eff"] < len(sec.x)                            # residual autocorrelation reduces the sample count
    for d, (lo, hi) in zip(fit["dips"], fit["ci"]):
        assert 30.0 - 5 <= lo <= d <= hi <= 70.0 + 5
    tab = FF.fit_table(fit)
    assert list(tab.columns) == ["x0_km", "dip_input", "dip_fit", "ci_lo", "ci_hi"]


def test_ensemble_shapes(vg1):
    sec, tops, base, det = vg1
    tab, Z = FF.restoration_ensemble(sec, [dict(label="c", method="cantilever", faults=det, kw=dict(Hd=8e3))],
                                     Te_values=(1000.0, 3000.0), beta=1.15, rift_end_Ma=155.0, fault_end_Ma=140.0)
    assert len(tab) == 2 and Z.shape == (2, len(sec.x))
    assert abs(np.nanmean(Z[0])) < 1e-6
