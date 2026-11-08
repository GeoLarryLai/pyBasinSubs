# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import hashlib
from pathlib import Path

import numpy as np
import pytest

from pybasinsubs import faults as F, flex2d
from pybasinsubs.fault_tracer import vg1_example
from pybasinsubs.growth import block_surface_scores, explore_growth_history, structural_history


@pytest.fixture(scope="module")
def vg1_growth():
    sec, tops, base = vg1_example()
    fs = F.auto_faults("curved_midramp", sec.x, tops, base)
    return sec, tops, base, fs


def test_history_rates_and_sediment(vg1_growth):
    sec, _, _, fs = vg1_growth
    out = structural_history(sec, fs, make_figures=False)
    h = out.history
    for increment, rate in [("throw_m", "throw_m_Myr"), ("heave_m", "heave_m_Myr"),
                            ("dip_slip_m", "dip_slip_m_Myr"), ("rotation_deg", "rotation_deg_Myr")]:
        np.testing.assert_allclose(h[increment], h[rate] * h.duration_Myr, equal_nan=True)
    assert len(h) == 10 * 5
    assert h[h.unit == sec.units[-1]].throw_m.isna().all()
    assert len(out.sediment) == 10 * 6
    np.testing.assert_allclose(out.sediment.accumulation_m,
                               out.sediment.accumulation_m_Myr * out.sediment.duration_Myr)
    assert out.metadata["datum_is_observed"] is False
    # A shorter interval must have a higher rate for the same increment.
    assert 100 / 5.3 > 100 / 34.5


def test_block_scores_do_not_penalize_different_block_depths(vg1_growth):
    sec, _, _, fs = vg1_growth
    z = np.zeros(len(sec.x))
    bounds = [f.x0 for f in fs] + [sec.x[-1] + 1]
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        z[(sec.x >= a) & (sec.x < b)] = 100 * (k + 1)
    score = block_surface_scores(sec.x, z[None], fs, [170.3], reference=np.zeros((1, len(z))))
    np.testing.assert_allclose(score.smooth_rms_m, 0)
    assert (score.datum_rmse_m > 0).all()
    # Offset-free block tilt is still penalized by the within-block score.
    tilt = z + .02 * sec.x
    s2 = block_surface_scores(sec.x, tilt[None], fs, [170.3], reference=np.zeros((1, len(z))))
    assert (s2.smooth_rms_m > 0).all()
    s3 = block_surface_scores(sec.x, z[None], fs, [170.3], reference=np.zeros((1, len(z))), assumed_datum_m=None)
    assert s3.datum_rmse_m.isna().all()


def test_mid_palaeocene_cap(vg1_growth):
    sec, _, _, fs = vg1_growth
    for bad in [3., 3.49, -1., np.nan, 9.]:
        with pytest.raises(ValueError, match="Tip cap"):
            explore_growth_history(sec, fs, tip_candidates=[bad], make_figures=False)


def test_export_does_not_change_defaults_or_geometry(vg1_growth):
    sec, tops, base, fs = vg1_growth
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "pybasinsubs" / "example_data").rglob("*"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    fbefore = [(f.x0, f.tip_unit, f.z_tip, f.throw.copy(), np.asarray(f.curve).copy()) for f in fs]
    hbefore = sec.h.copy();tbefore = tops.copy();bbefore = base.copy()
    structural_history(sec, fs, make_figures=False)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    np.testing.assert_array_equal(sec.h, hbefore);np.testing.assert_array_equal(tops, tbefore);np.testing.assert_array_equal(base, bbefore)
    for f, (x0, tip, ztip, throw, curve) in zip(fs, fbefore):
        assert (f.x0, f.tip_unit, f.z_tip) == (x0, tip, ztip)
        np.testing.assert_array_equal(f.throw, throw);np.testing.assert_array_equal(f.curve, curve)


def test_cantilever_history_is_explicitly_bulk_dip(vg1_growth):
    sec, _, _, fs = vg1_growth
    r = flex2d.restore(sec, fs, method="cantilever", Hd=8000.)
    out = structural_history(sec, fs, method="cantilever", restoration=r, make_figures=False)
    assert out.history.rotation_kind.str.contains("not rigid rotation").all()
    h = out.history[out.history.unit != sec.units[-1]]
    np.testing.assert_allclose(h.throw_m.to_numpy().reshape(9, 5), np.diff(r["slip"], axis=0))


def test_fixed_trace_primary_and_supplied_rates(vg1_growth):
    from pybasinsubs.growth import reconstruct_growth_history
    sec, _, _, fs = vg1_growth
    before = [(f.tip_unit, f.z_tip, np.array(f.curve, copy=True)) for f in fs]
    out = reconstruct_growth_history(sec, fs, throw_rates_m_Myr=2., max_nfev=2, search_sweeps=1, make_figures=False)
    assert out.metadata["geometry_mode"] == "fixed interpreted traces (primary workflow)"
    assert out.metadata["final_objective"] <= out.metadata["initial_objective"] + 1e-8
    for f, returned, (tip, ztip, curve) in zip(fs, out.faults, before):
        assert (f.tip_unit, f.z_tip) == (tip, ztip)
        np.testing.assert_array_equal(f.curve, curve);np.testing.assert_array_equal(returned.curve, curve)
    h = out.history
    for j, unit in enumerate(sec.units[:-1]):
        expected = 2. if j >= 6 else 0.
        np.testing.assert_allclose(h[h.unit == unit].throw_m_Myr, expected)
    assert len(out.spatial) == len(sec.units) * len(sec.x)
    np.testing.assert_allclose(out.spatial.accumulation_m,
                               out.spatial.accumulation_m_Myr * out.spatial.duration_Myr)
    # Rigid polygons retain their area: no overlap trimming or gap interpolation.
    r = out.restoration
    reference = flex2d.restore_faulted(sec, [], Hd=8000.)
    def area(x1, z1, x2, z2):
        xx = np.r_[x1, x2[::-1]];zz = np.r_[z1, z2[::-1]]
        return .5 * abs(np.dot(xx, np.roll(zz, 1)) - np.dot(zz, np.roll(xx, 1)))
    for i in [6, 9]:
        for mask in r["material_blocks"]:
            a0 = area(sec.x[mask], reference.top[i, i, mask], sec.x[mask], reference.base[i, i, mask])
            a1 = area(r["material_x_top"][i, i, mask], r.top[i, i, mask],
                      r["material_x_base"][i, i, mask], r.base[i, i, mask])
            assert a1 == pytest.approx(a0, rel=1e-10, abs=1e-5)
    assert np.nanmin(r.base - r.top) >= -1e-8


def test_tip_fraction_uses_dataset_age(vg1_growth):
    from pybasinsubs.growth import _tip_age, _tip_copy
    sec, tops, _, fs = vg1_growth
    assert _tip_age(sec, 3.5) == 61.
    assert _tip_age(sec, 4.5) == 83.25
    f = _tip_copy(sec, fs[0], 3.5)
    cap = tops[3] + .5 * sec.h[3]
    assert f.z_tip == pytest.approx(np.interp(f.curve[0, 0], sec.x, cap))
    assert fs[0].tip_unit == 6 and f.tip_unit == 3
