# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from copy import deepcopy
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybasinsubs import faults, flex2d
from pybasinsubs.fault_tracer import vg1_example
from pybasinsubs.growth_spatial import reconstruct_spatial_growth, binned_spatial_rates, plot_spatial_growth_steps


@pytest.fixture(scope="module")
def spatial():
    sec, t, b = vg1_example();fs = faults.auto_faults("curved_midramp", sec.x, t, b)
    return sec, fs, reconstruct_spatial_growth(sec, fs, make_figures=False)


def test_nonnegative_actual_spatial_and_temporal_variation(spatial):
    sec, fs, out = spatial
    field = out.restoration["interval_throw_field_m"]
    assert field.min() >= 0
    np.testing.assert_array_equal(field[:6], 0.)  # fixed tips: no throw through 140 Ma
    for m in out.restoration["material_blocks"]:assert np.ptp(field[6, m]) > 50.
    assert out.spatial[out.spatial.unit == sec.units[-1]].throw_m_Myr.isna().all()
    assert out.history[out.history.unit == sec.units[-1]].throw_m.isna().all()
    assert out.history.dip_slip_m.isna().all() and out.history.heave_m.isna().all()
    assert out.metadata["along_fault_down_dip_resolved"] is False
    assert out.metadata["maximum_placement_translation_m"] == 0
    assert out.metadata["maximum_interval_geological_throw_proxy_m"] > 1000.
    assert [f.tip_unit for f in fs] == [6] * 5
    np.testing.assert_allclose(out.spatial.throw_m, out.spatial.throw_m_Myr * out.spatial.duration_Myr)


def test_flattening_and_inactive_stage_scores(spatial):
    _, _, out = spatial
    oldest = out.scores[out.scores.age_Ma == 170.3]
    previous_rmse = np.array([78.04, 215.88, 263.66, 99.62, 263.22])
    assert np.all(oldest.datum_rmse_m.to_numpy() < previous_rmse)
    assert oldest.datum_rmse_m.max() < 70.
    hw = out.restoration.x >= out.faults[0].x0
    assert np.maximum(-out.restoration.top[-1, -1, hw], 0).max() < 200.  # do not trade hidden face spikes for interior fit
    np.testing.assert_allclose(out.scores[out.scores.age_Ma == 140.].smooth_rms_m,
                               [116.805103, 64.155136, 155.111016, 122.788881, 219.786347], atol=1e-5)
    assert out.metadata["datum_is_observed"] is False
    assert out.search.converged.all()


def test_vertical_shear_preserves_thickness_and_x(spatial):
    sec, _, out = spatial
    base = flex2d.restore_faulted(sec, [], Te=1500., Hd=8000., beta=1.15, rift_end_Ma=155.)
    np.testing.assert_allclose(out.restoration.base - out.restoration.top, base.base - base.top, atol=1e-9)
    np.testing.assert_array_equal(out.restoration["material_x_top"], np.broadcast_to(sec.x, base.top.shape))
    np.testing.assert_array_equal(out.restoration.top[:7], base.top[:7])
    assert out.contacts.envelope_overlap_m_after.max() == 0
    assert out.metadata["filled_gap_area_m2"] == out.metadata["clipped_area_m2"] == 0


def test_fine_dual_axis_aligned_layout(spatial):
    sec, _, out = spatial
    bins = binned_spatial_rates(out)
    assert len(bins[bins.unit == sec.units[6]]) > 130
    assert (bins.x_bin_hi_m - bins.x_bin_lo_m).max() == 1000.
    assert (bins.range_low_m_Myr <= bins.rate_m_Myr).all()
    assert (bins.range_high_m_Myr >= bins.rate_m_Myr).all()
    fig = plot_spatial_growth_steps(out)
    assert len(fig.axes) == 30
    for j in range(10):
        section, rate, right = fig.axes[3 * j:3 * j + 3]
        assert rate.get_shared_x_axes().joined(section, rate)
        assert rate.get_position().width == pytest.approx(section.get_position().width)
        assert rate.get_position().height < section.get_position().height / 2
        assert rate.get_position().x0 == pytest.approx(section.get_position().x0)
        assert right.get_position().bounds == pytest.approx(rate.get_position().bounds)
        assert "throw rate" in rate.get_ylabel() and "sed. rate" in right.get_ylabel()
    assert "NOT confidence intervals" in fig._suptitle.get_text()
    plt.close(fig)


def test_supplied_spatial_rates_and_no_input_mutation(spatial):
    sec, fs, _ = spatial;original = deepcopy(fs);original_h = sec.h.copy()
    root = Path(__file__).resolve().parents[1] / "pybasinsubs" / "example_data"
    paths = [p for p in root.rglob("*") if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rates = np.full((9, len(sec.x)), np.nan);rates[6] = 2. + np.sin(sec.x / 5000.)
    out = reconstruct_spatial_growth(sec, fs, throw_rates_m_Myr=rates, make_figures=False)
    m = sec.x >= fs[0].x0
    np.testing.assert_allclose(out.restoration["interval_throw_field_m"][6, m], rates[6, m] * (157.3 - 140.))
    np.testing.assert_array_equal(sec.h, original_h)
    for f, orig in zip(fs, original):
        assert (f.x0, f.tip_unit, f.z_tip) == (orig.x0, orig.tip_unit, orig.z_tip)
        np.testing.assert_array_equal(f.curve, orig.curve)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for bad in [-1., np.inf]:
        with pytest.raises(ValueError):reconstruct_spatial_growth(sec, fs, throw_rates_m_Myr=bad, make_figures=False)
    with pytest.raises(ValueError):binned_spatial_rates(out, bin_width_m=0)


def test_fault1_footwall_flattened_without_changing_hanging_walls():
    from pybasinsubs import app_core as core
    model, fs = core.load_section(None, None, None)
    new = reconstruct_spatial_growth(model, fs, make_figures=False)
    old = reconstruct_spatial_growth(model, fs, make_figures=False, footwall_flattening=False)
    x = new.restoration.x;fw = x < new.faults[0].x0;last = len(model.units) - 1
    # Hanging-wall partitions (faults 1-5) and present day are byte-identical.
    np.testing.assert_array_equal(new.restoration["interval_throw_field_m"][:, ~fw], old.restoration["interval_throw_field_m"][:, ~fw])
    np.testing.assert_array_equal(new.restoration.top[0], old.restoration.top[0])
    assert np.all(old.restoration["interval_throw_field_m"][:, fw] == 0)
    # Old: fault #1 footwall Triassic left ~1.5 km deep (detached-looking); new: near datum.
    zone = fw & (x > 6500)
    assert np.max(old.restoration.top[last, last, zone]) > 1000
    assert np.max(np.abs(new.restoration.top[last, last, zone])) < 400
    assert new.metadata["footwall_flattening"] and np.all(np.isnan(new.spatial.throw_m[np.tile(fw, len(model.units))]))
