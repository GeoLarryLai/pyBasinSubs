# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from copy import deepcopy
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybasinsubs import faults
from pybasinsubs.fault_tracer import vg1_example
from pybasinsubs.growth import reconstruct_growth_history
from pybasinsubs.growth_contacts import (tighten_block_contacts, binned_growth_rates,
                                         plot_growth_layer_steps, reconstruct_contact_history)


@pytest.fixture(scope="module")
def outputs():
    sec, tops, base = vg1_example();fs = faults.auto_faults("curved_midramp", sec.x, tops, base)
    old = reconstruct_growth_history(sec, fs, make_figures=False, search_sweeps=3, max_nfev=80)
    return sec, fs, old, tighten_block_contacts(old)


def test_contact_constraints_and_no_z_change(outputs):
    _, _, old, out = outputs
    np.testing.assert_array_equal(old.restoration.top, out.restoration.top)
    np.testing.assert_array_equal(old.restoration.base, out.restoration.base)
    np.testing.assert_array_equal(old.scores.datum_rmse_m, out.scores.datum_rmse_m)
    np.testing.assert_array_equal(old.scores.smooth_rms_m, out.scores.smooth_rms_m)
    np.testing.assert_array_equal(old.restoration["material_x_top"][:7], out.restoration["material_x_top"][:7])
    assert out.metadata["previous_maximum_overlap_m"] == pytest.approx(3620.316666, abs=.01)
    assert out.metadata["maximum_overlap_m"] < 1e-7
    assert out.metadata["maximum_gap_m"] == 1250.  # unchanged inactive-stage mesh spacing, not new missing strata
    assert out.metadata["active_stage_maximum_gap_m"] < 400.
    active = out.contacts[out.contacts.active_restoration_stage]
    assert active.envelope_overlap_m_after.max() < 1e-7
    assert active.envelope_gap_m_after.max() < 1e-7
    assert out.contacts.sidewall_separation_m_after.max() < .6 * out.contacts.sidewall_separation_m_before.max()
    assert out.contacts.sidewall_separation_m_after.max() > 1400.  # honest vertical mismatch remains
    assert out.metadata["clipped_area_m2"] == out.metadata["filled_gap_area_m2"] == 0
    assert out.metadata["contact_auxiliary_translation_is_fault_slip"] is False
    assert out.history.throw_m_Myr.dropna().min() >= 0


def test_area_preserving_rigid_translation(outputs):
    _, fs, old, out = outputs
    x = old.restoration.x;masks = [x < fs[0].x0] + old.restoration["material_blocks"]
    for i in range(len(out.restoration.ages)):
        for b, m in enumerate(masks):
            for key in ["material_x_top", "material_x_base"]:
                np.testing.assert_allclose(out.restoration[key][i, :, m] - old.restoration[key][i, :, m],
                                           out.restoration["contact_translation_m"][i, b], atol=1e-9)
    # Same x shift for every vertex + identical z = area-preserving rigid motion.
    assert not np.any(old.restoration["material_x_top"] == np.inf)


def test_every_layer_rate_bins_and_no_invented_oldest_slip(outputs):
    sec, _, _, out = outputs
    d = binned_growth_rates(out)
    assert set(d[d.kind == "sedimentation"].unit) == set(sec.units)
    assert sec.units[-1] not in set(d[d.kind == "fault_throw"].unit)
    assert (d.rate_m_Myr >= 0).all()
    assert (d.range_low_m_Myr <= d.rate_m_Myr).all() and (d.range_high_m_Myr >= d.rate_m_Myr).all()
    assert set(d[d.kind == "sedimentation"].errorbar) == {"within-bin spatial min/max"}
    fig = plot_growth_layer_steps(out)
    assert len(fig.axes) == 30
    assert "every unit" in fig._suptitle.get_text()
    plt.close(fig)
    with pytest.raises(ValueError):binned_growth_rates(out, bin_width_m=0)


def test_no_live_file_or_trace_mutation(outputs):
    sec, fs, _, _ = outputs;original = deepcopy(fs)
    root = Path(__file__).resolve().parents[1] / "pybasinsubs" / "example_data"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    for f, orig in zip(fs, original):
        assert (f.x0, f.tip_unit, f.z_tip) == (orig.x0, orig.tip_unit, orig.z_tip)
        np.testing.assert_array_equal(f.curve, orig.curve)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    with pytest.raises(ValueError):reconstruct_contact_history(sec, fs, tip_candidates=[3.5], make_figures=False)
    with pytest.raises(ValueError):reconstruct_contact_history(sec, fs, throw_rates_m_Myr=-1., make_figures=False)


def test_primary_contact_export_returns_notebook_figures(outputs, monkeypatch):
    from pybasinsubs import growth_contacts
    sec, fs, old, _ = outputs
    monkeypatch.setattr(growth_contacts, "reconstruct_growth_history", lambda *a, **kw: old)
    out = growth_contacts.reconstruct_contact_history(sec, fs)
    assert set(out.figures) == {"history", "space_time", "comparison", "every_unit"}
    assert len(out.figures["every_unit"].axes) == 30
    for fig in out.figures.values():plt.close(fig)
