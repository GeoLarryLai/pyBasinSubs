# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybasinsubs import faults
from pybasinsubs.fault_tracer import vg1_example
from pybasinsubs.growth_spatial import reconstruct_spatial_growth
from pybasinsubs.growth_display import (display_material_polygons, display_area_audit, restored_fault_traces,
                                        plot_spatial_growth_steps, plot_growth_subjects)
from pybasinsubs.material_colors import section_material_colors


@pytest.fixture(scope="module")
def output():
    sec, t, b = vg1_example();fs = faults.auto_faults("curved_midramp", sec.x, t, b)
    return reconstruct_spatial_growth(sec, fs, make_figures=False)


def test_complete_existing_mesh_cells_not_invented_gap_fill(output):
    audit = display_area_audit(output)
    assert audit.display_minus_source_area_m2.abs().max() < 1e-6
    assert audit.recovered_existing_cell_area_m2.max() > 0
    polygons = [p for p in display_material_polygons(output, 9) if p["unit_index"] == 9]
    assert polygons[0]["x_m"][0] == output.restoration.x[0]
    assert polygons[-1]["x_m"][-1] == output.restoration.x[-1]
    for a, b in zip(polygons[:-1], polygons[1:]):assert a["x_m"][-1] == b["x_m"][0]


def test_traces_slipped_top_clipped_and_deeper_nodes_retained(output):
    before = deepcopy(output.faults)
    for stage in [1, 6, 7, 8, 9]:
        d = restored_fault_traces(output, stage)
        assert set(d.fault) == {f.label for f in output.faults}
        assert (d.restored_z_m >= d.restored_top_m - 1e-6).all()
        present_top = output.restoration.top[0, stage]
        assert (d.original_z_m.to_numpy() >= np.interp(d.x_m, output.restoration.x, present_top) - 1e-6).all()
        assert (d.groupby("fault").original_z_m.max() - d.groupby("fault").original_z_m.min()).min() > 1.
        assert d.groupby("fault").size().min() >= 4
        if stage == 9:
            assert np.max(np.abs(d.groupby("fault").apply(lambda g: (g.restored_z_m - g.restored_top_m).min(), include_groups=False))) < 1e-6
            assert (np.abs(d.restored_z_m - d.original_z_m) > 100).any()
    for f, old in zip(output.faults, before):
        np.testing.assert_array_equal(f.curve, old.curve)
        assert f.tip_unit == old.tip_unit


def test_independent_axes_zero_line_and_marker_style(output):
    fig = plot_spatial_growth_steps(output)
    colors = section_material_colors(output.restoration.units)
    for j in range(10):
        section, left, right = fig.axes[3 * j:3 * j + 3]
        assert left.get_ylabel() == "throw rate\n(m/Myr)"
        assert right.get_ylabel() == "sed. rate\n(m/Myr)"
        assert left.yaxis.label.get_color() == right.yaxis.label.get_color() == colors[j]
        assert left.get_ylim() != right.get_ylim()
        assert right.get_position().width == pytest.approx(section.get_position().width)
        if j < 6:
            assert any("throw = 0" in t.get_text() for t in left.texts)
            assert left.get_ylim()[0] < 0 < left.get_ylim()[1]
        left_lines = [ln for ln in left.lines if ln.get_marker() == "o"]
        right_lines = [ln for ln in right.lines if ln.get_marker() == "s"]
        if j < 9:assert left_lines[0].get_markerfacecolor() == "white"
        assert right_lines[0].get_markerfacecolor() == colors[j]
        assert right_lines[0].get_markeredgecolor() == colors[j]
    plt.close(fig)


def test_individual_subjects_do_not_refit(output, monkeypatch):
    from pybasinsubs import growth_spatial
    def no_fit(*args, **kwargs):raise AssertionError("Plotting must not refit throw")
    monkeypatch.setattr(growth_spatial, "reconstruct_spatial_growth", no_fit)
    fields = output.restoration["interval_throw_field_m"].copy()
    figures = plot_growth_subjects(output)
    assert set(figures) == {"every_unit_sections_rates", "throw_rates", "sedimentation_rates", "restored_sections", "flattening_history"}
    np.testing.assert_array_equal(fields, output.restoration["interval_throw_field_m"])
    for fig in figures.values():plt.close(fig)


def test_saved_notebooks_embed_ordered_inputs_frames_and_tables():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ["pyBasinSubs_5_own_data_and_UI.ipynb", "pyBasinSubs_8_2d_flexural_backstripping.ipynb", "pyBasinSubs_9_fault_interpretation.ipynb"]:
        nb = json.loads((root / "notebooks" / name).read_text())
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "flattening_history" not in text and "vg1_fault_throw_rates" not in text
        assert "plot_growth_subjects" not in text  # duplicate standalone / long-stack galleries deleted
        start = next(i for i, c in enumerate(nb["cells"]) if "## 2-D analysis order" in "".join(c["source"]))
        codes = [c for c in nb["cells"][start:] if c["cell_type"] == "code" and "analysis-order" in c.get("metadata", {}).get("tags", [])]
        assert len(codes) == 4 and all(c["execution_count"] is not None for c in codes)
        assert "interpreted_section(" in "".join(codes[0]["source"])
        assert "compaction_relations(" in "".join(codes[1]["source"])
        assert "plot_growth_frames(" in "".join(codes[2]["source"])
        assert "analysis_growth.history" in "".join(codes[3]["source"])
        assert [sum("image/png" in o.get("data", {}) for o in c["outputs"]) for c in codes] == [1, 1, 11, 0]
        assert all(not any(o.get("output_type") == "error" for o in c["outputs"]) for c in codes)


def test_square_fault_throw_history_uses_same_partition_range(output, monkeypatch):
    from matplotlib.collections import PolyCollection
    from pybasinsubs import growth_display
    import pandas as pd
    before = output.spatial.copy(deep=True)
    def no_bins(*args, **kwargs):raise AssertionError("Summary must aggregate cached fields, not bin/refit")
    monkeypatch.setattr(growth_display, "_bins", no_bins)
    fig = growth_display.plot_growth_subjects(output, subjects=("flattening_history",))["flattening_history"]
    assert tuple(fig.get_size_inches()) == (6., 6.) and len(fig.axes) == 1 and fig.dpi == 180
    fig.canvas.draw();ax = fig.axes[0];box = ax.get_window_extent()
    assert box.width == pytest.approx(box.height)
    assert all(np.ptp(line.get_ydata()) == 0 for line in ax.lines)  # zero reference only
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)];assert len(bands) == 5
    table = growth_display.fault_throw_ranges(output.spatial, [f.label for f in output.faults])
    assert set(table.label) == {f"Fault #{j}" for j in range(1, 6)}
    for k, band in enumerate(bands, 1):
        d = table[table.fault_number == k]
        vertices = band.get_paths()[0].vertices
        for row in d.itertuples():
            original = output.spatial[(output.spatial.unit == row.unit) & (output.spatial.block == row.fault)].throw_m_Myr
            assert row.mean_rate_m_Myr == pytest.approx(original.mean())
            ys = vertices[vertices[:, 0] == row.age_Ma, 1]
            assert ys.min() == pytest.approx(original.min())
            assert ys.max() == pytest.approx(original.max())
    assert ax.get_ylabel() == "Throw rate (m/Myr)"
    assert not any("auto-" in t.get_text() for t in ax.get_legend().texts)
    assert not (table.unit == output.restoration.units[-1]).any()
    pd.testing.assert_frame_equal(output.spatial, before);plt.close(fig)


def test_single_frames_have_identical_geometry_axes_and_optional_numbers(output, monkeypatch):
    from pybasinsubs import growth_display, growth_spatial
    def no_fit(*args, **kwargs):raise AssertionError("Frame rendering must not refit")
    monkeypatch.setattr(growth_spatial, "reconstruct_spatial_growth", no_fit)
    bins = growth_display._bins(output, (), 1000.)
    figs = growth_display.plot_growth_frames(output, bins=bins)
    assert len(figs) == 11
    positions = [[ax.get_position().bounds for ax in fig.axes] for fig in figs]
    for fig, pos in zip(figs, positions):
        assert tuple(fig.get_size_inches()) == (12.5, 5.5) and len(fig.axes) == 3
        np.testing.assert_allclose(pos, positions[0])
        for ax, ref in zip(fig.axes, figs[0].axes):
            assert ax.get_xlim() == ref.get_xlim() and ax.get_ylim() == ref.get_ylim()
        assert {t.get_text() for t in fig.axes[0].texts if t.get_text().startswith("#")} == {f"#{i}" for i in range(1, 6)}
    hidden = growth_display.plot_growth_frame(output, 8, bins=bins, show_fault_numbers=False)
    assert not any(t.get_text().startswith("#") for t in hidden.axes[0].texts)
    assert any("not inferred" in t.get_text() for t in figs[-1].axes[1].texts)
    for fig in figs + [hidden]:plt.close(fig)


def test_animation_is_fixed_frame_gif_from_existing_results(output, tmp_path):
    from PIL import Image
    from pybasinsubs.growth_display import save_growth_animation
    filename = save_growth_animation(output, tmp_path / "frames.gif", fps=2)
    with Image.open(filename) as image:
        assert image.size == (1375, 605) and image.n_frames == 11
        assert image.info["duration"] == 500


def test_frame_caption_honors_disabled_datum_target(output):
    from copy import copy
    from pybasinsubs.growth_display import plot_growth_frame
    trial=copy(output);trial.metadata=dict(output.metadata,assumed_datum_m=None)
    fig=plot_growth_frame(trial,7)
    assert any('datum target off' in t.get_text() for t in fig.texts)
    assert not any('assumed zero' in t.get_text() for t in fig.texts)
    plt.close(fig)
