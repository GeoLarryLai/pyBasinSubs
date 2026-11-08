# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Exact-result timestep geometry, stable rendering, slider and GIF contracts."""
import io

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np
from PIL import Image
import pytest

from pybasinsubs import (
    backstrip, from_sodir, timestep_column, timestep_steps, timestep_thermal_fields,
    plot_geohistory_timestep, plot_thermal_timestep, timestep_view,
    save_geohistory_animation, save_thermal_animation,
)
from pybasinsubs import plotting


@pytest.fixture(scope='module')
def result():
    return backstrip(from_sodir('30/4-1'), mc=False)


@pytest.fixture(scope='module')
def fields(result):
    return timestep_thermal_fields(result, heat_flow=.059)


def test_present_stack_matches_input_and_engine(result):
    state = timestep_column(result)
    u = result.column.units
    assert len(state['layers']) == (u.thickness_m > 0).sum()
    np.testing.assert_allclose([v['thickness_m'] for v in state['layers']], u.thickness_m[u.thickness_m > 0], rtol=1e-8)
    assert state['surface_m'] == pytest.approx(result.best['W'][0, -1] - result.best['SL'][0, -1])
    assert state['layers'][-1]['base_m'] == pytest.approx(result.best['sigma'][0, -1])


def test_oldest_empty_with_paleo_surface_and_water(result):
    state = timestep_column(result, 0)
    assert state['layers'] == []
    assert state['age_Ma'] == result.column.units.age_base_Ma.max()
    assert state['surface_m'] == pytest.approx(100.)
    assert state['water_depth_m'] == pytest.approx(100.)
    fig = plot_geohistory_timestep(result, 0)
    ax = next(a for a in fig.axes if a.get_label() == 'strata')
    assert len(ax.patches) == 1
    assert ax.patches[0].get_height() == pytest.approx(100.)
    assert any(np.allclose(line.get_ydata(), [100., 100.]) for line in ax.lines)
    plt.close(fig)


def test_intermediate_strips_younger_and_uses_existing_decompaction(result):
    k = next(s['index'] for s in timestep_steps(result) if s['age_Ma'] == 140)
    state = timestep_column(result, k)
    assert len(state['layers']) < len(timestep_column(result)['layers'])
    for layer in state['layers']:
        j = layer['unit_index']
        assert result.column.units.age_top_Ma.iloc[j] >= 140
        assert layer['top_m'] == pytest.approx(result.best['tops'][0, k, j] + state['surface_m'])
        assert layer['thickness_m'] == result.best['thick'][0, k, j]
    assert any(v['thickness_m'] > result.column.units.thickness_m.iloc[v['unit_index']] for v in state['layers'])


def test_nodes_are_constraints_with_actual_age_spacing(result):
    steps = timestep_steps(result)
    assert [s['age_Ma'] for s in steps] == list(result.best['age'][0])
    assert set(result.column.units.age_top_Ma) <= {s['age_Ma'] for s in steps}
    assert 15 in [s['age_Ma'] for s in steps]  # hiatus/deposition-start boundary
    fig, slider = timestep_view(result)
    np.testing.assert_array_equal(slider.valstep, np.sort(-result.best['age'][0]))
    slider.set_val(-140)
    assert '140 Ma' in fig._suptitle.get_text()
    # The same-age nodes need explicit ordered keyboard navigation.
    fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, key='left'))
    assert '166 Ma' in fig._suptitle.get_text()
    plt.close(fig)


def test_fixed_limits_common_column_style_and_gradient_alignment(result, fields):
    limits = []
    positions = []
    for k in [0, 5, -1]:
        for function, kwargs in [(plot_geohistory_timestep, {}), (plot_thermal_timestep, {'fields': fields})]:
            fig = function(result, k, **kwargs)
            axes = {a.get_label(): a for a in fig.axes}
            limits.append(axes['strata'].get_ylim())
            positions.append(axes['strata'].get_position().bounds)
            assert axes['history'].get_ylim() == axes['strata'].get_ylim()
            if 'gradient' in axes:
                assert axes['gradient'].get_ylim() == axes['strata'].get_ylim()
                assert axes['gradient'].get_position().x0 > axes['strata'].get_position().x1
            plt.close(fig)
    assert all(v == limits[0] for v in limits)
    assert all(v == positions[0] for v in positions)
    assert positions[0][2] < .05


def test_highlight_nodes_and_half_alpha_age_line(result):
    k = 5
    fig = plot_geohistory_timestep(result, k)
    ax = next(a for a in fig.axes if a.get_label() == 'history')
    lines = [v for v in ax.lines if v.get_linestyle() == '--' and v.get_alpha() == .5]
    assert len(lines) == 1 and lines[0].get_xdata() == [result.best['age'][0, k]] * 2
    highlights = [v for v in ax.lines if v.get_markerfacecolor() == 'white']
    assert len(highlights) == 2
    assert {v.get_ydata()[0] for v in highlights} == {result.best['sigma'][0, k], result.best['zeta'][0, k]}
    plt.close(fig)


def test_subaerial_no_water_and_negative_depth():
    col = from_sodir('30/4-1')
    col.units['water_depth_m'] = -150.
    col.units['sea_level_m'] = 25.
    result = backstrip(col, mc=False)
    state = timestep_column(result, 0)
    assert state['water_depth_m'] == 0 and state['surface_m'] == -175
    assert state['ylim'][1] < -175
    fig = plot_geohistory_timestep(result, 0)
    ax = next(a for a in fig.axes if a.get_label() == 'strata')
    assert not ax.patches
    plt.close(fig)


def test_missing_basal_marker_gets_empty_start_without_mutating_solver(result):
    from pybasinsubs import Column
    col = Column(result.column.units.iloc[:-1].copy(), result.column.lithologies.copy(), name='without TD')
    r = backstrip(col, mc=False)
    nodes = list(r.best['nodes'])
    state = timestep_column(r, 0)
    assert state['node_index'] is None and state['age_Ma'] == 199 and not state['layers']
    assert r.best['nodes'] == nodes


def test_duplicate_erosion_events_remain_ordered():
    r = backstrip(from_sodir('7120/9-2', 'barents_hammerfest'), mc=False)
    steps = timestep_steps(r)
    events = [s for s in steps if s['age_Ma'] == 2.7]
    assert len(events) >= 2
    layers = [len(timestep_column(r, s['index'])['layers']) for s in events]
    assert max(layers) > min(layers)


def test_gradient_uses_node_conductivity_and_kinetic_windows(result, fields):
    from pybasinsubs import thermal
    fig = plot_thermal_timestep(result, fields=fields)
    ax = next(a for a in fig.axes if a.get_label() == 'gradient')
    z, _, k, _ = thermal.column_profile(result, below=0)
    temp = thermal.steady_geotherm(z, k, .059, 4.)
    actual_z = ax.lines[0].get_ydata() - timestep_column(result)['surface_m']
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.interp(actual_z, z, temp))
    assert len(ax.collections) == 3
    assert any('oil 0.6–1.3 %Ro' in t.get_text() for leg in fig.legends for t in leg.get_texts())
    plt.close(fig)


def _png(fig):
    data = io.BytesIO(); fig.savefig(data, format='png'); plt.close(fig)
    return data.getvalue()


def test_original_static_helpers_are_unchanged_by_new_views(result, fields):
    def originals():
        fig, ax = plt.subplots(); plotting.geohistory(result, ax=ax)
        assert len(fig.axes) == 1
        geo = _png(fig)
        fig, ax = plt.subplots(); plotting.geohistory_thermal(result, fields['temperature'], fields['maturity'], ax=ax)
        assert len(fig.axes) == 1
        thermo = _png(fig)
        fig = plotting.maturity_plot([2., 1., 0.], np.ones((3, 1))*70, np.ones((3, 1)), ['unit'])
        assert len(fig.axes) == 2
        return geo, thermo, _png(fig)
    before = originals()
    for fn, kw in [(plot_geohistory_timestep, {}), (plot_thermal_timestep, {'fields': fields})]:
        plt.close(fn(result, **kw))
    assert originals() == before


@pytest.mark.parametrize('thermal_view', [False, True])
def test_package_gifs_match_nodes_and_keep_fixed_frame_size(tmp_path, result, fields, thermal_view):
    fn = save_thermal_animation if thermal_view else save_geohistory_animation
    kw = {'fields': fields} if thermal_view else {}
    figures_before = plt.get_fignums()
    path = fn(result, tmp_path/'history.gif', fps=2, dpi=45, **kw)
    with Image.open(path) as image:
        assert image.n_frames == len(timestep_steps(result))
        size = image.size
        for i in range(image.n_frames):
            image.seek(i); assert image.size == size
        assert image.info['duration'] == 500
    assert plt.get_fignums() == figures_before


def test_animation_validation(result, tmp_path):
    with pytest.raises(ValueError, match='.gif'):
        save_geohistory_animation(result, tmp_path/'x.mp4')
    with pytest.raises(ValueError, match='fps'):
        save_geohistory_animation(result, tmp_path/'x.gif', fps=0)


@pytest.fixture(scope='module')
def maturity_data(result):
    from pybasinsubs import maturity_timestep_data
    return maturity_timestep_data(result, heat_flow=.059)


@pytest.mark.parametrize('step', [0, 5, -1])
def test_windows_always_visible_and_labeled_without_fake_depths(result, fields, step):
    fig = plot_thermal_timestep(result, step, fields=fields)
    ax = next(a for a in fig.axes if a.get_label() == 'gradient')
    names = {'oil', 'wet gas', 'dry gas'}
    assert names <= {t.get_text() for t in ax.texts}
    keys = [p for p in ax.patches if (p.get_gid() or '').startswith('window-key-')]
    assert len(keys) == 3
    assert {p.get_gid().removeprefix('window-key-') for p in keys} == names
    assert all(p.get_data_transform() == ax.transAxes and p.get_alpha() > 0 for p in keys)
    assert any(t.get_text() == 'key' for t in ax.texts)
    if step == 0:
        assert not [c for c in ax.collections if (c.get_gid() or '').startswith('kinetic-window-')]
    plt.close(fig)


@pytest.mark.parametrize('step', [0, 5, -1])
def test_two_panel_overlay_full_curves_and_matching_blue_units(result, fields, maturity_data, step):
    from pybasinsubs import plot_maturity_timestep, thermal_unit_colors
    colors = thermal_unit_colors(result)
    thermal_fig = plot_thermal_timestep(result, step, fields=fields)
    history = next(a for a in thermal_fig.axes if a.get_label() == 'history')
    horizons = {int(v.get_gid().split('-')[-1]): v for v in history.lines if (v.get_gid() or '').startswith('unit-top-')}
    fig = plot_maturity_timestep(result, step, data=maturity_data)
    state = timestep_column(result, step); age = state['age_Ma']
    for ax, key in zip(fig.axes, ['temperature', 'ro']):
        guides = [v for v in ax.lines if v.get_gid() == 'timestep-age']
        assert len(guides) == 1 and guides[0].get_alpha() == .5 and guides[0].get_linestyle() == '--'
        np.testing.assert_array_equal(guides[0].get_xdata(), [age, age])
        curves = [v for v in ax.lines if (v.get_gid() or '').startswith('unit-curve-')]
        row = np.flatnonzero(np.isclose(maturity_data['age'], age))[-1]
        expected = {uid for j, uid in enumerate(maturity_data['unit_indices']) if np.isfinite(maturity_data[key][row, j])}
        markers = [v for v in ax.lines if (v.get_gid() or '').startswith('unit-highlight-')]
        assert {int(v.get_gid().split('-')[-1]) for v in markers} == expected
        for curve in curves:
            uid = int(curve.get_gid().split('-')[-1])
            j = maturity_data['unit_indices'].index(uid)
            assert curve.get_color() == colors[uid] == horizons[uid].get_color()
            np.testing.assert_array_equal(curve.get_xdata(), maturity_data['age'])
            np.testing.assert_allclose(curve.get_ydata(), maturity_data[key][:, j], equal_nan=True)
        for marker in markers:
            uid = int(marker.get_gid().split('-')[-1])
            assert marker.get_markerfacecolor() == 'white' and marker.get_markeredgecolor() == colors[uid]
    assert fig.axes[1].get_yscale() == 'log' and fig.axes[1].get_ylim() == (.2, 4.5)
    assert fig.axes[0].get_title(loc='left') == 'Burial temperature'
    assert fig.axes[1].get_title(loc='left') == 'Thermal maturity (log scale)'
    assert len(fig.axes[1].patches) == 3
    plt.close(thermal_fig); plt.close(fig)


def test_slider_false_uses_track_space_but_default_geohistory_is_unchanged(result, fields):
    plain_before = _png(plot_geohistory_timestep(result))
    for fn, kwargs in [(plot_geohistory_timestep, {}), (plot_thermal_timestep, {'fields': fields})]:
        default = fn(result, **kwargs)
        expanded = fn(result, slider=False, **kwargs)
        da = {a.get_label(): a for a in default.axes}
        ea = {a.get_label(): a for a in expanded.axes}
        assert 'time' in da and 'time' not in ea
        assert ea['history'].get_position().height > da['history'].get_position().height
        assert ea['strata'].get_position().bounds[1:] == pytest.approx((.16, .045, .74))
        plt.close(default); plt.close(expanded)
    plain = plot_geohistory_timestep(result)
    history = next(a for a in plain.axes if a.get_label() == 'history')
    assert not any((v.get_gid() or '').startswith('unit-top-') for v in history.lines)
    assert history.lines[0].get_color() == 'k'  # original unit-base outline
    assert plain.get_size_inches() == pytest.approx([12, 6])
    plt.close(plain)
    assert _png(plot_geohistory_timestep(result)) == plain_before


def test_maturity_cache_reuses_burial_and_original_static_remains_plasma(result, monkeypatch):
    from pybasinsubs import thermal, maturity_timestep_data, plot_maturity_timestep, app_core
    bt = thermal.burial_temperatures(result, heat_flow=.059)
    fig, table = app_core.maturity_figure(result, bt)
    expected_colors = [plt.get_cmap('plasma')(i / 7) for i in range(8)]
    assert [v.get_color() for v in fig.axes[0].lines] == expected_colors
    original = _png(fig)
    def forbidden(*a, **kw):
        raise AssertionError('Must reuse supplied burial data')
    monkeypatch.setattr(thermal, 'burial_temperatures', forbidden)
    data = maturity_timestep_data(result, burial=bt)
    for step in [0, -1]:
        plt.close(plot_maturity_timestep(result, step, data=data))
    fig, again = app_core.maturity_figure(result, bt)
    assert _png(fig) == original
    np.testing.assert_allclose(data['temperature'][-1], table.present_T_C, equal_nan=True)
    np.testing.assert_allclose(data['ro'][-1], again.present_Ro, equal_nan=True)


def test_equal_age_maturity_overlays_are_same_while_columns_differ():
    from pybasinsubs import maturity_timestep_data, plot_maturity_timestep
    r = backstrip(from_sodir('7120/9-2', 'barents_hammerfest'), mc=False)
    data = maturity_timestep_data(r)
    events = [s for s in timestep_steps(r) if s['age_Ma'] == 2.7]
    assert len(events) >= 2
    pngs = [_png(plot_maturity_timestep(r, e['index'], data=data)) for e in events]
    assert all(p == pngs[0] for p in pngs)
    assert timestep_column(r, events[0]['index'])['layers'] != timestep_column(r, events[1]['index'])['layers']


def test_maturity_gif_exact_nodes_fixed_size_and_no_recompute(result, maturity_data, tmp_path, monkeypatch):
    from pybasinsubs import save_maturity_animation, thermal
    def forbidden(*a, **kw):
        raise AssertionError('Export should reuse cached histories')
    monkeypatch.setattr(thermal, 'burial_temperatures', forbidden)
    before = plt.get_fignums()
    path = save_maturity_animation(result, tmp_path/'maturity.gif', data=maturity_data, dpi=50)
    with Image.open(path) as image:
        assert image.n_frames == len(timestep_steps(result))
        assert image.size == (550, 220) and image.info['duration'] == 1000
    assert plt.get_fignums() == before
    with pytest.raises(ValueError, match='fps'):
        save_maturity_animation(result, tmp_path/'bad.gif', data=maturity_data, fps=0)
    with pytest.raises(ValueError, match='.gif'):
        save_maturity_animation(result, tmp_path/'bad.mp4', data=maturity_data)


def test_maturity_gif_preserves_identical_overlay_equal_age_events(tmp_path):
    from pybasinsubs import maturity_timestep_data, save_maturity_animation
    r = backstrip(from_sodir('7120/9-2', 'barents_hammerfest'), mc=False)
    data = maturity_timestep_data(r)
    path = save_maturity_animation(r, tmp_path/'erosion.gif', data=data, dpi=45)
    with Image.open(path) as image:
        assert image.n_frames == len(timestep_steps(r))
