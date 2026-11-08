# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Cached independent-column thermal histories, material support and causal Ro."""
from copy import deepcopy
import io
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from pybasinsubs import (app_core, section_thermal_data, plot_section_thermal_timestep,
                         save_section_thermal_animation, thermal)
from pybasinsubs.growth_spatial import reconstruct_spatial_growth
from pybasinsubs.webapp.routes import create_app


@pytest.fixture(scope="module")
def model():
    section, faults = app_core.load_section()
    out = reconstruct_spatial_growth(section, faults, make_figures=False)
    data = section_thermal_data(out, section, nsub=6)
    return section, out, data


def test_public_api_and_default_conditions(model):
    _, out, data = model
    assert callable(section_thermal_data) and callable(plot_section_thermal_timestep) and callable(save_section_thermal_animation)
    assert data["conditions"]["heat_flow_W_m2"] == .059
    assert data["conditions"]["surface_temperature_C"] == 4.
    assert data["conditions"]["bin_width_m"] == 1000.
    assert len(data["frames"]) == len(out.restoration.units) + 1 == 11


def test_only_surviving_sediment_has_ro_or_temperature(model):
    _, out, data = model
    r = out.restoration
    for stage in range(len(r.ages)):
        assert np.isnan(data["ro"][stage, :stage]).all()
        assert np.isnan(data["temperature"][stage, :stage]).all()
        for unit in range(stage, len(r.units)):
            z = data["depths"][stage, unit]
            ok = np.isfinite(data["ro"][stage, unit])
            assert np.all(z[ok] >= np.broadcast_to(r.top[stage, unit, :, None], z.shape)[ok])
            assert np.all(z[ok] <= np.broadcast_to(r.base[stage, unit, :, None], z.shape)[ok])
    assert np.isnan(data["frames"][-1]["ro"]).all()
    assert np.isnan(data["frames"][-1]["T"]).all()


def test_provenance_absent_unit_and_fraction_bounds(model):
    _, _, data = model
    for f in data["frames"]:
        d = data["fractions"]
        d = d[(d.frame == f["index"]) & (d.unit_index == f["selected_unit"])]
        assert d.fraction.between(0, 1).all()
        if f["selected_unit"] < f["stage"] or not f["supported"]:
            assert not d.fraction.any() and not d.supported_fraction.any()
    fractions = data["fractions"]
    assert fractions.fraction.between(0, 1).all()
    assert (fractions.groupby(["frame", "unit", "x_m"]).fraction.sum() <= 1. + 1e-12).all()
    # Buried VG1 units have genuine x variation, not a copied scalar rate.
    spreads = fractions.groupby(["frame", "unit", "window"]).fraction.agg(lambda a: a.max() - a.min())
    assert spreads.max() > .1


def test_windows_use_existing_easy_ro_not_temperature_cutoffs(model, monkeypatch):
    section, out, _ = model
    calls = []
    def fixed_ro(ages, temperatures, **kw):
        calls.append((ages.copy(), temperatures.copy()))
        return np.full(len(ages), 1.5)
    monkeypatch.setattr(thermal, "easy_ro", fixed_ro)
    data = section_thermal_data(out, section, nsub=2)
    assert calls and all(np.all(np.diff(a) < 0) for a, _ in calls)
    d = data["fractions"]
    supported = d.supported_fraction > 0
    assert (d[supported & (d.window == "wet gas")].fraction == 1.).all()
    assert not d[d.window != "wet gas"].fraction.any()
    assert data["temperature"][np.isfinite(data["temperature"])].max() > 100


def test_no_future_maturation(model):
    section, out, data = model
    altered = deepcopy(out)
    # Alter only modern burial, leaving every historical node identical.
    altered.restoration.top[0, 1:] += 2000
    altered.restoration.base[0] += 2000
    changed = section_thermal_data(altered, section, nsub=6)
    np.testing.assert_allclose(changed["ro"][1:], data["ro"][1:], equal_nan=True)
    assert not np.allclose(changed["ro"][0], data["ro"][0], equal_nan=True)


def test_fixed_axes_no_right_strip_and_gif(model, tmp_path, monkeypatch):
    _, _, data = model
    def forbidden(*a, **kw):
        raise AssertionError("cached plotting/export must not recompute kinetics")
    monkeypatch.setattr(thermal, "easy_ro", forbidden)
    axes = []
    for i in [0, 5, 9, 10]:
        fig = plot_section_thermal_timestep(data, i)
        assert len(fig.axes) == 4
        section, *fractions = fig.axes
        for fraction in fractions:
            assert fraction.get_ylim() == (-3., 103.)
            assert section.get_xlim() == fraction.get_xlim()
            assert fraction.get_position().width == section.get_position().width
            assert fraction.get_position().height < .15
        assert section.get_ylabel() == "depth (m)"
        assert not section.yaxis.label.get_wrap()
        assert not fractions[1].yaxis.label.get_wrap()
        assert fractions[1].get_ylabel() == "% in petroleum windows"
        axes.append((section.get_xlim(), section.get_ylim(), fig.get_size_inches().tolist()))
        plt.close(fig)
    assert all(a == axes[0] for a in axes)
    path = save_section_thermal_animation(data, tmp_path/"thermal.gif")
    with Image.open(path) as im:
        assert im.n_frames == 11 and im.info["loop"] == 0
        assert np.asarray(im.convert("RGB")).min() < 30  # preserve black text in shared palette
    for bad in ["thermal.mp4", "thermal.png"]:
        with pytest.raises(ValueError):
            save_section_thermal_animation(data, tmp_path/bad)


@pytest.mark.parametrize("kwargs", [{"heat_flow": 0}, {"bin_width_m": -1}, {"T_s": float("nan")},
                                    {"nsub": 1}, {"substep_Myr": 0}])
def test_invalid_controls(model, kwargs):
    section, out, _ = model
    with pytest.raises(ValueError):
        section_thermal_data(out, section, **kwargs)


def test_wizard_explicit_compute_and_optional_export(model, tmp_path, monkeypatch):
    from pybasinsubs import section_thermal as module
    section, out, data = model
    client = TestClient(create_app(testing=True, faults_path=tmp_path/"faults.json"))
    calls = []
    def compute(cached_out, cached_section, **kwargs):
        calls.append((cached_out, cached_section, kwargs))
        return data
    monkeypatch.setattr(module, "section_thermal_data", compute)
    # Keep this wiring test fast; actual numerical/restoration paths tested above.
    monkeypatch.setattr("pybasinsubs.growth_spatial.reconstruct_spatial_growth", lambda *a, **kw: out)
    def post(path, **body):
        response = client.post('/api/wizard/'+path, json=body)
        assert response.status_code == 200, response.text
        return response.json()
    opened = post('section/input')
    applied = post('section/compaction', input_id=opened['input_id'])
    run = post('section/run', model_id=applied['model_id'], n_mc=0, method='domino')
    assert not calls
    bad = client.post('/api/wizard/section/thermal', json=dict(growth_id=run['result_id']))
    assert bad.status_code == 400
    growth = post('section/growth', result_id=run['result_id'])
    assert not calls
    result = post('section/thermal', growth_id=growth['growth_id'])
    assert len(calls) == 1 and calls[0][2] == dict(heat_flow=.059, T_s=4.)
    assert len(result['frames']) == 11
    ident = result['thermal_id']
    def no_reconstruction(*a, **kw):
        raise AssertionError('No recomputation during export')
    monkeypatch.setattr(module, "section_thermal_data", no_reconstruction)
    monkeypatch.setattr("pybasinsubs.growth_spatial.reconstruct_spatial_growth", no_reconstruction)
    for include in [False, True]:
        response = client.get(f'/api/wizard/export/{ident}?section_thermal={str(include).lower()}')
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert ('section_thermal.gif' in archive.namelist()) == include
            pngs = [name for name in archive.namelist() if name.startswith('section_thermal_') and name.endswith('.png')]
            assert len(pngs) == (11 if include else 0)
    assert len(calls) == 1


def test_wizard_and_notebook_step_wiring():
    import json
    root = Path(__file__).resolve().parents[1]
    html = (root/'pybasinsubs/webapp/static/index.html').read_text()
    js = (root/'pybasinsubs/webapp/static/app.js').read_text()
    css = (root/'pybasinsubs/webapp/static/app.css').read_text()
    assert html.index('id="s_step4"') < html.index('id="s_step5"') < html.index('id="s_step6"')
    assert '5 · Thermal history' in html and '6 · Output & export' in html
    assert 'id="s_thermal_time"' in html and 'aria-orientation="vertical"' in html
    assert '#s_thermal_time' in css and 's_export_thermal' in html
    assert "action('s',5,'/api/wizard/section/thermal'" in js
    assert '$("s_thermal").disabled=!st.last||st.busy' in js
    assert "if(m==='s'&&from<=5){st.thermal=null;" in js
    notebook = json.loads((root/'notebooks/pyBasinSubs_8_2d_flexural_backstripping.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'section_thermal_data' in text and 'plot_section_thermal_timestep' in text and 'save_section_thermal_animation' in text
    assert 'type="range"' in text and '0–100' in text


def test_all_remaining_units_percent_lines(model):
    from matplotlib.colors import to_rgb
    from pybasinsubs.material_colors import section_material_colors
    _, out, data = model
    colors = section_material_colors(out.restoration.units)
    for index in [0, 6, 9, 10]:
        frame = data["frames"][index]
        table = data["fractions"]
        table = table[table.frame == index]
        units = sorted(table.loc[table.supported_fraction > 0, "unit_index"].unique(), reverse=True)
        fig = plot_section_thermal_timestep(data, index)
        for ax, name in zip(fig.axes[1:], ["oil", "wet gas", "dry gas"]):
            assert ax.get_title(loc="left") == name
            assert not ax.containers and not ax.patches
            assert len(ax.collections) == len(units)
            assert len(ax.lines) == len(units)
            assert ax.get_ylim() == (-3., 103.)
            assert [line.get_label() for line in ax.lines] == [out.restoration.units[u] for u in units]
            for layer, (line, unit) in enumerate(zip(ax.lines, units)):
                values = table[(table.unit_index == unit) & (table.window == name)].sort_values("x_m")
                assert unit >= frame["stage"]
                np.testing.assert_allclose(line.get_ydata(), values.fraction * 100.)
                np.testing.assert_allclose(line.get_xdata(), values.x_m / 1e3)
                assert line.get_alpha() == 1.
                assert line.get_zorder() == 3 + layer
                expected_rgb = to_rgb(colors[unit])
                np.testing.assert_allclose(line.get_color(), expected_rgb)
                shade = ax.collections[layer]
                assert shade.get_gid() == f"thermal-shade-unit-{unit}"
                assert shade.get_alpha() == .3
                np.testing.assert_allclose(shade.get_facecolor()[0], (*expected_rgb, .3))
                assert shade.get_zorder() == 2.5 + layer
                vertices = shade.get_paths()[0].vertices
                assert vertices[:, 1].min() == 0.
                np.testing.assert_allclose(vertices[:, 1].max(), (values.fraction * 100.).max())
        assert fig.axes[-1].get_xlabel() == "interpreted input distance (km)"
        assert not fig.axes[1].get_xlabel() and not fig.axes[2].get_xlabel()
        if index == 0:
            # Real along-x variation in multiple currently buried units.
            oil = fig.axes[1]
            assert sum(np.ptp(line.get_ydata()) > 0 for line in oil.lines) >= 2
        if index == 10:
            assert all(not ax.lines for ax in fig.axes[1:])
        plt.close(fig)


def test_translucent_thermal_artists_leave_growth_unchanged(model):
    from pybasinsubs.growth_display import draw_restored_section
    _, out, data = model
    fig = plot_section_thermal_timestep(data, 0)
    section = fig.axes[0]
    units = [c for c in section.collections if c.get_gid() == "thermal-unit-fill"]
    assert units and all(c.get_alpha() == .35 for c in units)
    windows = [c for c in section.collections if c.get_gid() != "thermal-unit-fill"]
    assert windows and any(c.get_alpha() == .20 for c in windows)
    assert any(line.get_gid() == "thermal-unit-contact" for line in section.lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    draw_restored_section(ax, out, 0)
    assert all(c.get_alpha() == .9 for c in ax.collections)
    plt.close(fig)
