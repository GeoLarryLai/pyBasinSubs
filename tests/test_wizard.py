# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Guided app snapshots, actual optional fits and cached-result exports."""
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pybasinsubs import from_sodir, app_core
from pybasinsubs.webapp.routes import create_app
from pybasinsubs.webapp.wizard import configure_compaction


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(testing=True, faults_path=tmp_path/'faults.json'))


def post(client, url, body):
    response = client.post('/api/wizard/'+url, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_real_point_per_lithology_shared_and_sparse_fallback():
    col = from_sodir('30/4-1'); before = col.lithologies.copy()
    z = np.array([100, 500, 1000, 2000, 3000.])
    obs = pd.DataFrame(dict(depth_m=z, porosity=.52*np.exp(-z/2400), lithology='shale')).to_csv(index=False)
    fitted, pts, table, notes = configure_compaction(col, points=obs)
    assert fitted.lithologies.loc['shale','phi0'] == pytest.approx(.52)
    assert fitted.lithologies.loc['shale','C'] == pytest.approx(2400)
    assert fitted.lithologies.loc['chalk','C'] == col.lithologies.loc['chalk','C']
    shared, _, _, _ = configure_compaction(col, points=obs, fit_mode='shared')
    assert np.allclose(shared.lithologies.loc[shared.lith_names, 'phi0'], .52)
    sparse = pd.read_csv(io.StringIO(obs)).iloc[:2].to_csv(index=False)
    sparse_model, shown, _, notes = configure_compaction(col, points=sparse)
    pd.testing.assert_frame_equal(sparse_model.lithologies, before)
    assert len(shown) == 2 and 'retained' in notes[0]
    flat = pd.DataFrame(dict(depth_m=z,porosity=[.3]*len(z),lithology='shale')).to_csv(index=False)
    model, _, _, notes = configure_compaction(col, points=flat)
    pd.testing.assert_frame_equal(model.lithologies, before)
    pd.testing.assert_frame_equal(col.lithologies, before)
    plt.close('all')


def test_imported_parameters_are_applied_and_validate():
    col = from_sodir('30/4-1')
    model, _, _, _ = configure_compaction(col, curves='lithology,phi0,C\nshale,0.4,2300\n')
    assert model.lithologies.loc['shale','C'] == 2300
    for bad in ['lithology,phi0,C\nshale,1.2,100\n', 'lithology,phi0,C\nunknown,0.3,100\n']:
        with pytest.raises(ValueError):configure_compaction(col,curves=bad)


def test_column_snapshot_fit_is_used_thermal_reuses_result_and_export(client,monkeypatch):
    opened=post(client,'column/input',dict(source='North Viking Graben well (Sodir)',well='30/4-1'))
    assert opened['figure'] and 'column.csv' in opened['files']
    points='depth_m,porosity,lithology\n'+''.join(f'{z},{.5*np.exp(-z/2500)},shale\n' for z in [100,500,1000,2000])
    model=post(client,'column/compaction',dict(input_id=opened['input_id'],points=points))
    original=app_core.run_case;seen=[]
    def tracked(col,*args,**kwargs):
        seen.append(col.lithologies.loc['shale','C'])
        return original(col,*args,**kwargs)
    monkeypatch.setattr(app_core,'run_case',tracked)
    run=post(client,'column/run',dict(model_id=model['model_id'],n=5))
    assert seen == pytest.approx([2500])
    thermal=post(client,'column/thermal',dict(result_id=run['result_id'],heat_flow_mW_m2=59,surface_temperature_C=4))
    assert len(seen)==1 and thermal['figure'] and '59' in thermal['note']
    archive=client.get('/api/wizard/export/'+thermal['thermal_id'])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        assert {'column.csv','compaction.csv','geohistory.png','thermal.png','temperature_history.csv','maturity_history.csv','input_column.png','compaction.png'} <= set(z.namelist())
        assert not any('flattening' in p or 'fault_throw' in p for p in z.namelist())
        params=pd.read_csv(z.open('compaction.csv')).set_index('lithology')
        assert params.loc['shale','C']==pytest.approx(2500)
    history=post(client,'column/thermal',dict(result_id=run['result_id'],geothermal_csv='age_Ma,heat_flow_mW_m2\n0,59\n200,65\n'))
    assert 'Imported heat-flow history' in history['note']
    bht=post(client,'column/thermal',dict(result_id=run['result_id'],geothermal_csv='depth_m,temperature_C\n2000,65\n3000,95\n'))
    assert 'imported temperature/depth' in bht['note']
    assert client.post('/api/wizard/column/run',json=dict(model_id=opened['input_id'])).status_code==400


@pytest.mark.parametrize('placement,zero',[('auto',True),('off',False),('tighten',True)])
def test_section_explicit_growth_toggles_and_export(client,monkeypatch,placement,zero):
    from pybasinsubs import growth_spatial
    original=growth_spatial.reconstruct_spatial_growth;seen=[]
    def tracked(*args,**kwargs):
        seen.append(kwargs.copy());return original(*args,**kwargs)
    monkeypatch.setattr(growth_spatial,'reconstruct_spatial_growth',tracked)
    opened=post(client,'section/input',{})
    model=post(client,'section/compaction',dict(input_id=opened['input_id']))
    run=post(client,'section/run',dict(model_id=model['model_id'],method='domino',n_mc=0))
    assert not seen
    growth=post(client,'section/growth',dict(result_id=run['result_id'],contact_placement=placement,assume_zero=zero))
    assert len(seen)==3 and all(p['contact_placement']==placement for p in seen)
    assert all(p['assumed_datum_m']==(0 if zero else None) for p in seen)
    assert len(growth['frames'])==11 and growth['metadata']['contact_placement_option']==placement
    assert not growth['metadata']['along_fault_down_dip_resolved'] and not growth['metadata']['default_files_modified']
    if placement=='auto':
        response=client.get('/api/wizard/export/'+growth['growth_id']+'?animation=true')
        assert response.status_code==200
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            assert 'restoration.gif' in z.namelist() and len([p for p in z.namelist() if p.startswith('timestep_')])==11
        assert len(seen)==3  # exports do not rerun inversion


def test_point_plot_does_not_claim_fitted_parameters_were_unchanged():
    from pybasinsubs.input_plots import compaction_relations
    col=from_sodir('30/4-1')
    fig=compaction_relations(col, measured=[dict(depth_m=100,porosity=.4)],
                             measured_note="Real observations shown; curves reflect applied fitting/import choices.")
    assert any('applied fitting' in t.get_text() for t in fig.texts)
    assert not any('model parameters unchanged' in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_plane_placement_is_explicit_and_input_snapshot_is_preserved(client,monkeypatch):
    original=app_core.load_section;seen=[]
    def tracked(*args,**kw):
        seen.append(kw.get('cutoff_planes',False));return original(*args,**kw)
    monkeypatch.setattr(app_core,'load_section',tracked)
    plain=post(client,'section/input',{})
    placed=post(client,'section/input',dict(cutoff_planes=True))
    assert seen == [False,True] and plain['figure'] != placed['figure']
    assert plain['files']['faults.csv'] == placed['files']['faults.csv']


def test_curve_plot_choices_do_not_remove_used_model_lithologies(client, monkeypatch):
    opened = post(client, 'column/input', dict(source='North Viking Graben well (Sodir)', well='30/4-1'))
    used = {v['name'] for v in opened['curve_options'] if v['used']}
    assert used and 'mudstone' not in used
    result = post(client, 'column/compaction', dict(input_id=opened['input_id'], plot_curves=['mudstone']))
    assert set(pd.DataFrame(result['table']['data'], columns=result['table']['columns']).lithology) == used
    assert result['plot_curves'] == ['mudstone']
    from pybasinsubs.webapp.wizard import model_rows
    seen = []
    original = app_core.run_case
    def tracked(col, *a, **kw):
        seen.append(set(model_rows(col).lithology));return original(col, *a, **kw)
    monkeypatch.setattr(app_core, 'run_case', tracked)
    post(client, 'column/run', dict(model_id=result['model_id'], n=0))
    assert seen == [used]
    assert client.post('/api/wizard/column/compaction', json=dict(input_id=opened['input_id'], plot_curves=['not_available'])).status_code == 400
    assert client.post('/api/wizard/column/compaction', json=dict(input_id=opened['input_id'], plot_curves='shale')).status_code == 400
    blank = post(client, 'column/compaction', dict(input_id=opened['input_id'], plot_curves=[]))
    assert blank['plot_curves'] == [] and len(blank['table']['data']) == len(used)


def test_imported_reference_options_and_used_defaults(client):
    opened = post(client, 'section/input', {})
    used = {v['name'] for v in opened['curve_options'] if v['used']}
    csv = 'lithology,phi0,C\ncustom_comparison,0.4,2400\n'
    options = post(client, 'section/curve-options', dict(input_id=opened['input_id'], curves=csv))
    assert set(options['selected']) == used | {'custom_comparison'}
    ref = next(v for v in options['curve_options'] if v['name'] == 'custom_comparison')
    assert not ref['used'] and ref['C'] == 2400
    model = post(client, 'section/compaction', dict(input_id=opened['input_id'], curves=csv, plot_curves=['custom_comparison']))
    assert {v['name'] for v in model['curve_options'] if v['used']} == used
    assert 'custom_comparison' not in pd.DataFrame(model['table']['data'], columns=model['table']['columns']).lithology.tolist()
    assert any('plot-only' in note for note in model['notes'])
    bad = 'lithology,phi0,C\nbad,0.4,-2\n'
    assert client.post('/api/wizard/section/curve-options', json=dict(input_id=opened['input_id'], curves=bad)).status_code == 400


def test_used_imported_curve_is_not_mislabeled_as_library_or_reference(client):
    opened = post(client, 'column/input', dict(source='North Viking Graben well (Sodir)', well='30/4-1'))
    csv = 'lithology,phi0,C\nshale,0.5,2500\n'
    choices = post(client, 'column/curve-options', dict(input_id=opened['input_id'], curves=csv))
    assert 'reference' not in next(v['label'] for v in choices['curve_options'] if v['name']=='shale')
    applied = post(client, 'column/compaction', dict(input_id=opened['input_id'], curves=csv))
    row = next(v for v in applied['curve_options'] if v['name']=='shale')
    assert row['used'] and row['C']==2500 and 'opened/applied' in row['label'] and 'baseline Sclater' in row['label']


def test_column_timesteps_current_snapshot_and_optional_exports(client, monkeypatch):
    from pybasinsubs import thermal
    from pybasinsubs.webapp import wizard
    frame_calls = []
    for name in ['plot_geohistory_timestep', 'plot_thermal_timestep']:
        original_plot = getattr(wizard, name)
        def tracked_plot(*args, _fn=original_plot, **kwargs):
            assert kwargs.get('slider') is False
            fig = _fn(*args, **kwargs)
            assert 'time' not in {ax.get_label() for ax in fig.axes}
            frame_calls.append(1)
            return fig
        monkeypatch.setattr(wizard, name, tracked_plot)
    original_burial = thermal.burial_temperatures
    burial_calls = []
    def tracked_burial(*args, **kwargs):
        burial_calls.append(1)
        return original_burial(*args, **kwargs)
    monkeypatch.setattr(thermal, 'burial_temperatures', tracked_burial)
    opened = post(client, 'column/input', dict(source='North Viking Graben well (Sodir)', well='30/4-1'))
    model = post(client, 'column/compaction', dict(input_id=opened['input_id']))
    run = post(client, 'column/run', dict(model_id=model['model_id'], n=0))
    frames = run['frames']
    assert frames[0]['age_Ma'] == 199 and not frames[0]['layers']
    assert frames[-1]['age_Ma'] == 0 and len(frames[-1]['layers']) == 8
    assert len({tuple(f['ylim']) for f in frames}) == 1
    assert all(f['figure'] for f in frames)
    def forbidden(*a, **kw):
        raise AssertionError('Thermal/export must not recompute backstripping')
    monkeypatch.setattr(app_core, 'run_case', forbidden)
    therm = post(client, 'column/thermal', dict(result_id=run['result_id'], heat_flow_mW_m2=59))
    assert [f['age_Ma'] for f in therm['frames']] == [f['age_Ma'] for f in frames]
    assert therm['frames'][-1]['layers'] == frames[-1]['layers']
    assert all(f['maturity_figure'] for f in therm['frames'])
    assert therm['frames'][0]['maturity_figure'] != therm['frames'][-1]['maturity_figure']
    assert not any('maturity_figure' in f for f in frames)
    assert len(frame_calls) == len(frames) * 2 and len(burial_calls) == 1
    monkeypatch.setattr(thermal, 'burial_temperatures', forbidden)
    response = client.get('/api/wizard/export/'+therm['thermal_id']+'?animation=true')
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert {'geohistory.png', 'thermal.png', 'geohistory_history.gif', 'thermal_history.gif', 'maturity_history.gif', 'column_timesteps.json'} <= names
        assert len([n for n in names if n.startswith('geohistory_timestep_')]) == len(frames)
        assert len([n for n in names if n.startswith('thermal_timestep_')]) == len(frames)
        assert len([n for n in names if n.startswith('maturity_timestep_')]) == len(frames)
        import base64
        assert archive.read('thermal.png') == base64.b64decode(therm['figure'])
        from PIL import Image
        with Image.open(io.BytesIO(archive.read('maturity_history.gif'))) as image:
            assert image.n_frames == len(frames)
    response = client.get('/api/wizard/export/'+run['result_id'])
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert not any(n.endswith('.gif') for n in archive.namelist())
    response = client.get('/api/wizard/export/'+run['result_id']+'?animation=true')
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert 'geohistory_history.gif' in archive.namelist()
        assert 'thermal_history.gif' not in archive.namelist()
