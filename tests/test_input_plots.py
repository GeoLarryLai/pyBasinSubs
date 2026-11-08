"""Input figures reflect the model without changing data or inventing measurements."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pybasinsubs.column import Column
from pybasinsubs.input_plots import compaction_relations, interpreted_section, stratigraphic_column


def sample():
    return Column(pd.DataFrame(dict(unit=['upper', 'lower', 'base'], age_top_Ma=[0., 20., 50.],
                                     age_base_Ma=[20., 50., 50.], thickness_m=[100., 200., 0.],
                                     water_depth_m=[20., 300., 100.], f_shale=[1., 1., 1.])))


def test_column_boundaries_water_orientation_and_no_mutation():
    col = sample(); before = col.units.copy(deep=True)
    fig = stratigraphic_column(col); left, water = fig.axes
    assert left.get_shared_y_axes().joined(left, water)
    assert np.allclose(left.get_yticks(), [0, 100, 300])
    assert np.allclose(water.lines[0].get_ydata(), [0, 100, 300])
    assert water.get_xlim()[0] > water.get_xlim()[1]
    assert left.get_ylim()[0] > left.get_ylim()[1]
    assert any('50 Ma' in t.get_text() for t in left.texts)
    pd.testing.assert_frame_equal(col.units, before)
    plt.close(fig)


def test_eroded_units_do_not_add_present_thickness():
    col = sample(); col.units.loc[0, 'eroded'] = 1
    fig = stratigraphic_column(col)
    assert fig.axes[0].get_ylim()[0] == 200
    assert len(fig.axes[0].patches) == 1
    plt.close(fig)


def test_model_curves_use_actual_constants_and_only_real_supplied_points():
    col = sample(); col.lithologies.loc['shale', ['phi0', 'C']] = [.42, 2100.]
    fig = compaction_relations(col); ax = fig.axes[0]
    assert len(ax.lines) == 1 and not ax.collections
    line = ax.lines[0]
    assert np.allclose(line.get_xdata(), .42 * np.exp(-line.get_ydata() / 2100.))
    plt.close(fig)
    col.meta['porosity_data'] = [{'depth_m': 100., 'porosity': .3, 'lithology': 'shale'}]
    fig = compaction_relations(col)
    assert np.allclose(fig.axes[0].collections[0].get_offsets(), [[.3, 100.]])
    plt.close(fig)


def test_section_curves_and_numbered_unrestored_input():
    from pybasinsubs.app_core import load_section
    sec, faults = load_section()
    before = sec.h.copy(); fig = compaction_relations(sec)
    for line, i in zip(fig.axes[0].lines, np.flatnonzero(np.any(sec.F > 0, axis=(0, 1)))):
        assert np.allclose(line.get_xdata(), sec.phi0[i] * np.exp(-line.get_ydata() / sec.C[i]))
    assert not fig.axes[0].collections
    plt.close(fig)
    fig = interpreted_section(sec, faults)
    assert {'#1', '#2', '#3', '#4', '#5'} <= {t.get_text() for t in fig.axes[0].texts}
    assert fig.axes[0].get_ylim()[0] > max(np.max(np.asarray(f.stick)[:, 1]) for f in faults)
    np.testing.assert_array_equal(sec.h, before)
    plt.close(fig)
    fig = interpreted_section(sec, faults, show_fault_numbers=False)
    assert not fig.axes[0].texts
    plt.close(fig)


def test_selected_plot_curves_and_references_never_change_model():
    col = sample();before = col.lithologies.copy(deep=True);fractions = col.fractions().copy()
    extra = pd.DataFrame([dict(lithology='mudstone_reference', phi0=.42, C=2200.)])
    fig = compaction_relations(col, selected_curves=['mudstone_reference'], extra_curves=extra)
    line = fig.axes[0].lines[0]
    assert len(fig.axes[0].lines) == 1 and line.get_linestyle() == '--' and 'reference only' in line.get_label()
    np.testing.assert_allclose(line.get_xdata(), .42*np.exp(-line.get_ydata()/2200))
    pd.testing.assert_frame_equal(col.lithologies, before);np.testing.assert_array_equal(col.fractions(), fractions)
    plt.close(fig)
    fig = compaction_relations(col, selected_curves=[], extra_curves=extra)
    assert not fig.axes[0].lines and any('No porosity curves' in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)
    fig = compaction_relations(col)
    assert len(fig.axes[0].lines) == 1 and fig.axes[0].lines[0].get_label().startswith('shale:')
    plt.close(fig)


def test_curve_picker_keeps_real_point_overlay_and_sparse_fit_context():
    col = sample()
    points = [dict(depth_m=100., porosity=.3, lithology='shale')]
    fig = compaction_relations(col, measured=points, selected_curves=[])
    assert not fig.axes[0].lines and len(fig.axes[0].collections) == 1
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(), [[.3, 100.]])
    plt.close(fig)
