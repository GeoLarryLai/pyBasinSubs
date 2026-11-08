# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Separate 1-D timestep views; the original static plotting API is unchanged.

All geometry comes from ``BackstripResult.best``, not a second decompaction
solver. Depth is positive downward relative to PRESENT sea level: seafloor
W - SL, paleo sea surface -SL. Only W > 0 is filled with water.
"""
from __future__ import annotations

from pathlib import Path
import io

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

from . import plotting, thermal

# Fixed normalized axes. Standalone figures retain their painted age track.
HISTORY_RECT = (.08, .29, .66, .61)
COLUMN_RECT = (.80, .29, .045, .61)
GRADIENT_RECT = (.89, .29, .085, .61)

# Unit identity only: navy / steel / sky / teal blues, never window or T colors.
THERMAL_UNIT_PALETTE = ("#0b2545", "#236a8a", "#3c8dad", "#0e606b",
                        "#517f9f", "#1864a0", "#72a9cf", "#234b73")


def thermal_unit_colors(result):
    """Stable blueish colors keyed by input-unit index, shared by thermal views.

    Positive-thickness units cycle the eight explicit navy/steel/sky/teal blues;
    zero-thickness markers are gray and do not consume a unit color.
    """
    colors = []
    used = 0
    for thickness in result.column.units.thickness_m:
        colors.append(THERMAL_UNIT_PALETTE[used % len(THERMAL_UNIT_PALETTE)] if thickness > 0 else '#777777')
        used += int(thickness > 0)
    return colors


def _thermal_unit_horizons(ax, result):
    """Replace static black unit-base outlines with color-keyed UNIT TOP paths.

    Temperature / Ro curves track unit tops, so using the next unit's base color
    here would misidentify a boundary. Keep the independent deepest TD outline.
    """
    b = result.best; u = result.column.units
    ids = [j for j in range(len(u)) if u.thickness_m.iloc[j] > 0]
    # geohistory draws one base line per positive-thickness unit before its
    # water/zero-datum lines. Contours are collections, not these Line2D objects.
    for line in ax.lines[:len(ids)]:
        line.set_visible(False)
    # First two PolyCollections are water/land; then the unit fill bands.
    for fill in ax.collections[2:2 + len(ids)]:
        fill.set_alpha(.22)
    ages = b['age'][0]; order = np.argsort(-ages); sea = b['W'][0] - b['SL'][0]
    colors = thermal_unit_colors(result)
    for j in ids:
        top = b['tops'][0, :, j] + sea
        active = np.isfinite(top)
        x, y = ages[order][active[order]], top[order][active[order]]
        ab = float(u.age_base_Ma.iloc[j])
        if ab > x[0]:
            x = np.r_[ab, x]
            y = np.r_[np.interp(-ab, -ages[order], sea[order]), y]
        line, = ax.plot(x, y, color=colors[j], lw=1.25, alpha=.95, zorder=3)
        line.set_gid(f'unit-top-{j}')
        line.set_label(str(u.unit.iloc[j]))
    if ids:
        j = ids[-1]
        base = b['tops'][0, :, j] + b['thick'][0, :, j] + sea
        ax.plot(ages[order], base[order], color='.4', lw=.55, alpha=.7)



def timestep_steps(result):
    """Oldest-to-youngest node metadata (including hiatus/erosion events).

    The basal deposition-start age is added only if not already represented.
    For that empty initial surface, the oldest available W/SL are held constant.
    Equal-age pre/post-erosion nodes remain separate, ordered like the solver.
    """
    b = result.best
    steps = [dict(index=k, node_index=k, age_Ma=float(age), label=f"{age:g} Ma · {lab}")
             for k, (age, lab) in enumerate(zip(b['age'][0], result.node_labels))]
    oldest = float(result.column.units.age_base_Ma.max())
    if oldest > steps[0]['age_Ma'] + 1e-9:
        steps.insert(0, dict(index=-1, node_index=None, age_Ma=oldest,
                             label=f"{oldest:g} Ma · deposition starts (empty surface)"))
    for i, step in enumerate(steps):
        step['index'] = i
    return steps


def timestep_column(result, step=-1):
    """Return one exact node's layers, paleo surface/water and fixed y-limits.

    ``step`` indexes :func:`timestep_steps`, not an arbitrary interpolated age.
    Zero-thickness markers (e.g. TD) are not strata. Present means youngest
    computed node, which can predate 0 Ma for an uplifted/eroded column.
    """
    steps = timestep_steps(result)
    s = steps[step]
    b = result.best
    k = s['node_index']
    ref = 0 if k is None else k
    water = float(b['W'][0, ref]); sea = float(-b['SL'][0, ref])
    surface = water + sea
    units = result.column.units
    layers = []
    if k is not None:
        for j in range(len(units)):
            top, thick = b['tops'][0, k, j], b['thick'][0, k, j]
            if np.isfinite(top) and np.isfinite(thick) and thick > 1e-8:
                layers.append(dict(unit_index=j, unit=str(units.unit.iloc[j]),
                                   top_m=float(top + surface), base_m=float(top + surface + thick),
                                   thickness_m=float(thick)))
    surfaces = b['W'][0] - b['SL'][0]
    lo = min(0., float(np.min(-b['SL'][0])), float(np.min(surfaces)))
    hi = max(0., float(np.max(surfaces + b['eps'][0])))
    pad = max(20., (hi - lo) * .04)
    return dict(**s, layers=layers, surface_m=surface, sea_surface_m=sea,
                water_depth_m=max(0., water), ylim=(hi + pad, lo - pad))


def _unit_colors(result):
    u = result.column.units
    mids = (u.age_top_Ma + u.age_base_Ma) / 2
    if u.age_base_Ma.max() - u.age_top_Ma.min() > 40:
        return [plotting.ics_color(a)[0] for a in mids]
    norm = mpl.colors.Normalize(u.age_top_Ma.min(), u.age_base_Ma.max())
    return [plt.get_cmap('viridis_r')(norm(a)) for a in mids]


def _draw_column(ax, result, state):
    colors = _unit_colors(result)
    if state['water_depth_m'] > 0:
        ax.add_patch(Rectangle((0, state['sea_surface_m']), 1, state['water_depth_m'],
                              facecolor=plotting.C_WATERFILL, alpha=.35, lw=0))
    for layer in state['layers']:
        j = layer['unit_index']
        ax.add_patch(Rectangle((0, layer['top_m']), 1, layer['thickness_m'],
                              facecolor=colors[j], edgecolor='white', linewidth=.6,
                              alpha=.45 if result.column.units.eroded.iloc[j] else .95,
                              hatch='//' if result.column.units.eroded.iloc[j] else None))
        if layer['thickness_m'] > (state['ylim'][0] - state['ylim'][1]) * .025:
            ax.text(.5, (layer['top_m'] + layer['base_m']) / 2, str(j + 1),
                    ha='center', va='center', fontsize=6, color='.2')
    ax.axhline(state['surface_m'], color=plotting.C_W, lw=1.6)
    ax.axhline(state['sea_surface_m'], color='.4', lw=.6, ls=':')
    ax.set(xlim=(0, 1), ylim=state['ylim'], xticks=[])
    ax.set_title('Strata', fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)


def timestep_thermal_fields(result, *, heat_flow=.059, T_s=4., A=1e-6, hr=None):
    """Cache the existing temperature/Easy%Ro fields for all timestep views.

    Heat flow is W/m² (scalar or age callable); windows are kinetic %Ro, NOT
    instantaneous temperature thresholds. Pass this cache as ``fields=`` when
    rendering multiple thermal frames. Model conventions match thermal.py.
    """
    tf = thermal.temperature_field(result, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)
    mf = thermal.maturity_field(result, tf)
    return dict(temperature=tf, maturity=mf, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)


def _profile(result, state, fields):
    k = state['node_index']
    if k is None:
        return np.array([state['surface_m']]), np.array([_at(fields['T_s'], state['age_Ma'])])
    z, _, conductivity, _ = thermal.column_profile(result, node=k, below=0.)
    # End exactly at the sediment base; no invented window in basement/water.
    base = float(result.best['eps'][0, k])
    depth = np.unique(np.r_[z[z < base], base])
    temperature = thermal.steady_geotherm(z, conductivity, _at(fields['heat_flow'], state['age_Ma']),
                                         _at(fields['T_s'], state['age_Ma']), fields['A'], fields['hr'])
    return depth + state['surface_m'], np.interp(depth, z, temperature)


def _at(value, age):
    return value(age) if callable(value) else value


def _draw_gradient(ax, result, state, fields):
    z, temp = _profile(result, state, fields)
    # Fixed temperature limits over the whole history, just like depth limits.
    profiles = [_profile(result, timestep_column(result, s['index']), fields)[1]
                for s in timestep_steps(result)]
    tmin = min(0., min(float(np.min(t)) for t in profiles))
    tmax = max(25., max(float(np.max(t)) for t in profiles))
    ax.plot(temp, z, color='#b2182b', lw=1.6, marker='o' if len(z) == 1 else None, ms=3)
    mf = fields['maturity']; tf = fields['temperature']
    if state['layers']:
        # Interpolate each Lagrangian tracer at the selected age; preserve NaN
        # outside its deposited history. Never infer maturity from T alone.
        ages = tf['age'][::-1]
        depths, ros = [], []
        active = {p['unit_index'] for p in state['layers']}
        for j, uid in enumerate(mf['unit']):
            if uid not in active:
                continue
            d = np.interp(state['age_Ma'], ages, mf['paths'][::-1, j])
            ro = np.interp(state['age_Ma'], ages, mf['ro_paths'][::-1, j])
            if np.isfinite(d) and np.isfinite(ro):
                depths.append(d); ros.append(ro)
        if len(depths) > 1:
            order = np.argsort(depths)
            # Shade only the depth range sampled by actual sediment tracers.
            zg = np.linspace(min(depths), max(depths), 600)
            rg = np.interp(zg, np.asarray(depths)[order], np.asarray(ros)[order])
            for name, low, high, color in thermal.WINDOWS[1:]:
                mask = (rg >= low) & (rg < high)
                if mask.any():
                    band = ax.fill_betweenx(zg, tmin, tmax * 1.08, where=mask, color=color, alpha=.25, lw=0)
                    band.set_gid('kinetic-window-' + name)
                    # Label each contiguous tracer-defined interval, not a T cutoff.
                    runs = np.split(np.flatnonzero(mask), np.where(np.diff(np.flatnonzero(mask)) > 1)[0] + 1)
                    for run in runs:
                        ax.text(.5, float(np.mean(zg[run])), name, transform=ax.get_yaxis_transform(),
                                ha='center', va='center', fontsize=6.2, color='.2',
                                bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=.25))
    ax.set(xlim=(tmin, tmax * 1.08), ylim=state['ylim'], xlabel='T (°C)')
    ax.set_title('Geotherm', fontsize=9)
    # Persistent AXES-coordinate key, explicitly not inferred depth intervals.
    # It stays visible even in an empty/water-only timestep. Actual depth bands
    # above are drawn only where sediment tracers support kinetic Easy%Ro.
    ax.text(.74, .995, 'key', transform=ax.transAxes, ha='center', va='top', fontsize=6, color='.35')
    for i, (name, _, _, color) in enumerate(thermal.WINDOWS[1:]):
        y = .925 - i * .065
        swatch = Rectangle((.46, y - .025), .53, .05, transform=ax.transAxes,
                           facecolor=color, alpha=.55, edgecolor='none', zorder=5)
        swatch.set_gid('window-key-' + name)
        ax.add_patch(swatch)
        ax.text(.725, y, name, transform=ax.transAxes, ha='center', va='center', fontsize=6.2,
                color='.15', zorder=6)
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='y', labelleft=False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)


def _plot(result, step, fields=None, *, figsize=(12, 6), column_only=False, slider=True, _figure=None):
    state = timestep_column(result, step)
    fig = _figure if _figure is not None else plt.figure(figsize=figsize)
    if not column_only:
        history_rect = HISTORY_RECT if slider else (.08, .16, .66, .74)
        ax = fig.add_axes(history_rect, label='history')
        if fields is None:
            plotting.geohistory(result, ax=ax, legend=False)
            handles = [mpl.lines.Line2D([], [], color=c, label=lab) for c, lab in
                       [(plotting.C_TOTAL, 'Total subsidence'), (plotting.C_TECT, 'Tectonic subsidence')]]
            handles.append(Patch(color=plotting.C_WATERFILL, alpha=.35, label='Paleo-water'))
            ax.legend(handles=handles, fontsize=8, loc='lower left')
        else:
            existing_axes = set(fig.axes)
            plotting.geohistory_thermal(result, fields['temperature'], fields['maturity'], ax=ax)
            # Short-span columns make the static helper add an age colorbar.
            # Our dedicated thermal unit key is the paired blue line palette.
            for extra in set(fig.axes) - existing_axes:
                fig.delaxes(extra)
            ax.set_position(history_rect)
            _thermal_unit_horizons(ax, result)
            # The static helper's extra below-axis legend would overlap our time track.
            for artist in list(ax.artists):
                if isinstance(artist, mpl.legend.Legend):
                    artist.remove()
        ax.set_ylim(state['ylim'])
        steps = timestep_steps(result)
        ax.set_xlim(steps[0]['age_Ma'], steps[-1]['age_Ma'])
        ax.axvline(state['age_Ma'], color='.15', ls='--', alpha=.5, lw=1.5)
        for key, color in [('sigma', plotting.C_TOTAL), ('zeta', plotting.C_TECT)]:
            k = state['node_index']
            value = state['surface_m'] if k is None else result.best[key][0, k]
            ax.plot(state['age_Ma'], value, 'o', mfc='white', mec=color, mew=1.8, ms=8 if key == 'sigma' else 6, zorder=10, clip_on=False)
        if slider:
            # A time-proportional track for exported animations/stills. The browser
            # and Matplotlib interactive view replace this with a genuine slider.
            track = fig.add_axes((.08, .16, .66, .025), label='time')
            ages = [s['age_Ma'] for s in timestep_steps(result)]
            track.plot(ages, np.zeros(len(ages)), 'o-', color='.65', ms=3, lw=1)
            track.plot(state['age_Ma'], 0, 'o', color=plotting.C_TOTAL, ms=7)
            track.set(xlim=ax.get_xlim(), ylim=(-1, 1), yticks=[], xlabel='Age (Ma) · oldest ← → youngest')
            track.xaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
            track.tick_params(labelsize=8); track.spines[:].set_visible(False)
    col_rect = (COLUMN_RECT if slider else (.80, .16, .045, .74)) if not column_only else (.25, .23, .10, .65)
    ac = fig.add_axes(col_rect, label='strata')
    _draw_column(ac, result, state)
    if fields is not None:
        grad_rect = (GRADIENT_RECT if slider else (.89, .16, .085, .74)) if not column_only else (.53, .23, .32, .65)
        at = fig.add_axes(grad_rect, label='gradient')
        _draw_gradient(at, result, state, fields)
        handles = [Patch(color=color, alpha=.35, label=f'{name} {lo:g}–{hi:g} %Ro')
                   for name, lo, hi, color in thermal.WINDOWS[1:]]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .01),
                   fontsize=7, ncol=1 if column_only else 3, frameon=False)
    fig.suptitle(f"{result.column.name} · {state['label']}", x=.08, ha='left', fontsize=11)
    fig.text(.08, .06 if not column_only else .14,
             ('Depth below present sea level (m), positive down · fixed range · water from −SL to W−SL'
              if not column_only else 'Depth below present sea level (m), positive down\nFixed range · water: −SL to W−SL'),
             fontsize=7 if not column_only else 6, color='.35')
    return fig


def plot_geohistory_timestep(result, step=-1, *, figsize=(12, 6), slider=True):
    """Geohistory + thin column; ``slider=False`` expands app axes, no age-track gutter."""
    return _plot(result, step, figsize=figsize, slider=slider)


def plot_thermal_timestep(result, step=-1, *, fields=None, heat_flow=.059, T_s=4.,
                          A=1e-6, hr=None, column_only=False, figsize=None, slider=True):
    """New thermal history + thin column/geotherm; windows use tracer Easy%Ro.

    ``column_only=True`` produces a compact per-timestep column/geotherm still.
    ``slider=False`` uses the track space for taller axes in the app.
    A supplied ``fields`` cache is authoritative (heat_flow/T_s/A/hr ignored).
    """
    fields = fields if fields is not None else timestep_thermal_fields(result, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)
    return _plot(result, step, fields, column_only=column_only, slider=slider,
                 figsize=figsize or ((4, 6) if column_only else (12, 6)))


def timestep_view(result, *, thermal_view=False, fields=None, **thermal_options):
    """Matplotlib node-snapping slider for an interactive notebook/GUI backend.

    Returns (figure, slider); keep both alive. Inline-only notebook backends show
    a still: display the dedicated GIF exports there instead. Equal-age events
    can be selected with Left/Right keys while retaining true age spacing.
    """
    from matplotlib.widgets import Slider
    steps = timestep_steps(result)
    if thermal_view and fields is None:
        fields = timestep_thermal_fields(result, **thermal_options)
    index = len(steps) - 1
    fig = _plot(result, index, fields)
    fig.delaxes(next(a for a in fig.axes if a.get_label() == 'time'))
    sa = fig.add_axes((.08, .16, .66, .025))
    values = -np.array([s['age_Ma'] for s in steps])
    slider = Slider(sa, 'Age (Ma)', values[0], values[-1], valinit=values[-1], valstep=np.unique(values))
    slider.valtext.set_text(f'{-slider.val:g}')
    sa.plot(values, np.full(len(values), .5), '|', color='.4', ms=10, zorder=3)
    sa.set_xlim(values[0], values[-1])

    def draw(i):
        nonlocal index
        index = i
        for a in list(fig.axes):
            if a is not sa:
                fig.delaxes(a)
        for text in list(fig.texts):
            text.remove()
        fig._suptitle = None
        for legend in list(fig.legends):
            legend.remove()
        _plot(result, i, fields, _figure=fig)
        fig.delaxes(next(a for a in fig.axes if a.get_label() == 'time'))
        slider.valtext.set_text(f"{steps[i]['age_Ma']:g}")
        fig.canvas.draw_idle()

    slider.on_changed(lambda value: draw(int(np.argmin(np.abs(values - value)))))

    def key(event):
        if event.key in ('left', 'right'):
            i = int(np.clip(index + (1 if event.key == 'right' else -1), 0, len(steps)-1))
            slider.eventson = False; slider.set_val(values[i]); slider.eventson = True
            draw(i)
    fig.canvas.mpl_connect('key_press_event', key)
    # Slider widgets use weak canvas references; figure owns it for notebooks.
    fig._timestep_slider = slider
    return fig, slider


def _save_animation(result, filename, fields, fps, dpi):
    from PIL import Image
    path = Path(filename)
    if path.suffix.lower() != '.gif':
        raise ValueError('Timestep animations currently export .gif (Pillow; no extra dependencies)')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    frames = []
    for s in timestep_steps(result):
        fig = _plot(result, s['index'], fields)
        try:
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=dpi)
            with Image.open(buf) as image:
                frames.append(image.convert('RGB'))
        finally:
            plt.close(fig)
    palette = frames[-1].quantize(colors=256)
    images = [im.quantize(palette=palette) for im in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps),
                   loop=0, optimize=False, disposal=2)
    return path


def save_geohistory_animation(result, filename, *, fps=1., dpi=110):
    """Save an oldest-to-youngest geohistory/column GIF without changing static APIs."""
    return _save_animation(result, filename, None, fps, dpi)


def save_thermal_animation(result, filename, *, fields=None, heat_flow=.059, T_s=4.,
                           A=1e-6, hr=None, fps=1., dpi=110):
    """Save thermal-history/column/geotherm GIF with kinetic oil/gas windows."""
    fields = fields if fields is not None else timestep_thermal_fields(result, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)
    return _save_animation(result, filename, fields, fps, dpi)


def maturity_timestep_data(result, *, burial=None, heat_flow=.059, T_s=4., A=1e-6, hr=None):
    """Cache full unit-top T/Easy%Ro histories for the linked two-panel view.

    ``burial`` reuses a previously computed ``thermal.burial_temperatures`` dict.
    Kinetics and sorting match app_core.maturity_figure; no backstripping rerun.
    """
    bt = burial if burial is not None else thermal.burial_temperatures(result, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)
    u = result.column.units
    ids = [j for j in range(len(u)) if u.thickness_m.iloc[j] > 0]
    ages = np.asarray(bt['age']); order = np.argsort(-ages)
    ages = ages[order]; temps = np.asarray(bt['T'])[order][:, ids]
    ro = np.full_like(temps, np.nan)
    for j in range(len(ids)):
        valid = np.isfinite(temps[:, j])
        if valid.sum() >= 2:
            ro[valid, j] = thermal.easy_ro(ages[valid], temps[valid, j])
    return dict(age=ages, temperature=temps, ro=ro, unit_indices=ids,
                labels=[str(u.unit.iloc[j]) for j in ids])


def plot_maturity_timestep(result, step=-1, *, data=None, burial=None, heat_flow=.059,
                            T_s=4., A=1e-6, hr=None):
    """Familiar full two-panel T/Easy%Ro curves with blue unit identity + age overlay.

    Future curves are NOT clipped. A supplied ``data`` cache is authoritative.
    Equal-age events share the overlay at the youngest/post-event row at that
    numeric age; their column geometries remain separately traversable.
    """
    data = data if data is not None else maturity_timestep_data(result, burial=burial, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)
    state = timestep_column(result, step)
    fig = plotting.maturity_plot(data['age'], data['temperature'], data['ro'], data['labels'],
                                 title=f'{result.column.name}: burial temperature and maturity')
    colors = thermal_unit_colors(result)
    exact = np.flatnonzero(np.isclose(data['age'], state['age_Ma'], rtol=0, atol=1e-9))
    row = int(exact[-1]) if len(exact) else None
    # Recolor only this NEW figure; the original static helper stays plasma.
    for ax, key in zip(fig.axes, ['temperature', 'ro']):
        for line, uid in zip(ax.lines, data['unit_indices']):
            line.set_color(colors[uid]); line.set_gid(f'unit-curve-{uid}')
        limits = ax.get_xlim(), ax.get_ylim()
        guide = ax.axvline(state['age_Ma'], color='.15', ls='--', lw=1.4, alpha=.5)
        guide.set_gid('timestep-age')
        if row is not None:
            for j, uid in enumerate(data['unit_indices']):
                value = data[key][row, j]
                if np.isfinite(value):
                    marker, = ax.plot(state['age_Ma'], value, 'o', mfc='white', mec=colors[uid],
                                      mew=1.5, ms=5, zorder=8, clip_on=False)
                    marker.set_gid(f'unit-highlight-{uid}')
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    fig.axes[0].legend(fontsize=7, ncol=2, title='Unit (top)', title_fontsize=7)
    # A separate, fixed-position age annotation distinguishes animation frames
    # even when oldest/youngest guide lines coincide with the panel border.
    fig.text(.99, .985, f"{state['age_Ma']:g} Ma", ha='right', va='top', fontsize=9, color='.3')
    return fig


def save_maturity_animation(result, filename, *, data=None, burial=None, heat_flow=.059,
                             T_s=4., A=1e-6, hr=None, fps=1., dpi=110):
    """Full-curve two-panel maturity GIF, exact nodes oldest→youngest (default 1 s/node)."""
    from PIL import Image
    path = Path(filename)
    if path.suffix.lower() != '.gif':
        raise ValueError('Maturity animations export .gif (Pillow; no extra dependencies)')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    data = data if data is not None else maturity_timestep_data(result, burial=burial, heat_flow=heat_flow, T_s=T_s, A=A, hr=hr)
    frames = []
    steps = timestep_steps(result)
    for step in steps:
        fig = plot_maturity_timestep(result, step['index'], data=data)
        # Keep equal-age event frames distinct in GIF encoders which otherwise
        # coalesce identical consecutive images; curves/age overlays stay equal.
        fig.text(.99, .01, f"Constraint {step['index'] + 1}/{len(steps)} · {step['label']}",
                 ha='right', va='bottom', fontsize=7, color='.35')
        try:
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=dpi)
            with Image.open(buf) as image:
                frames.append(image.convert('RGB'))
        finally:
            plt.close(fig)
    palette = frames[-1].quantize(colors=256)
    images = [frame.quantize(palette=palette) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps),
                   loop=0, optimize=False, disposal=2)
    return path
