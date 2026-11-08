# Changelog

## 0.2.0 (2026-11-08)

- Add explicit cached whole-section thermal timesteps (independent steady columns + kinetic Easy%Ro), translucent sediment-only window overlays and unit contacts, three 0–100% line strips (oil, wet gas, dry gas) with one opaque line per supported remaining unit in its original unit color, transparent shade fills (about alpha 0.3) under each window curve and a y-axis padded slightly past 0 and 100, wizard step 5, notebook slider and optional GIF export.
- Growth reconstruction now flattens the footwall of the westernmost fault (#1) by default (`footwall_flattening=True`), so older VG1 frames stay continuous across #1; this flattening is stored separately and is not counted as fault throw.

### Fixed
- Browser visibility now respects hidden controls and figures; fault renaming keeps the renamed fault selected.
- The section API uses the same 50-draw default advertised by the browser when `n_mc` is omitted.
- Listric fault restoration (`flex2d.kinematic`, `restore_kinematic(method="listric")`): the inclined-shear solver now finds
  the shear-line base and the subsidence by bisection. The old fixed-point loops did not converge when the fault was steeper
  than the shear angle (65 degrees), which gave wrong heave and subsidence.
- Listric dip fit (`fit_listric_stick`): only the part of the stick above 0.9 x detachment depth is fitted. Sticks that end at
  the detachment used to push the fitted dip to the 85 degree bound. On the template section the fitted dips are now
  38, 36, 32, 40, 37 degrees.
- Template-section listric results change with the two fixes above (oldest-horizon std 534 m -> 843 m, mean fault offset
  520 m); earlier listric numbers in notebooks and posts are not comparable.
- A fault CSV without the columns `fault_id, x, z` now gives a clear error in `app_core.load_section`.

### Changed
- Public README: clearer data workflows, app screenshots, linked notebooks, and a readable two-panel post-rift overview. PyPI uses a separate overview-only `README-pypi.md`.
- Notebook descriptions clarify the current fault-geometry comparison and distinguish offsets sampled at picked traces from offsets sampled at each geometry's own traces; saved results are unchanged.
- One browser app replaces the Panel and Streamlit apps: `pybasinsubs-app` (FastAPI + plain HTML/JS, opens in your browser)
  with three pages: 1-D column, 2-D section, fault editor. `pybasinsubs-ui` and `pybasinsubs-tracer` stay as aliases.
  The 2-D Monte Carlo default is 50 draws (200+ for figures; listric is about 8x slower).
- `pybasinsubs.faults` is now a package (`sticks`, `detect`, `formats`, `curves`, `user_store`). Every old
  `pybasinsubs.faults.<name>` import still works. Fault JSON/CSV formats and default paths are unchanged.
- `fault_tracer.FaultTracer` no longer has a Panel/Bokeh view. New `FaultTracer.run_restoration(method, Te, Hd)`.

### Removed
- Streamlit, Panel, Bokeh and ipywidgets dependencies; `pybasinsubs.ui`, `pybasinsubs.cli`, `pybasinsubs.apps`.

### Added
- 1-D constraint-age sliders with thin, decompacting stratigraphic columns, fixed depth ranges and paleowater columns; original static plots remain available unchanged.
- A linked thermal view pairs the column + geotherm with slider-driven burial-temperature / Easy%Ro panels, shared blueish unit colors and always-visible kinetic-window labels.
- Dedicated timestep plotting and GIF animation helpers for geohistory, thermal columns and maturity; optional app animation export and executable notebook examples.
- Dependencies: `fastapi>=0.110`, `uvicorn>=0.27` (dev: `httpx` for the app tests).
- `tests/test_webapp.py`.

## 0.1.2
Previous release (PyPI / conda-forge).
