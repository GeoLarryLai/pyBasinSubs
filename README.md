<p align="center"><img src="image/logo.png" alt="pyBasinSubs" width="160"></p>

# pyBasinSubs

[![PyPI version](https://img.shields.io/pypi/v/pybasinsubs.svg)](https://pypi.org/project/pybasinsubs/) [![pepy downloads](https://static.pepy.tech/badge/pybasinsubs)](https://pepy.tech/project/pybasinsubs) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.ZENODO_CONCEPT_DOI.svg)](https://doi.org/10.5281/zenodo.ZENODO_CONCEPT_DOI)

v0.1.0

![Geohistory of North Sea well 30/4-1](image/hero_geohistory_north_sea_30-4-1.png)

Reconstructs how a sedimentary basin sank and filled through time, by backstripping its layers. Python version and
extension of [BasinSubs](https://github.com/GeoLarryLai/BasinSubs) (R; [Lai et al., 2022](https://doi.org/10.1038/s43247-022-00429-2)).

Status: early-stage research software. APIs may change; check results before use.

## Installation

```bash
git clone https://github.com/GeoLarryLai/pyBasinSubs && cd pyBasinSubs
pip install -e ".[ui,notebooks,test]"     # or: conda env create -f environment.yml
python -m pytest -q
```

## Citation

Lai, L.S.-H. (2026) pyBasinSubs. *Zenodo*, [https://doi.org/10.5281/zenodo.ZENODO_CONCEPT_DOI](https://doi.org/10.5281/zenodo.ZENODO_CONCEPT_DOI)

Please also cite the method paper:
[Lai et al. (2022)](https://doi.org/10.1038/s43247-022-00429-2).
Other references: [Roberts et al. (1998)](https://doi.org/10.1144/petgeo.4.4.327),
[Miller et al. (2020)](https://doi.org/10.1126/sciadv.aaz1346) (sea-level curve, downloaded on first use, not shipped),
[IHFC (2024)](https://doi.org/10.5880/fidgeo.2024.014). Data credits are in `NOTICE`.

## Key Features

- **Backstripping:** 1-D Airy backstripping with Monte Carlo uncertainty, erosion and over-compaction; `legacy`
  reproduces the BasinSubs R results.
- **Porosity and water depth:** porosity–depth fits with confidence bands; paleo-water depth from foraminifera.
- **Thermal history:** geotherms, McKenzie β fit, Easy%Ro maturity and oil/gas windows.
- **Flexure and sections:** 2-D flexural backstripping, T_e fitting and fault restoration across a section.
- **Your own data:** CSV templates in `pybasinsubs/example_data/template/`, plus desktop and Streamlit apps.

## Results

![Tectonic subsidence with uncertainty](image/nb3_geohistory_N.png)
*Northern Coastal Range (Taiwan): geohistory with total and tectonic subsidence and uncertainty bands.*

![Source-rock maturity](image/nb6_maturity_viking_30-4-1.png)
*North Sea well 30/4-1: burial temperature and maturity (Easy%Ro) with oil and gas windows.*

![Restored section through time](image/nb8_viking_restored_sequence.png)
*North Viking Graben: a 2-D section restored through time by flexural backstripping.*

Executed notebooks in `notebooks/` cover each feature (re-run with `python tools/run_notebooks.py`).

## Quick start

```python
from pybasinsubs import from_sodir, backstrip
from pybasinsubs.plotting import geohistory

r = backstrip(from_sodir("30/4-1"), n=2000)   # or from_basinsubs("N"), read_column("my.csv")
geohistory(r); r.summary()
```

Apps (`ui` extra): `pybasinsubs-ui` (backstripping), `pybasinsubs-tracer` (fault editor and restoration);
add `--streamlit` for the Streamlit versions.

## License

Copyright (C) 2026 Larry Syu-Heng Lai. AGPL-3.0-or-later (`LICENSE`). `legacy.py` and the bundled BasinSubs data keep the
BSD-3-Clause notice (`LICENSE-BasinSubs-BSD3`).
