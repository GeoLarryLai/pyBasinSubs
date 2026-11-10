# Example data – sources and licences

Shipped inside the package; locate files with `pybasinsubs.data_path("north_sea", ...)`. Full attributions: `NOTICE` at the repository root.

| folder / file | content | source | licence |
|---|---|---|---|
| `coastal_range/Thickness, Sealevel, Backstripping` | inputs of the BasinSubs R code (thickness, ages, lithology, foraminifera bathymetry, porosity curves, densities) | Lai et al. (2022) | AGPL-3.0-or-later (this package) |
| `coastal_range/r_outputs` | outputs written by the R code (`output results/`), used only for parity tests | Lai et al. (2022) | AGPL-3.0-or-later (this package) |
| *(not shipped)* `SeaLevel_Miller2020.csv` | global sea level 0–66.6 Ma, Miller et al. (2020) *Sci. Adv.* 6, eaaz1346 | downloaded on first use from the BasinSubs repository, SHA-256 checked, cached in `~/.cache/pybasinsubs` (`pybasinsubs.sealevel.miller2020_path`) | CC BY-NC 4.0 – non-commercial use only; not redistributable under AGPL-3.0 |
| `coastal_range/Sealevel/SL_Unit_N.csv`, `SL_Unit_S.csv` | unit-averaged sea level written by the R code (reference outputs for parity tests) | derived from Miller et al. (2020) | AGPL-3.0-or-later (this package) |
| `north_sea/roberts1998_fig4a_vg1_digitized.csv`, `faults/*` | horizon depths of line Viking Graben 1 digitized from Roberts et al. (1998) *Petrol. Geosci.* 4, 327–338, Fig. 4a; fault-stick examples derived from it | digitized by the pyBasinSubs authors | derived numerical data, attribution required; the figure is not distributed (see `NOTICE`) |
| `figshare_lai2022/SuppData2–4*.xlsx` | supplementary data of Lai et al. (2022): microfossils, paleobathymetry, age model / subsidence | Figshare, Lai, L.S.-H. et al. (2022) | CC BY 4.0 |
| `north_sea/sodir_*_wells.csv`, `sodir_*_group_tops.csv` | well headers (location, KB, water depth, TD, BHT) and lithostratigraphic group tops, North Viking Graben (8 wells) and Hammerfest Basin (6 wells) | Norwegian Offshore Directorate (Sodir) FactPages, downloaded 2026-10 | Norwegian Licence for Open Government Data (NLOD 2.0) |
| `north_sea/group_lookup.csv` | group ages, paleo-water depth, lithology fractions and the reconstructed eroded Cenozoic overburden | **compiled for this package** from the ICS 2023 time scale and standard North Sea / Barents lithostratigraphy (Gradstein et al. 2010; Halland et al. 2014 CO₂ atlas; Henriksen et al. 2011). Values are approximate, for demonstration | AGPL-3.0-or-later (this package) |
| `heatflow/ihfc2024_northsea_barents.csv` | 326 heat-flow sites 55–75° N, 0–35° E | IHFC Global Heat Flow Database, 2024 release (GFZ Data Services, https://doi.org/10.5880/fidgeo.2024.014) | CC BY 4.0 |
| `north_sea/lithology_priors.csv` | property-evolution priors (grain size, quartz and clay fractions, grain coatings, Schneider β, Kozeny–Carman factor). Every value has an error and a source column. Chalk permeability uses the Mortensen / Fabricius Kozeny form rather than grain size | **literature or assumed priors compiled for this package**; not measured at the bundled wells. Citations are in the file | AGPL-3.0-or-later (this package) |
| `template/my_column_template.csv` | example input column (well 30/4-1) | derived from the Sodir data above | NLOD 2.0 / AGPL-3.0-or-later |

## Assumptions in the North Sea / Barents demo data
- Paleo-water depths per group are generic estimates (±50 % error), not from well micropalaeontology.
- Group tops are measured depths (MD) converted to depth below seafloor with KB and water depth; wells assumed vertical.
- Sodir does not report time since circulation for BHT: 12 h is assumed.
- Eroded Cenozoic in the Barents wells: 1100 ± 400 m removed at 2.7 Ma (within published 1–1.5 km estimates for the Hammerfest Basin).
- Sea level before 66.6 Ma: 0 ± 100 m (Miller 2020 does not extend further).
- No IHFC site falls inside the transect windows; nearest sites are ~150–200 km away.
- Diagenesis priors in `lithology_priors.csv` are literature values or assumed typical numbers, not grain-size, mineralogy, porosity, permeability or cement measurements from these wells.

## Not included / fetch notes
- The GFZ heat-flow portal timed out; the IHFC 2024 release was taken from its backup xlsx mirror and subset.
- Figshare required a browser user-agent for download.
- No Easy%RoDL weight table could be verified from an open source; Easy%RoDL weights must be supplied by the user (`thermal.easy_ro(..., kinetics=dict(A=2e14, E_kcal=[...], f=[...]))`).
