# pyBasinSubs

**pyBasinSubs** is a Python toolkit for **1-D and 2-D backstripping and decompaction basin analysis**, with a user-friendly, semi-automated local browser workflow and reproducible notebooks. It extends the backstripping workflow used in Lai et al. (2022), linking stratigraphic data to subsidence, burial, thermal history and experimental structural restoration.

## Data and capabilities

- **Columns and well tops:** decompaction and 1-D Airy backstripping, erosion and over-compaction, with Monte Carlo uncertainty.
- **Compaction data and microfossils:** porosity–depth fitting and paleo-water-depth estimation.
- **Thermal history:** geotherms, McKenzie stretching-factor fits, burial temperature, Easy%Ro maturity and oil/gas windows.
- **Property modeling:** porosity, permeability and stress from published compaction and cementation relations; existing backstrip results are unchanged unless these functions are called.
- **Well transects:** thin-plate flexural profiles through well-based columns and effective elastic thickness fitting, distinct from fault-resolved restoration.
- **Correlated sections and interpreted faults:** 2-D flexural backstripping and decompaction, cantilever/listric/domino restoration options, and a fixed-trace growth-history prototype with spatially variable nonnegative apparent throw and sedimentation rates.
- **Local browser app:** two main analysis workflows, with fault-stick editing available in the navigation; the app and notebooks share calculation engines. The legacy module preserves the original R-workflow results.

Bundled examples cover Taiwan's Coastal Range, North Sea and Barents Sea wells, and the Viking Graben. CSV templates support user columns and sections; fault import supports JSON, CSV, GeoJSON and selected ASCII layouts.

## Status and documentation

Under active development; many methods, especially fault restoration, are experimental and not fully validated. The growth prototype uses regularized vertical-shear apparent-throw fields within fixed hanging-wall partitions after decompaction and flexural unloading. It does not resolve down-dip slip, geological heave or curved-fault balancing; zero is an assumed datum, and interpreted traces remain user-controlled. Restored geometry is not a unique geological solution.

Requires Python 3.12 or newer. Code license: **AGPL-3.0-or-later**. Third-party data retain their own terms; the Miller et al. (2020) sea-level curve is downloaded on first use, not bundled, and has non-commercial restrictions. Data and method credits are in NOTICE.

Illustrated documentation, notebooks, installation guidance and release history: [GitHub](https://github.com/GeoLarryLai/pyBasinSubs).

## Citation

Lai, L.S.-H. (2026) pyBasinSubs. *Zenodo*, [https://doi.org/10.5281/zenodo.23832018](https://doi.org/10.5281/zenodo.23832018).

Method paper: Lai et al. (2022), *Extremely rapid up-and-down motions of island arc crust during arc-continent collision*, [Communications Earth & Environment 3, 100](https://doi.org/10.1038/s43247-022-00429-2).
