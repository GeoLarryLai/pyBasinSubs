# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Lithology parameters for porosity-depth compaction (paper Eq. 5).

phi(z) = phi0 * exp(-z / C)   [z in m below seafloor, C = 1/c decay length in m]

Each lithology carries phi0 (surface porosity, fraction), C (decay length, m),
rho_g (grain density, kg/m^3), their 1-sigma errors and a matrix thermal
conductivity k_m (W/m/K) used by :mod:`pybasinsubs.thermal`.
"""
from __future__ import annotations
import pandas as pd

_ROWS = [
    # name, phi0, phi0_err, C, C_err, rho_g, rho_g_err, k_m, source
    ("mudstone", 0.4253052, 0.01924488, 1502.024, 44.32066, 2649.0, 43.43, 1.98,
     "Lai et al. (2022) PoFun.csv (fit to Taiwan mudstone curves)"),
    ("sandstone_tw", 0.3734163, 0.01140228, 2860.291, 91.89712, 2666.67, 8.82, 3.16,
     "Lai et al. (2022) PoFun.csv (fit to Taiwan sandstone curves)"),
    ("marine_avg", 0.663, 0.03315, 1333.0, 66.65, 2647.0, 52.94, 2.0,
     "Lai et al. (2022) PoFun.csv (average marine sediment)"),
    ("shale", 0.63, 0.03, 1960.8, 100.0, 2720.0, 30.0, 1.98, "Sclater & Christie (1980): c=0.51 /km"),
    ("sandstone", 0.49, 0.03, 3703.7, 200.0, 2650.0, 20.0, 3.16, "Sclater & Christie (1980): c=0.27 /km"),
    ("chalk", 0.70, 0.03, 1408.5, 80.0, 2710.0, 20.0, 2.90, "Sclater & Christie (1980): c=0.71 /km"),
    ("shaly_sandstone", 0.56, 0.03, 2564.1, 150.0, 2680.0, 20.0, 2.50, "Sclater & Christie (1980): c=0.39 /km"),
]
COLUMNS = ["name", "phi0", "phi0_err", "C", "C_err", "rho_g", "rho_g_err", "k_m", "source"]

# Matrix conductivities: Hantschel & Kauerauf (2009) typical values at 20 C.
WATER_K = 0.6


def default_lithologies() -> pd.DataFrame:
    """Built-in table (Lai et al. 2022 + Sclater & Christie 1980). 1-sigma errors
    for Sclater & Christie are nominal (~5 %) and should be replaced if known."""
    return pd.DataFrame(_ROWS, columns=COLUMNS).set_index("name")


def lithology_table(extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Default table updated/extended by a user table with the same columns."""
    t = default_lithologies()
    if extra is not None:
        e = extra.set_index("name") if "name" in extra.columns else extra
        for c in COLUMNS[1:]:
            if c not in e.columns:
                e[c] = t[c].mean() if c != "source" else "user"
        t = pd.concat([t.drop(index=[i for i in e.index if i in t.index]), e[COLUMNS[1:]]])
    return t
