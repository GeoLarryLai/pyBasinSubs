# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyBasinSubs - Python backstripping, geohistory and basin thermal history.

Layers
------
legacy      exact port of the BasinSubs R scripts (Lai et al. 2022), delta-method errors
backstrip   generic Monte Carlo 1-D Airy backstripping for any column
thermal     geotherms, McKenzie, Easy%Ro, BHT corrections
flexure     2-D flexural backstripping, Te fit
constants   single source of physical defaults (rho_m = 3300, E = 70 GPa, ...)
"""
from ._paths import data_path
from .column import Column, read_column, from_basinsubs, from_sodir
from .backstrip import backstrip, BackstripResult

__version__ = "0.1.0"
__all__ = ["data_path", "Column", "read_column", "from_basinsubs", "from_sodir", "backstrip", "BackstripResult"]
