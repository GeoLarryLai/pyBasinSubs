# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pyBasinSubs - Python backstripping, geohistory and basin thermal history.

Layers
------
legacy      exact port of the BasinSubs R scripts (Lai et al. 2022), delta-method errors
backstrip   generic Monte Carlo 1-D Airy backstripping for any column
thermal     geotherms, McKenzie, Easy%Ro, BHT corrections
flexure     flexural profile through well columns (thin plate, Te fit)
flex2d      2-D section: flexural backstripping, fault restoration, Monte Carlo (package)
faults      fault sticks, detection, user fault files, curved faults (package)
webapp      browser app: 1-D column, 2-D section, fault editor (``pybasinsubs-app``)
constants   single source of physical defaults (rho_m = 3300, E = 70 GPa, ...)
"""
from ._paths import data_path
from .column import Column, read_column, from_basinsubs, from_sodir
from .backstrip import backstrip, BackstripResult

__version__ = "0.2.0"
__all__ = ["data_path", "Column", "read_column", "from_basinsubs", "from_sodir", "backstrip", "BackstripResult"]

from .timestep import (
    timestep_steps, timestep_column, timestep_thermal_fields, timestep_view,
    plot_geohistory_timestep, plot_thermal_timestep,
    save_geohistory_animation, save_thermal_animation,
    thermal_unit_colors, maturity_timestep_data, plot_maturity_timestep, save_maturity_animation,
)

__all__ += ["timestep_steps", "timestep_column", "timestep_thermal_fields", "timestep_view",
            "plot_geohistory_timestep", "plot_thermal_timestep",
            "save_geohistory_animation", "save_thermal_animation",
            "thermal_unit_colors", "maturity_timestep_data", "plot_maturity_timestep", "save_maturity_animation"]

from .section_thermal import section_thermal_data, plot_section_thermal_timestep, save_section_thermal_animation

__all__ += ["section_thermal_data", "plot_section_thermal_timestep", "save_section_thermal_animation"]
