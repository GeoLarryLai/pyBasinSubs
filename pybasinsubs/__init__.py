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

__version__ = "0.3.1"
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

from .diagenesis import evolve_properties, evolve_section_properties, PropertyResult, SectionPropertyResult, default_priors, mechanical_porosity
from . import petrophysics, stress
from .property_plots import (
    plot_geohistory_porosity, plot_porosity_depth_paths,
    plot_property_timeseries, plot_porosity_partition, plot_calibration_layout,
    plot_property_logs, plot_property_logs_snapshots, plot_section_properties,
    property_log_limits, property_log_frame_ages, plot_property_logs_view,
    save_property_logs_animation, plot_section_property, section_property_limits,
    save_section_property_animation, PROPERTY_CMAPS, cmap_for,
)

__all__ += [
    "evolve_properties", "evolve_section_properties",
    "PropertyResult", "SectionPropertyResult", "default_priors", "mechanical_porosity",
    "petrophysics", "stress", "PROPERTY_CMAPS", "cmap_for",
    "plot_geohistory_porosity", "plot_porosity_depth_paths",
    "plot_property_timeseries", "plot_porosity_partition", "plot_calibration_layout",
    "plot_property_logs", "plot_property_logs_snapshots", "plot_section_properties",
    "property_log_limits", "property_log_frame_ages", "plot_property_logs_view",
    "save_property_logs_animation", "plot_section_property", "section_property_limits",
    "save_section_property_animation",
]
