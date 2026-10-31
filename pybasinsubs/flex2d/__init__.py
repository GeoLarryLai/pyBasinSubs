# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""True 2-D flexural backstripping of a correlated well section.

Method (Kusznir et al. 1995; Roberts et al. 1998; Watts 2001, ch. 6-7;
Allen & Allen 2013, ch. 9; plate solutions after Turcotte & Schubert 2014, s.3.17)
------------------------------------------------------------------------------
1. Correlate units by name between wells; interpolate each unit's present
   thickness h_j(x) and lithology fractions onto a regular grid x (a unit absent
   in a well has h=0 there -> pinch-out).
2. For each age t_i = top age of unit i: strip units younger than t_i, and
   restack the rest from the seafloor, decompacting every column with grain
   (solid) volume conservation
       G_j = T_j - sum_k F_jk C_k phi0_k e^{-D_j/C_k} (1 - e^{-T_j/C_k}),
   phi(z) = sum_k F_k phi0_k e^{-z/C_k}  (mixed porosity law).
3. Sediment load (water-filled basin):
       q(x) = (rho_s(x) - rho_w) g S(x),  rho_s = bulk density of the column.
4. Plate response (thin elastic plate, continuous or broken):
       D w'''' + (rho_m - rho_w) g w = q,   D = E Te^3 / 12(1-nu^2).
5. Water-loaded (tectonic) subsidence along the section
       Z(x,t) = W(x,t) - dSL(t) rho_m/(rho_m-rho_w) + S(x,t) - w(x,t)
   which equals the Airy 1-D formula when Te -> 0 (w -> q/((rho_m-rho_w)g)).
6. Restored sections:
   * 'bathymetric': decompacted stack hung below the paleo-seafloor W - dSL
     (paleo-water depth from the well data, as in 1-D backstripping);
   * 'unloaded': present basement minus the flexural rebound
     dw = w_0 - w_t  and the removed sediment, i.e. pure isostatic unloading
     with no thermal-subsidence reversal (Roberts et al. 1998 step 1-3).

Package layout (``from pybasinsubs import flex2d`` and every former
``flex2d.<name>`` keep working):
  plate        thin-plate solutions (plate_deflection, line_load_analytic, ...)
  montecarlo   MonteCarlo2D, monte_carlo_section, plot_mc_section (seeded envelopes)
  section      Section, correlate, section_from_horizons, section_from_csv, decompaction helpers
  restoration  backstrip_section, reverse_postrift, fit_beta_section, te_sensitivity
  cantilever   Fault, planar flexural-cantilever restoration (restore_faulted), fault_at, faults_from_offsets
  kinematic    listric / domino restoration (restore_kinematic)
  metrics      restoration_metrics, layer_areas
  result       Restoration2D (dict-style access kept)
  montecarlo   monte_carlo_section (2-D uncertainty envelopes)
"""
# ruff: noqa: F401  (re-exports keep the pre-Stage-3 flex2d namespace)
from __future__ import annotations
import os
import numpy as np

from .plate import flexural_parameter, line_load_analytic, _fd_plate, plate_deflection
from .section import (Section, correlate, _solid, _thickness, _mass, section_from_horizons, section_from_csv,
                      section_to_csv, read_section_template)
from .restoration import Restoration, backstrip_section, te_sensitivity, reverse_postrift, fit_beta_section
from .cantilever import (Fault, cantilever_displacement, _block_widths, fault_slips, _heave_shift, restore_faulted,
                         faults_from_offsets, fault_at)
from .kinematic import (listric_depth, fit_listric, fit_listric_stick, _shear_subsidence, _fit_heave, _composite,
                        _apply_listric, _apply_domino, _growth_target, _flexural_coupling, restore_kinematic)
from .metrics import restoration_metrics, layer_areas
from .result import Restoration2D
from .montecarlo import MonteCarlo2D, monte_carlo_section, plot_mc_section
from ..flexure import G, rigidity, deflection_fft
from ..constants import RHO_W, RHO_M, RHO_C, MCK


def restore(sec: Section, faults, method="cantilever", **kw):
    """Single entry point for fault restoration. method:
    'cantilever' - planar flexural cantilever (restore_faulted; Kusznir et al. 1991, 1995),
    'listric'    - listric fault + inclined antithetic simple shear (alpha, default 65 deg), slip per step from growth strata,
    'domino'     - rigid domino rotation on planar faults, rotation per step from growth strata.
    faults may also be a source string (faults.load_faults): 'auto' / 'curved_midramp' (auto curved faults, mid-ramp),
    'curved_crest', 'straight' (planar auto picks), 'user' (stored user fault file, auto if absent) or a path.
    Extra keywords go to the underlying function."""
    if isinstance(faults, (str, os.PathLike)):
        from .. import faults as _F
        tops = sec.W0 + np.r_[np.zeros((1, sec.h.shape[1])), np.cumsum(sec.h[:-1], 0)]
        base = sec.W0 + sec.h.sum(0)
        faults, _src = _F.load_faults(faults, sec.x, tops, base, Hd=kw.get("Hd", 10e3))
    if method == "cantilever":
        kw.pop("alpha", None)
        return restore_faulted(sec, faults, **kw)
    if method in ("listric", "domino"):
        return restore_kinematic(sec, faults, method=method, **kw)
    raise ValueError(f"unknown method {method!r}; use 'cantilever', 'listric' or 'domino'")
