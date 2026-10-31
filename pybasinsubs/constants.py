# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Physical constants and default parameters, in one place (SI units).

Every module imports its defaults from here, so a single value is used for
1-D backstripping, 2-D flexure, fault restoration and McKenzie thermal
subsidence. All functions that use them also accept them as arguments.

==========  ========  =====================================================
name        value     note
==========  ========  =====================================================
G           9.81      gravity (m/s^2)
RHO_W       1025      sea water (kg/m^3); BasinSubs value
RHO_W_ERR   2         1-sigma of RHO_W used in Monte Carlo / delta method
RHO_M       3300      asthenosphere / mantle (kg/m^3). BasinSubs (Lai et al.
                      2022) and the Roberts et al. (1998) isostasy use 3300;
                      kept as default for R parity. McKenzie (1978) used
                      3330 for the thermal model: pass
                      ``p=dict(MCK, rho_m=3330)`` to reproduce it (~1 %
                      change in thermal subsidence).
RHO_M_ERR   0         1-sigma of RHO_M (BasinSubs treats it as exact)
RHO_C       2800      crust (kg/m^3)
E_YOUNG     70e9      Young's modulus (Pa). Common choice for the lithosphere
                      (e.g. Watts 2001). Roberts et al. (1998) use 100 GPa;
                      since D = E Te^3 / 12(1-nu^2), Te(70 GPa) =
                      Te(100 GPa) * (100/70)^(1/3) = 1.126 Te(100 GPa).
NU          0.25      Poisson's ratio
==========  ========  =====================================================
"""
G = 9.81
RHO_W, RHO_W_ERR = 1025.0, 2.0
RHO_M, RHO_M_ERR = 3300.0, 0.0
RHO_C = 2800.0
E_YOUNG, NU = 70e9, 0.25

# McKenzie (1978) lithosphere parameters; rho_m/rho_c/rho_w follow the constants above
MCK = dict(a=125e3, T1=1333.0, alpha=3.28e-5, rho_m=RHO_M, rho_c=RHO_C, rho_w=RHO_W,
           tc=31.2e3, kappa=1e-6, k=3.138)


def te_equivalent(Te, E_from, E_to=E_YOUNG):
    """Elastic thickness giving the same rigidity under a different Young's modulus."""
    return Te * (E_from / E_to) ** (1.0 / 3.0)
