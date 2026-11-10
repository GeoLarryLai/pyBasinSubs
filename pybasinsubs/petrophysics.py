# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Permeability from porosity and lithology (phase 1).

Sands (Kozeny–Carman)
---------------------
Kozeny (1927) and Carman (1937) related permeability to a hydraulic radius.
With spherical grains of diameter d the commonly used form is

    k = (d^2 / C_kc)  φ^3 / (1 - φ)^2                                       (1)

with C_kc = 180 (Carman 1937; the equivalent constant 150 is also used).
d is a single representative grain size (d50). Panda & Lake (1994, AAPG Bulletin
78, 1028–1039) extend (1) with the variance and skewness of the grain-size
distribution; that extension is not applied here (phase 1 uses d50 only).

Chalk (Kozeny / Mortensen / Fabricius)
-------------------------------------
A spherical-grain Kozeny–Carman evaluation with an assumed nannofossil d50
(~5 μm) overestimates North Sea chalk permeability. The documented chalk form
is the original Kozeny (1927) expression used for North Sea chalk by Mortensen,
Engstrøm & Lind (1998, SPE Reservoir Evaluation & Engineering, SPE-31062-PA)
and reviewed by Fabricius (2007, Bulletin of the Geological Society of Denmark
55, 97–128):

    k = c  φ^3 / S^2                                                        (3)

Fabricius (2007) reports Mortensen's Kozeny factor c as 0.21 at φ = 0.20 and
0.27 at φ = 0.60; Kozeny's original constant is about 0.25. Phase 1 uses that
constant c = 0.25 rather than reconstructing Mortensen's full arccos
expression. Specific surface per bulk volume follows Fabricius (2007, eq. 19):

    S = S_BET  ρ_g  (1 - φ)                                                 (4)

S_BET is the BET specific surface. Fabricius (2007) reports calcite-contribution
values from about 0.9 m^2 g^{-1} (Dan/Gorm Tor) to about 2 m^2 g^{-1} (Tyra);
total BET is higher when silicates are present. The default S_BET = 2.0 m^2 g^{-1}
is a literature prior in that published North Sea chalk range, not a measurement
at a bundled well. Grain density defaults to 2710 kg m^{-3} (calcite).

Mallon, Swarbrick & Katsube (2005, Geology 33, 21–24) show that non-reservoir
North Sea chalk (φ about 5–20 %) can be 10^{-22}–10^{-18} m^2 with only a weak
φ–k relation. That paper does not give a closed-form equation, so it is not
fitted here. Equation (3) is intended for chalk with a measurable matrix pore
system and will overestimate tightly cemented non-reservoir chalk.

Mudstones (Yang & Aplin 2010)
-----------------------------
Yang & Aplin (2010, Marine and Petroleum Geology 27, 1692–1697) give bedding-
perpendicular permeability K (m^2) as a function of void ratio
e = φ / (1 - φ) and clay fraction CF (mass fraction of particles < 2 μm):

    ln K = a + b e + c e^{1/2}                                              (2)

    a = -69.59 - 26.79 CF + 44.07 CF^{1/2}
    b = -53.61 - 80.03 CF + 132.78 CF^{1/2}
    c =  86.61 + 81.91 CF - 163.61 CF^{1/2}

The last term is e^{1/2} as published, not e^2.

Mixed units
-----------
For a unit that is a mixture of lithologies, k is the weighted geometric mean
of the end-member permeabilities evaluated at the same porosity:
k = exp(Σ f_i ln k_i). This is an assumption, not a measurement.
"""
from __future__ import annotations
import numpy as np

KC_FACTOR = 180.0
# Yang & Aplin (2010) coefficients for a, b, c as linear + sqrt(CF) terms.
_YA = dict(a=(-69.59, -26.79, 44.07),
           b=(-53.61, -80.03, 132.78),
           c=(86.61, 81.91, -163.61))

# Kozeny (1927) constant ~0.25 (Fabricius 2007); Mortensen et al. (1998) c(φ)
# varies only from ~0.21 (φ=0.20) to ~0.27 (φ=0.60).
CHALK_KOZENY_C = 0.25
# Literature prior in the Fabricius (2007) North Sea chalk BET range (m^2/g).
CHALK_SBET_M2_G = 2.0
CHALK_RHO_G = 2710.0  # kg/m^3, calcite
CHALK_LITHOS = frozenset({"chalk", "carbonate", "limestone", "carbonate_chalk"})


def kozeny_carman(phi, grain_size_m, factor=KC_FACTOR):
    """Absolute permeability (m^2) from the Kozeny–Carman spherical-grain form."""
    phi = np.clip(np.asarray(phi, float), 1e-6, 0.95)
    d = np.maximum(np.asarray(grain_size_m, float), 1e-9)
    return (d ** 2 / float(factor)) * (phi ** 3) / (1.0 - phi) ** 2


def yang_aplin_2010(phi, clay_fraction):
    """Bedding-perpendicular mudstone permeability (m^2); Yang & Aplin (2010)."""
    phi = np.clip(np.asarray(phi, float), 1e-6, 0.95)
    cf = np.clip(np.asarray(clay_fraction, float), 0.01, 0.99)
    e = phi / (1.0 - phi)
    s = np.sqrt(cf)
    a = _YA["a"][0] + _YA["a"][1] * cf + _YA["a"][2] * s
    b = _YA["b"][0] + _YA["b"][1] * cf + _YA["b"][2] * s
    c = _YA["c"][0] + _YA["c"][1] * cf + _YA["c"][2] * s
    return np.exp(a + b * e + c * np.sqrt(e))


def chalk_specific_surface(phi, s_bet_m2_g=CHALK_SBET_M2_G, rho_g=CHALK_RHO_G):
    """Specific surface per bulk volume S (1/m); Fabricius (2007) eq. 19.

    S = S_BET · ρ_g · (1 − φ). ``s_bet_m2_g`` is in m²/g and ``rho_g`` in
    kg/m³, so S = s_bet_m2_g × 1000 × rho_g × (1 − φ).
    """
    phi = np.clip(np.asarray(phi, float), 1e-6, 0.95)
    s_bet = np.asarray(s_bet_m2_g, float)
    rho = np.asarray(rho_g, float)
    if np.any(s_bet <= 0) or np.any(rho <= 0):
        raise ValueError("s_bet_m2_g and rho_g must be positive")
    return s_bet * 1000.0 * rho * (1.0 - phi)


def mortensen_fabricius_chalk(phi, s_bet_m2_g=CHALK_SBET_M2_G,
                              rho_g=CHALK_RHO_G, c=CHALK_KOZENY_C):
    """Chalk permeability (m²) from k = c φ³ / S².

    Kozeny (1927) as applied to North Sea chalk by Mortensen, Engstrøm & Lind
    (1998, SPE Reservoir Evaluation & Engineering, SPE-31062-PA) and reviewed
    by Fabricius (2007, Bulletin of the Geological Society of Denmark 55,
    97–128). Default ``c=0.25`` is Kozeny's original constant (Fabricius 2007);
    default ``s_bet_m2_g=2.0`` is a literature prior in the published North Sea
    chalk BET range, not a well measurement.
    """
    phi = np.clip(np.asarray(phi, float), 1e-6, 0.95)
    c = float(c)
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Kozeny factor c must be a positive finite number")
    S = chalk_specific_surface(phi, s_bet_m2_g=s_bet_m2_g, rho_g=rho_g)
    return c * (phi ** 3) / (S ** 2)


def permeability(phi, grain_size_m, clay_fraction, lithology="sandstone",
                 kc_factor=KC_FACTOR, s_bet_m2_g=CHALK_SBET_M2_G,
                 rho_g=CHALK_RHO_G, chalk_c=CHALK_KOZENY_C):
    """Permeability (m^2) for one named end-member lithology.

    Mudstones use Yang & Aplin (2010); chalk/carbonate uses the Mortensen /
    Fabricius Kozeny form; sand and other clastics use spherical-grain
    Kozeny–Carman.
    """
    name = str(lithology).lower()
    if name in {"shale", "mudstone", "mud"}:
        return yang_aplin_2010(phi, clay_fraction)
    if name in CHALK_LITHOS:
        return mortensen_fabricius_chalk(phi, s_bet_m2_g=s_bet_m2_g,
                                         rho_g=rho_g, c=chalk_c)
    return kozeny_carman(phi, grain_size_m, factor=kc_factor)


def mixed_permeability(phi, fractions, grain_size_m, clay_fraction, names, kc_factor=KC_FACTOR):
    """Geometric-mean permeability of a lithology mixture at one porosity.

    fractions, grain_size_m, clay_fraction and names are length-K sequences.
    """
    f = np.asarray(fractions, float)
    f = np.where(f < 0, 0.0, f)
    s = f.sum()
    if s <= 0:
        return np.nan
    f = f / s
    logs = []
    wts = []
    for fi, d, cf, name in zip(f, grain_size_m, clay_fraction, names):
        if fi <= 0:
            continue
        k = permeability(phi, d, cf, name, kc_factor=kc_factor)
        k = np.maximum(k, 1e-32)
        logs.append(np.log(k))
        wts.append(fi)
    return float(np.exp(np.dot(wts, logs)))


def m2_to_md(k_m2):
    """Convert m^2 to millidarcy (1 mD = 9.869233e-16 m^2)."""
    return np.asarray(k_m2, float) / 9.869233e-16
