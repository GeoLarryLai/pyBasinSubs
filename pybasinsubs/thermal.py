# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""1-D thermal history, vitrinite maturity and BHT corrections.

Conductivity     k_bulk = k_m^(1-phi) * k_w^phi            (geometric mean)
Steady geotherm  dT/dz = q(z)/k(z),  q(z) = q_s - int_0^z A dz'  (A: radiogenic heat, W/m^3)
                 A constant, an array on the z grid, or A(z) = A0 exp(-z/hr) (hr given):
                 q(z) = q_s - A0 hr (1 - exp(-z/hr))
                 Analytic (uniform k): T = T_s + q_s z/k - A z^2/(2k)                     (constant A)
                                       T = T_s + (q_s - A0 hr) z/k + A0 hr^2 (1 - e^(-z/hr))/k  (exponential)
Transient        rho c dT/dt = d/dz(k dT/dz) + A, implicit finite differences,
                 T(0)=T_surface, -k dT/dz|base = q_base(t) (advection neglected)
McKenzie (1978)  uniform stretching: syn-rift S_i, post-rift
                 S(t) = E0 (beta/pi) sin(pi/beta) (1 - exp(-t/tau)),  E0 = 4 a rho_m alpha T1 / (pi^2 (rho_m - rho_w))
                 q(t) = (k T1/a) [1 + 2 sum_n (beta/(n pi)) sin(n pi/beta) exp(-n^2 t/tau)],  tau = a^2/(pi^2 kappa)
Easy%Ro          Sweeney & Burnham (1990): 20 parallel first-order reactions, A = 1e13 /s,
                 E = 34..72 kcal/mol, %Ro = exp(-1.6 + 3.7 F)
Easy%RoDL        KINETICS PLUG-IN ONLY. Burnham (2016, 2019 Org. Geochem. 131, 50-59) uses
                 A = 2e14 /s with a higher activation-energy distribution, but its weight table
                 could not be traced to a verifiable public copy, so it is NOT implemented here.
                 Supply the table yourself: ``kinetics=dict(A=2e14, E_kcal=[...], f=[...])``.
BHT corrections  Waples & Ramly (2001): f = (-0.1462 ln t + 1.699)/(0.572 z^0.075)
                 Waples et al. (2004b, Danish Central Graben): f = 3.07 t^-0.09 / (0.47 z^0.175)
                 T_true = T_s + f (T_meas - T_s);  Horner (1951) extrapolation for several logs.
"""
from __future__ import annotations
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility
from scipy.linalg import solve_banded

R_KCAL = 1.987204e-3          # kcal/mol/K
MYR = 3.15576e13              # s
SECONDS_PER_MYR = MYR

EASY_RO = dict(A=1.0e13,
               E_kcal=np.arange(34, 74, 2, dtype=float),
               f=np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
                           0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]),
               a=-1.6, b=3.7)
WINDOWS = [("immature", 0.0, 0.6, "#f2f2f2"), ("oil", 0.6, 1.3, "#7fc97f"),
           ("wet gas", 1.3, 2.0, "#fdc086"), ("dry gas", 2.0, 4.0, "#f0027f")]


# ------------------------------------------------------------ conductivity
def bulk_conductivity(phi, k_m, k_w=0.6):
    """Geometric-mean bulk conductivity k = k_m^(1-phi) * k_w^phi (W/m/K)."""
    return k_m ** (1 - phi) * k_w ** phi


def column_profile(result, node=-1, dz=10.0, k_w=0.6, basement_k=2.8, below=5000.0):
    """Porosity and conductivity vs depth below seafloor at a node of a
    :class:`BackstripResult` (best run). Returns z, phi, k, unit index (-1 = basement)."""
    b = result.best; col = result.column
    lt = col.lithologies.loc[col.lith_names]
    F = col.fractions()
    tops, th, eff = b["tops"][0, node], b["thick"][0, node], b["eff_tops"][0, node]
    eps = np.nansum(th)
    z = np.arange(0, eps + below + dz, dz)
    phi = np.zeros_like(z); k = np.full_like(z, basement_k); uid = np.full(len(z), -1)
    for j in np.where(~np.isnan(tops))[0]:
        m = (z >= tops[j]) & (z < tops[j] + th[j])
        zz = z[m] - tops[j] + eff[j]
        ph = np.zeros(m.sum()); lk = np.zeros(m.sum())
        for f, (_, L) in zip(F[j], lt.iterrows()):
            p_ = L.phi0 * np.exp(-zz / L.C)
            ph += f * p_
            lk += f * np.log(L.k_m)
        km = np.exp(lk)
        phi[m] = ph; k[m] = bulk_conductivity(ph, km, k_w); uid[m] = j
    return z, phi, k, uid


# ------------------------------------------------------------ geotherms
def radiogenic_heat(z, A0=1.0e-6, hr=None):
    """Radiogenic heat production A(z) (W/m^3): constant A0, or A0 exp(-z/hr) if hr (m) is given."""
    z = np.asarray(z, float)
    return np.full_like(z, A0) if hr is None else A0 * np.exp(-z / hr)


def steady_geotherm(z, k, q_s, T_s=4.0, A=1.0e-6, hr=None):
    """Steady temperature (C) on grid z (m) for surface heat flow q_s (W/m^2).
    A: constant (W/m^3), an array on z, or with hr (m) the surface value of A0 exp(-z/hr)."""
    z = np.asarray(z, float)
    if hr is not None:
        q = q_s - A * hr * (1 - np.exp(-(z - z[0]) / hr))
    elif np.ndim(A) == 0:
        q = q_s - A * z
    else:
        Aa = np.asarray(A, float)
        q = q_s - np.r_[0, np.cumsum(0.5 * (Aa[1:] + Aa[:-1]) * np.diff(z))]
    dz = np.diff(z)
    grad = q / k
    T = T_s + np.r_[0, np.cumsum(0.5 * (grad[1:] + grad[:-1]) * dz)]
    return T


def transient_geotherm(z, k_of_t, q_base_of_t, times_Ma, T_s=4.0, A=1.0e-6, rhoc_of_t=None, T_init=None, dt_Myr=0.1):
    """Implicit FD solution on a fixed grid (seafloor-attached frame).

    k_of_t(t) -> k(z) ; q_base_of_t(t) -> basal heat flow (W/m^2); times_Ma decreasing.
    Returns T array (len(times), len(z)).
    """
    z = np.asarray(z, float); n = len(z); dz = z[1] - z[0]
    out = []
    t = times_Ma[0]
    k = k_of_t(t)
    A = np.asarray(A, float)
    Aint = A * (z[-1] - z[0]) if A.ndim == 0 else _trapz(A, z)
    T = steady_geotherm(z, k, q_base_of_t(t) + Aint, T_s, A) if T_init is None else T_init
    out.append(T.copy())
    for t_next in times_Ma[1:]:
        nsteps = max(1, int(np.ceil((t - t_next) / dt_Myr)))
        h = (t - t_next) / nsteps * MYR
        for s in range(nsteps):
            tt = t - (s + 1) * (t - t_next) / nsteps
            k = k_of_t(tt); rc = rhoc_of_t(tt) if rhoc_of_t else np.full(n, 2.5e6)
            kh = 0.5 * (k[1:] + k[:-1])
            ab = np.zeros((3, n)); rhs = rc * T / h + A
            ab[1, 1:-1] = rc[1:-1] / h + (kh[1:] + kh[:-1]) / dz ** 2
            ab[0, 2:] = -kh[1:] / dz ** 2
            ab[2, :-2] = -kh[:-1] / dz ** 2
            ab[1, 0] = 1; rhs[0] = T_s
            ab[1, -1] = rc[-1] / h + kh[-1] / dz ** 2 * 2
            ab[2, -2] = -kh[-1] / dz ** 2 * 2
            rhs[-1] = rc[-1] * T[-1] / h + (A if A.ndim == 0 else A[-1]) + 2 * q_base_of_t(tt) / dz
            T = solve_banded((1, 1), ab, rhs)
        t = t_next
        out.append(T.copy())
    return np.array(out)


def burial_temperatures(result, heat_flow=0.06, T_s=4.0, A=1.0e-6, method="steady", dt_Myr=0.1, depth="top", hr=None):
    """Temperature history of every unit top (or 'mid') through the best backstrip run.

    heat_flow: float (W/m^2) or callable(age_Ma) -> W/m^2 (e.g. McKenzie).
    T_s: float or callable(age_Ma) seafloor temperature (C).
    A, hr: radiogenic heat A (W/m^3), or A0 exp(-z/hr) below the seafloor if hr (m) is given.
    Returns dict(age, T (nodes x units), z_bsf (nodes x units)).
    """
    b = result.best
    qf = heat_flow if callable(heat_flow) else (lambda a: heat_flow)
    Tsf = T_s if callable(T_s) else (lambda a: T_s)
    ages = b["age"][0]
    m, n = b["tops"].shape[1:]
    Tm = np.full((m, n), np.nan); Z = np.full((m, n), np.nan)
    zmax = np.nanmax(b["eps"]) + 6000
    profiles = [column_profile(result, node=k_, below=zmax - np.nansum(b["thick"][0, k_])) for k_ in range(m)]
    if method == "steady":
        for k_ in range(m):
            z, phi, k, _ = profiles[k_]
            T = steady_geotherm(z, k, qf(ages[k_]), Tsf(ages[k_]), A, hr)
            zz = b["tops"][0, k_] + (0.5 * b["thick"][0, k_] if depth == "mid" else 0)
            Tm[k_] = np.interp(zz, z, T, right=np.nan); Z[k_] = zz
    else:
        zg = profiles[0][0][: int(zmax / 10) + 1]
        zg = np.arange(0, zmax + 10, 10.0)

        def k_at(age):
            k_ = int(np.argmin(np.abs(ages - age)))
            z, phi, k, _ = profiles[k_]
            return np.interp(zg, z, k)

        def rc_at(age):
            k_ = int(np.argmin(np.abs(ages - age)))
            z, phi, k, _ = profiles[k_]
            ph = np.interp(zg, z, phi)
            return ph * 4.18e6 + (1 - ph) * 2.3e6
        Ts0 = Tsf(ages[0])
        Az = radiogenic_heat(zg, A, hr)
        T = transient_geotherm(zg, k_at, lambda a: qf(a) - _trapz(Az, zg), ages, Ts0, Az, rc_at, dt_Myr=dt_Myr)
        for k_ in range(m):
            zz = b["tops"][0, k_] + (0.5 * b["thick"][0, k_] if depth == "mid" else 0)
            Tm[k_] = np.interp(zz, zg, T[k_], right=np.nan); Z[k_] = zz
    return dict(age=ages, T=Tm, z_bsf=Z)


# ------------------------------------------------------------ McKenzie
from .constants import MCK  # rho_m = constants.RHO_M (3300) by default


def mckenzie_synrift(beta, p=MCK):


    """McKenzie (1978) instantaneous syn-rift (initial) subsidence for stretching factor beta (m), water-loaded."""
    a, T1, al, rm, rc, rw, tc = (p[k] for k in ("a", "T1", "alpha", "rho_m", "rho_c", "rho_w", "tc"))
    return a * ((rm - rc) * (tc / a) * (1 - al * T1 * tc / (2 * a)) - al * T1 * rm / 2) * (1 - 1 / beta) / (rm * (1 - al * T1) - rw)


def mckenzie_postrift(t_Myr, beta, p=MCK, nterms=1):
    """Water-loaded post-rift thermal subsidence (m) t Myr after rifting.
    nterms=1: the usual first-term approximation E0 (beta/pi) sin(pi/beta) (1 - exp(-t/tau)).
    nterms>1: McKenzie (1978) series over odd m = 2j+1,
        S = E0 sum_j (1/m^2) (beta/(m pi)) sin(m pi/beta) (1 - exp(-m^2 t/tau)),
    whose t -> infinity limit is a rho_m alpha T1 (1 - 1/beta) / (2 (rho_m - rho_w))."""
    a, T1, al, rm, rw, kap = (p[k] for k in ("a", "T1", "alpha", "rho_m", "rho_w", "kappa"))
    E0 = 4 * a * rm * al * T1 / (np.pi ** 2 * (rm - rw))
    tau = a ** 2 / (np.pi ** 2 * kap) / MYR
    t = np.maximum(np.asarray(t_Myr, float), 0)
    if nterms <= 1:
        return E0 * beta / np.pi * np.sin(np.pi / beta) * (1 - np.exp(-t / tau))
    m = 2 * np.arange(nterms) + 1.0
    return E0 * np.sum(1 / m ** 2 * beta / (m * np.pi) * np.sin(m * np.pi / beta) * (1 - np.exp(-m ** 2 * t[..., None] / tau)), -1)


def mckenzie_heat_flow(t_Myr, beta, p=MCK, nterms=50):
    """Surface heat flow (W/m^2) t Myr after instantaneous stretching by beta."""
    a, T1, kap, k = p["a"], p["T1"], p["kappa"], p["k"]
    tau = a ** 2 / (np.pi ** 2 * kap) / MYR
    t = np.maximum(np.asarray(t_Myr, float), 0)[..., None]
    nn = np.arange(1, nterms + 1)
    s = np.sum(beta / (nn * np.pi) * np.sin(nn * np.pi / beta) * np.exp(-nn ** 2 * t / tau), -1)
    return k * T1 / a * (1 + 2 * s)


def fit_beta(age_Ma, zeta, rift_end_Ma, z_rift_end=None, p=MCK):
    """Fit beta to post-rift tectonic subsidence (zeta relative to its value at rift end)."""
    from scipy.optimize import minimize_scalar
    age = np.asarray(age_Ma, float); z = np.asarray(zeta, float)
    m = age <= rift_end_Ma
    z0 = np.interp(rift_end_Ma, age[::-1], z[::-1]) if z_rift_end is None else z_rift_end
    obs = z[m] - z0; t = rift_end_Ma - age[m]
    f = lambda b: np.sum((mckenzie_postrift(t, b, p) - obs) ** 2)
    r = minimize_scalar(f, bounds=(1.01, 6), method="bounded")
    return r.x, np.sqrt(r.fun / max(1, m.sum()))


# ------------------------------------------------------------ maturity
def kinetic_conversion(times_Ma, temps_C, kinetics=EASY_RO, substep_Myr=0.01):
    """Fraction F(t) reacted at each input time (times decreasing = forward in time)."""
    t = np.asarray(times_Ma, float); T = np.asarray(temps_C, float)
    E = np.asarray(kinetics["E_kcal"]); f = np.asarray(kinetics["f"]); A = kinetics["A"]
    integ = np.zeros(len(E)); F = [0.0]
    for i in range(1, len(t)):
        ns = max(1, int(np.ceil(abs(t[i - 1] - t[i]) / substep_Myr)))
        dt = (t[i - 1] - t[i]) / ns * MYR
        Ts = np.linspace(T[i - 1], T[i], ns + 1)
        Tm = 0.5 * (Ts[1:] + Ts[:-1]) + 273.15
        integ += A * dt * np.sum(np.exp(-E[:, None] / (R_KCAL * Tm[None, :])), 1)
        F.append(np.sum(f * (1 - np.exp(-integ))))
    return np.array(F)


def easy_ro(times_Ma, temps_C, kinetics=EASY_RO, substep_Myr=0.01):
    """%Ro history along a temperature path (Sweeney & Burnham 1990 by default)."""
    F = kinetic_conversion(times_Ma, temps_C, kinetics, substep_Myr)
    return np.exp(kinetics.get("a", -1.6) + kinetics.get("b", 3.7) * F)


def maturity_window(ro):


    """Name of the maturity window for a %Ro value: immature (<0.6), oil (0.6-1.3), wet gas (1.3-2.0), dry gas (2.0-4.0)."""
    for name, lo, hi, _ in WINDOWS:
        if lo <= ro < hi:
            return name
    return "dry gas" if ro >= 2 else "immature"


# ------------------------------------------------------------ BHT
def bht_waples_ramly(T_meas, z_m, tsc_h=12.0, T_s=4.0):
    """Waples & Ramly (2001) BHT correction: f = (-0.1462 ln t + 1.699) / (0.572 z^0.075), T = T_s + f (BHT - T_s). tsc_h = time since circulation (h), z_m = depth below seafloor (m)."""
    f = (-0.1462 * np.log(np.asarray(tsc_h, float)) + 1.699) / (0.572 * np.asarray(z_m, float) ** 0.075)
    return T_s + f * (np.asarray(T_meas, float) - T_s)


def bht_waples_danish(T_meas, z_m, tsc_h=12.0, T_s=4.0):


    """Waples et al. (2004b) Danish Central Graben BHT correction: f = 3.07 t^-0.09 / (0.47 z^0.175), T = T_s + f (BHT - T_s). tsc_h = time since circulation (h), z_m = depth below seafloor (m)."""
    f = 3.07 * np.asarray(tsc_h, float) ** -0.09 / (0.47 * np.asarray(z_m, float) ** 0.175)
    return T_s + f * (np.asarray(T_meas, float) - T_s)


def bht_correct(T_meas, z_m, tsc_h=None, method="waples_ramly", T_s=4.0, default_tsc_h=12.0):
    """Correct BHTs with a per-measurement time since circulation. tsc_h: scalar, array, or None/NaN
    entries (-> default_tsc_h, flagged). Returns (T_corrected, tsc_used, tsc_assumed_mask)."""
    T_meas = np.atleast_1d(np.asarray(T_meas, float))
    t = np.full(T_meas.shape, np.nan) if tsc_h is None else np.broadcast_to(np.asarray(tsc_h, float), T_meas.shape).copy()
    assumed = ~np.isfinite(t); t[assumed] = default_tsc_h
    fn = {"waples_ramly": bht_waples_ramly, "waples_danish": bht_waples_danish}[method]
    return fn(T_meas, z_m, t, T_s), t, assumed


def bht_horner(t_since_circ_h, temps_C, circ_time_h):
    """Horner plot: regress T on log10((tc+dt)/dt); intercept = static temperature."""
    dt = np.asarray(t_since_circ_h, float)
    x = np.log10((circ_time_h + dt) / dt)
    b, a = np.polyfit(x, np.asarray(temps_C, float), 1)
    return a


def heat_flow_from_temperature(T_C, z_m, z_grid, k_grid, T_s=4.0):
    """q = (T - T_s) / thermal resistance (sum dz/k) down to z."""
    zg = np.asarray(z_grid); kg = np.asarray(k_grid)
    m = zg <= z_m
    Rth = _trapz(1 / kg[m], zg[m])
    return (T_C - T_s) / Rth


# ------------------------------------------------------------ 2-D (age x depth) fields for burial diagrams
def _unit_paths(result, ages_f, nsub=4):
    """Depth-below-sea-level paths (time x tracers) of points inside every unit.

    Each unit is split into ``nsub`` sub-layers; a tracer at fraction f of the
    unit follows top + f*thickness, linearly interpolated between backstrip
    nodes, starting at the paleo-seafloor at the unit's deposition age.
    """
    b = result.best; u = result.column.units
    ag = b["age"][0]; o = np.argsort(ag)
    sea = (b["W"][0] - b["SL"][0])
    sea_f = np.interp(ages_f, ag[o], sea[o])
    paths, ids = [], []
    for j in range(len(u)):
        if u.thickness_m[j] <= 0:
            continue
        top = b["tops"][0, :, j]; th = b["thick"][0, :, j]
        ab = float(u.age_base_Ma[j])
        for f in (np.arange(nsub) + 0.5) / nsub:
            z = top + f * th
            m = ~np.isnan(z)
            a_n = np.r_[ag[m], ab] if ab > ag[m].max() else ag[m]
            z_n = np.r_[z[m], 0.0] if ab > ag[m].max() else z[m]
            oo = np.argsort(a_n)
            zz = np.interp(ages_f, a_n[oo], z_n[oo], left=z_n[oo][0], right=np.nan)
            paths.append(zz + sea_f); ids.append(j)
    return np.array(paths).T, np.array(ids), sea_f


def temperature_field(result, heat_flow=0.06, T_s=4.0, A=1.0e-6, dt_Myr=1.0, dz=25.0, hr=None):
    """Steady-state temperature on an (age, depth below sea level) grid.

    At every backstrip node the geotherm T(z_bsf) = T_s + int (q_s - A z)/k dz
    is computed with the porosity-dependent conductivity of the column at that
    time; geotherms are linearly interpolated in time (in depth below seafloor)
    and shifted by the paleo-seafloor depth (W - dSL). Returns dict(age, z, T,
    seafloor) with T[age, z] (NaN in the water column). hr (m): A0 exp(-z/hr) radiogenic heat.
    """
    b = result.best
    qf = heat_flow if callable(heat_flow) else (lambda a: heat_flow)
    Tsf = T_s if callable(T_s) else (lambda a: T_s)
    ag = b["age"][0]
    zmax = float(np.nanmax(b["sigma"])) + 1500.0
    zb = np.arange(0, zmax + dz, dz)
    G = []
    for k_ in range(len(ag)):
        z, phi, k, _ = column_profile(result, node=k_, below=zmax)
        G.append(np.interp(zb, z, steady_geotherm(z, k, qf(ag[k_]), Tsf(ag[k_]), A, hr)))
    G = np.array(G); o = np.argsort(ag)
    ages_f = np.arange(0, ag.max() + dt_Myr / 2, dt_Myr)[::-1]
    Gf = np.array([np.interp(ages_f, ag[o], G[o, i]) for i in range(len(zb))]).T
    sea = b["W"][0] - b["SL"][0]
    sea_f = np.interp(ages_f, ag[o], sea[o])
    zs = np.arange(min(0.0, sea.min()), zmax + dz, dz)
    T = np.full((len(ages_f), len(zs)), np.nan)
    for i, s in enumerate(sea_f):
        m = zs >= s
        T[i, m] = np.interp(zs[m] - s, zb, Gf[i], right=np.nan)
    return dict(age=ages_f, z=zs, T=T, seafloor=sea_f)


def maturity_field(result, tf, nsub=4):
    """Easy%Ro of tracer points inside every unit along their burial paths.

    tf: output of :func:`temperature_field`. Returns dict(age, z, ro) as flat
    scattered arrays (for tricontour) plus per-tracer arrays.
    """
    ages = tf["age"]
    P, ids, _ = _unit_paths(result, ages, nsub)
    RO = np.full_like(P, np.nan)
    for i in range(P.shape[1]):
        m = ~np.isnan(P[:, i])
        if m.sum() < 2:
            continue
        Ti = np.array([np.interp(P[k, i], tf["z"], np.nan_to_num(tf["T"][k], nan=4.0)) for k in np.where(m)[0]])
        RO[m, i] = easy_ro(ages[m], Ti)
    A = np.repeat(ages[:, None], P.shape[1], 1)
    ok = ~np.isnan(RO)
    return dict(age=A[ok], z=P[ok], ro=RO[ok], paths=P, ro_paths=RO, unit=ids)
