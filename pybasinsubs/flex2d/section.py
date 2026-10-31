# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Section container, well correlation, horizon/CSV import and decompaction helpers."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..constants import RHO_W


@dataclass
class Section:
    x: np.ndarray                 # (nx,) m
    units: list                   # youngest first
    age_top: np.ndarray           # (nu,)
    age_base: np.ndarray
    h: np.ndarray                 # (nu, nx) present thickness
    F: np.ndarray                 # (nu, nx, nl) lithology fractions
    W: np.ndarray                 # (nu, nx) paleo-water depth at end of deposition of unit (top age)
    SL: np.ndarray                # (nu,) sea level at top age of unit
    W0: np.ndarray                # (nx,) present water depth
    phi0: np.ndarray; C: np.ndarray; rho_g: np.ndarray   # (nl,)
    xw: np.ndarray = None
    wells: list = field(default_factory=list)
    lith: list = field(default_factory=list)     # lithology names of the last axis of F (phi0, C, rho_g order)


def correlate(columns, x_wells, dx=1000.0, pad=0.0):
    """Build a :class:`Section` from Column objects (wells) at distances x_wells (m)."""
    xw = np.asarray(x_wells, float); o = np.argsort(xw); xw = xw[o]; columns = [columns[i] for i in o]
    x = np.arange(xw[0] - pad, xw[-1] + pad + dx / 2, dx)
    lith = list(dict.fromkeys(l for c in columns for l in c.lith_names))
    lt = columns[0].lithologies.loc[lith]
    info = {}
    for c in columns:
        u = c.units
        for j in range(len(u)):
            if u.thickness_m[j] > 0:
                info.setdefault(u.unit[j], (u.age_top_Ma[j], u.age_base_Ma[j]))
    names = sorted(info, key=lambda n: info[n][0] + 1e-3 * info[n][1])
    nu, nw, nl = len(names), len(columns), len(lith)
    h = np.zeros((nu, nw)); F = np.zeros((nu, nw, nl)); W = np.full((nu, nw), np.nan); SL = np.zeros(nu)
    for i, c in enumerate(columns):
        u = c.units; Fc = c.fractions()
        for j in range(len(u)):
            n = u.unit[j]
            if n not in info:
                continue
            k = names.index(n)
            h[k, i] = u.thickness_m[j]
            for m, ln in enumerate(c.lith_names):
                F[k, i, lith.index(ln)] = Fc[j, m]
            W[k, i] = u.water_depth_m[j]; SL[k] = u.sea_level_m[j]
    # fill fractions / water depth where a unit pinches out with the mean of wells that have it
    for k in range(nu):
        has = h[k] > 0
        F[k, ~has] = F[k, has].mean(0); W[k, np.isnan(W[k])] = np.nanmean(W[k])
    I = lambda v: np.interp(x, xw, v)
    hx = np.array([I(h[k]) for k in range(nu)])
    Fx = np.stack([np.stack([I(F[k, :, l]) for l in range(nl)], -1) for k in range(nu)])
    Fx /= Fx.sum(-1, keepdims=True)
    Wx = np.array([I(W[k]) for k in range(nu)])
    return Section(x=x, units=names, age_top=np.array([info[n][0] for n in names]),
                   age_base=np.array([info[n][1] for n in names]), h=hx, F=Fx, W=Wx, SL=SL, W0=Wx[0].copy(),
                   phi0=lt.phi0.to_numpy(float), C=lt.C.to_numpy(float), rho_g=lt.rho_g.to_numpy(float),
                   xw=xw, wells=[c.name for c in columns], lith=list(lith))


def _solid(T, D, F, phi0, C):
    """Grain thickness of layers [D, D+T] (arrays nx) with mixed lithology F (nx, nl)."""
    pore = (F * C * phi0 * np.exp(-D[:, None] / C) * (1 - np.exp(-T[:, None] / C))).sum(-1)
    return T - pore


def _thickness(Gs, D, F, phi0, C, it=80):
    """Invert _solid for T given grain thickness Gs at top depth D (Newton)."""
    T = Gs.copy() * 1.5
    for _ in range(it):
        f = _solid(T, D, F, phi0, C) - Gs
        dfdT = 1 - (F * phi0 * np.exp(-(D[:, None] + T[:, None]) / C)).sum(-1)
        T = np.maximum(T - f / dfdT, Gs)
        if np.max(np.abs(f)) < 1e-6:
            break
    return np.where(Gs > 0, T, 0.0)


def _mass(T, D, F, phi0, C, rho_g, rho_w=RHO_W):
    """Mass per unit area of layers: rho_g G + rho_w (T - G)."""
    Gs = _solid(T, D, F, phi0, C)
    return (F * rho_g).sum(-1) * Gs + rho_w * (T - Gs)


def section_from_horizons(x, tops, base, units, age_top, age_base, lith, seafloor, lith_table=None):
    """Build a :class:`Section` from digitized horizon depths (m below sea level).

    tops: (nu, nx) top of each unit (youngest first); base: (nx,) base of the oldest unit;
    lith: list of dicts {lithology: fraction} per unit. Paleo-water depths are not known: W = 0
    (only reverse post-rift / fault restoration uses this section, not bathymetric mode)."""
    from ..lithology import lithology_table
    lt = lithology_table() if lith_table is None else lith_table
    names = list(dict.fromkeys(k for d in lith for k in d)); lt = lt.loc[names]
    tops = np.asarray(tops, float); nu, nx = tops.shape
    bot = np.vstack([tops[1:], base[None]])
    h = np.maximum(bot - tops, 0.0)
    F = np.zeros((nu, nx, len(names)))
    for k, d in enumerate(lith):
        for n, v in d.items():
            F[k, :, names.index(n)] = v
    F /= F.sum(-1, keepdims=True)
    return Section(x=np.asarray(x, float), units=list(units), age_top=np.asarray(age_top, float),
                   age_base=np.asarray(age_base, float), h=h, F=F, W=np.zeros((nu, nx)), SL=np.zeros(nu),
                   W0=np.asarray(seafloor, float), phi0=lt.phi0.to_numpy(float), C=lt.C.to_numpy(float),
                   rho_g=lt.rho_g.to_numpy(float), xw=np.array([]), wells=[], lith=names)


# ---------------------------------------------------------------- CSV import / export (2-D template system)
SECTION_TEMPLATE = ("template", "my_section_horizons.csv"), ("template", "my_section_units.csv")


def read_section_template():
    """Paths of the example 2-D template files (horizons, units) and the matching fault-stick CSV."""
    from .._paths import data_path
    return dict(horizons=data_path(*SECTION_TEMPLATE[0]), units=data_path(*SECTION_TEMPLATE[1]),
                faults=data_path("template", "my_section_faults.csv"))


def _read(obj):
    import io
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (bytes, bytearray)):
        obj = io.BytesIO(obj)
    return pd.read_csv(obj, encoding="utf-8-sig") if not hasattr(obj, "read") else pd.read_csv(obj)


def section_from_csv(horizons, units, dx=None, lith_table=None, base_name="base"):
    """Build a :class:`Section` from two CSV tables (paths, bytes, file objects or DataFrames).

    horizons (long format, one row per pick): ``x_m`` (distance along the section, m; or ``x_km``), ``unit``
    (name of the unit whose TOP is picked), ``depth_m`` (m below sea level, positive down); optional ``well``.
    The base of the oldest unit is a horizon named ``base_name`` (default "base"). The top of the youngest
    unit is the present seafloor. A unit not picked at some x pinches out there (zero thickness).

    units (same columns as the 1-D column template, one row per unit, youngest first): ``unit``,
    ``age_top_Ma``, ``age_base_Ma`` (optional: next unit's top), lithology fractions ``f_<name>`` (names from
    :func:`pybasinsubs.lithology.lithology_table`); optional ``water_depth_m`` (paleo-water depth at the end of
    deposition, default 0) and ``sea_level_m`` (default 0). Lithology and paleo-water depth are uniform
    along the section for each unit (first version).

    dx: regular grid spacing (m); default = the spacing of the picks if regular, else 250 m.
    Templates: :func:`read_section_template`."""
    import pandas as pd
    from ..lithology import lithology_table
    H = _read(horizons); U = _read(units)
    if "x_m" not in H.columns and "x_km" in H.columns:
        H["x_m"] = H["x_km"] * 1e3
    for c in ("x_m", "unit", "depth_m"):
        if c not in H.columns:
            raise ValueError(f"horizons table is missing column {c!r} (need x_m, unit, depth_m)")
    for c in ("unit", "age_top_Ma"):
        if c not in U.columns:
            raise ValueError(f"units table is missing column {c!r}")
    U = U[~U.unit.astype(str).str.lower().str.startswith(base_name.lower())].reset_index(drop=True)
    if np.any(np.diff(U.age_top_Ma.to_numpy(float)) < 0):
        raise ValueError("units must be ordered youngest (top) to oldest")
    fcols = [c for c in U.columns if c.startswith("f_")]
    if not fcols:
        raise ValueError("units table has no lithology fraction columns (f_<name>)")
    lt = lithology_table() if lith_table is None else lith_table
    unknown = [c[2:] for c in fcols if c[2:] not in lt.index]
    if unknown:
        raise ValueError(f"lithologies {unknown} not in the lithology table")
    H["unit"] = H["unit"].astype(str); names = U.unit.astype(str).tolist()
    missing = [n for n in set(H.unit) if n not in names and n.lower() != base_name.lower()]
    if missing:
        raise ValueError(f"horizon units {sorted(missing)} not in the units table")
    if not (H.unit.str.lower() == base_name.lower()).any():
        raise ValueError(f"no base horizon (unit = {base_name!r}) in the horizons table")
    xs = np.unique(H.x_m.to_numpy(float))
    if dx is None:
        d = np.diff(xs); dx = float(d[0]) if len(d) and np.allclose(d, d[0]) else 250.0
    x = np.arange(xs[0], xs[-1] + dx / 2, dx)
    seq = names + [base_name]
    Z = np.full((len(seq), len(x)), np.nan)
    for k, n in enumerate(seq):
        h = H[H.unit.str.lower() == n.lower()].sort_values("x_m")
        if len(h):
            z = np.interp(x, h.x_m.to_numpy(float), h.depth_m.to_numpy(float), left=np.nan, right=np.nan)
            z[(x < h.x_m.min() - 1e-6) | (x > h.x_m.max() + 1e-6)] = np.nan
            Z[k] = z
    if np.isnan(Z[0]).any() or np.isnan(Z[-1]).any():
        raise ValueError("the youngest unit (seafloor) and the base must be picked along the whole section")
    for k in range(len(seq) - 2, 0, -1):                 # unpicked tops pinch out onto the next older surface
        Z[k] = np.where(np.isnan(Z[k]), Z[k + 1] if not np.isnan(Z[k + 1]).all() else Z[-1], Z[k])
    for k in range(1, len(seq)):                         # older surfaces cannot lie above younger ones
        Z[k] = np.maximum(Z[k], Z[k - 1])
    age_top = U.age_top_Ma.to_numpy(float)
    age_base = (U.age_base_Ma.to_numpy(float) if "age_base_Ma" in U.columns else np.full(len(U), np.nan))
    nxt = np.r_[age_top[1:], np.nan]; age_base = np.where(np.isnan(age_base), nxt, age_base)
    age_base = np.where(np.isnan(age_base), age_top, age_base)
    lith = [{c[2:]: float(v) for c, v in zip(fcols, U.loc[j, fcols].fillna(0.0)) if v > 0} for j in range(len(U))]
    if any(not d for d in lith):
        raise ValueError("every unit needs at least one non-zero lithology fraction")
    sec = section_from_horizons(x, Z[:-1], Z[-1], names, age_top, age_base, lith, Z[0], lith_table=lt)
    # keep every lithology column of the table, also those with zero fractions
    if "water_depth_m" in U.columns:
        sec.W = np.repeat(U.water_depth_m.fillna(0.0).to_numpy(float)[:, None], len(x), 1)
    if "sea_level_m" in U.columns:
        sec.SL = U.sea_level_m.fillna(0.0).to_numpy(float)
    if "well" in H.columns:
        w = H.dropna(subset=["well"]).groupby("well").x_m.first().sort_values()
        sec.xw = w.to_numpy(float); sec.wells = w.index.astype(str).tolist()
    _ = pd                                                # (pandas used by _read)
    return sec


def section_to_csv(sec, horizons_path=None, units_path=None, base_name="base", every=1):
    """Write a Section to the two template tables (inverse of :func:`section_from_csv`).
    Returns (horizons DataFrame, units DataFrame); writes CSV files if paths are given."""
    import pandas as pd
    tops = sec.W0 + np.r_[np.zeros((1, sec.h.shape[1])), np.cumsum(sec.h, 0)]
    names = list(sec.units) + [base_name]
    xs = sec.x[::every]
    H = pd.DataFrame([dict(x_m=float(xx), unit=n, depth_m=float(tops[k][j * every]))
                      for k, n in enumerate(names) for j, xx in enumerate(xs)])
    lith = sec.lith or [f"lith{k}" for k in range(sec.F.shape[-1])]
    rows = []
    for k, n in enumerate(sec.units):
        r = dict(unit=n, age_top_Ma=float(sec.age_top[k]), age_base_Ma=float(sec.age_base[k]))
        fr = sec.F[k].mean(0)
        r.update({f"f_{ln}": float(v) for ln, v in zip(lith, fr)})
        r["water_depth_m"] = float(np.mean(sec.W[k])); r["sea_level_m"] = float(sec.SL[k])
        rows.append(r)
    U = pd.DataFrame(rows)
    if horizons_path:
        H.to_csv(horizons_path, index=False)
    if units_path:
        U.to_csv(units_path, index=False)
    return H, U
