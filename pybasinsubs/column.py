# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Stratigraphic column: the one input object the generic engine needs.

A column is a table with one row per unit, ordered **youngest (top) first**.
Each row describes the unit *below* a dated horizon:

=====================  ==========================================================
column                 meaning
=====================  ==========================================================
unit                   label
age_top_Ma             age of the unit top (end of deposition), Ma
age_top_err_Ma         1-sigma error of that age
age_base_Ma            (optional) start of deposition; default = next row's top age
thickness_m            present (compacted) thickness T0, m. For an *eroded* unit:
                       its thickness just after deposition (at the surface)
thickness_err_m        1-sigma error of thickness
water_depth_m          paleo-water depth W at the time of the unit top, m
water_depth_err_m      1-sigma error of W
sea_level_m            eustatic sea level at that time (+ = higher than today), m
sea_level_err_m        1-sigma error
f_<lithology>          volume fractions of each lithology (normalised to 1)
eroded                 1 if the unit is now eroded (not in the present column)
eroded_at_Ma           when it was removed (blank = after the youngest node, as in
                       BasinSubs, where unit 0 is the eroded unit at 0.5 Ma)
water_depth_at_erosion_m   W at the erosion time (default 0)
=====================  ==========================================================

The last row is normally the *basement* (thickness 0); its age is the start of
the basin record. This mirrors BasinSubs ``Ages_*_new.csv`` (unit 14 has T0=0).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .lithology import lithology_table
from ._paths import data_path

REQUIRED = ["unit", "age_top_Ma", "thickness_m", "water_depth_m"]
DEFAULTS = dict(age_top_err_Ma=0.0, thickness_err_m=0.0, water_depth_err_m=0.0,
                sea_level_m=0.0, sea_level_err_m=0.0, eroded=0, eroded_at_Ma=np.nan,
                water_depth_at_erosion_m=0.0)


@dataclass
class Column:
    units: pd.DataFrame
    lithologies: pd.DataFrame = field(default_factory=lithology_table)
    name: str = "column"
    lat: float = np.nan
    lon: float = np.nan
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        u = self.units.copy()
        miss = [c for c in REQUIRED if c not in u.columns]
        if miss:
            raise ValueError(f"column table is missing {miss}")
        for k, v in DEFAULTS.items():
            if k not in u.columns:
                u[k] = v
            u[k] = u[k].fillna(v) if k != "eroded_at_Ma" else u[k]
        if "age_base_Ma" not in u.columns:
            u["age_base_Ma"] = np.nan
        nxt = u["age_top_Ma"].shift(-1)
        u["age_base_Ma"] = u["age_base_Ma"].fillna(nxt).fillna(u["age_top_Ma"])
        if not len(self.frac_cols(u)):
            raise ValueError("no lithology fraction columns (f_<name>)")
        unknown = [c[2:] for c in self.frac_cols(u) if c[2:] not in self.lithologies.index]
        if unknown:
            raise ValueError(f"lithologies {unknown} not in the lithology table")
        if np.any(np.diff(u["age_top_Ma"].to_numpy()) < 0):
            raise ValueError("rows must be ordered youngest (top) to oldest")
        self.units = u.reset_index(drop=True)

    @staticmethod
    def frac_cols(u):
        return [c for c in u.columns if c.startswith("f_")]

    @property
    def lith_names(self):
        return [c[2:] for c in self.frac_cols(self.units)]

    def fractions(self, normalise=True):
        F = self.units[self.frac_cols(self.units)].fillna(0).to_numpy(float)
        if normalise:
            s = F.sum(1, keepdims=True)
            F = np.where(s > 0, F / np.where(s > 0, s, 1), 0)
        return F

    def __len__(self):
        return len(self.units)

    def to_csv(self, path):
        self.units.to_csv(path, index=False)

    def copy(self, **kw):
        d = dict(units=self.units.copy(), lithologies=self.lithologies.copy(), name=self.name,
                 lat=self.lat, lon=self.lon, meta=dict(self.meta))
        d.update(kw)
        return Column(**d)


def read_column(path, lithology_csv=None, name=None, **kw) -> Column:
    """Read a column CSV (see module docstring); optional extra lithology CSV."""
    u = pd.read_csv(path, encoding="utf-8-sig")
    lt = lithology_table(pd.read_csv(lithology_csv) if lithology_csv else None)
    return Column(u, lt, name=name or os.path.splitext(os.path.basename(path))[0], **kw)


def from_basinsubs(region="N", root=None, litho_mode="per_unit") -> Column:
    """Build a Column from the BasinSubs Coastal Range input CSVs.

    litho_mode='per_unit' uses each unit's fractions from Litho_*.csv (as the
    paper intends); 'r_first_row' copies unit 0's fractions to all units, which
    is what the R script actually does (see :mod:`pybasinsubs.legacy`).
    """
    root = root or data_path("coastal_range")
    j = lambda *p: os.path.join(root, *p)
    A = pd.read_csv(j("Thickness", f"Ages_{region}_new.csv"))
    L = pd.read_csv(j("Thickness", f"Litho_{region}.csv"), encoding="utf-8-sig")
    PB = pd.read_csv(j("Thickness", f"PB_Unit_{region}.csv"))
    SL = pd.read_csv(j("Sealevel", f"SL_Unit_{region}.csv"))
    P = pd.read_csv(j("Backstripping", "PoFun.csv"))
    lt = lithology_table()
    names = {"Mdt": "mudstone", "Sst": "sandstone_tw", "AveMS": "marine_avg"}
    # use the exact PoFun values from the data folder
    for _, r in P.iterrows():
        lt.loc[names[r.Litho] if r.Litho in names else r.Litho, ["phi0", "phi0_err", "C", "C_err", "rho_g", "rho_g_err"]] = \
            [r.Porosity_0, r.Porosity_0_err, r.DecayC, r.DecayC_err, r.rho_g, r.rho_g_err]
    u = pd.DataFrame(dict(unit=A.Units, age_top_Ma=A.Age, age_top_err_Ma=A.Age_err, thickness_m=A.oT,
                          thickness_err_m=A.T0_err, water_depth_m=PB.W, water_depth_err_m=PB.Werr,
                          sea_level_m=SL.dSL, sea_level_err_m=SL.dSLerr))
    for k, v in names.items():
        col = L[k].to_numpy(float)[:len(A)]
        u["f_" + v] = np.repeat(col[0], len(col)) if litho_mode == "r_first_row" else col
    u["eroded"] = (u.unit == 0).astype(int)
    u["eroded_at_Ma"] = np.nan
    return Column(u, lt, name=f"Coastal Range {region}",
                  meta=dict(region=region, present_elevation_m=MAX_ELEV_M[region]))


MAX_ELEV_M = {"N": 1334.0, "S": 1682.0}


def from_sodir(well, transect="viking_graben", root=None, lookup=None, tsc=None) -> Column:
    """Column for one Sodir (Norwegian Offshore Directorate) well from the bundled
    group tops (MD RKB) + ``group_lookup.csv`` (ages, lithology, paleo-W assumptions).

    Depths are converted to metres below seafloor: z = MD - KB - WD (vertical well
    assumed). The deepest group is cut at TD (its thickness is a minimum).
    """
    root = root or data_path("north_sea")
    W = pd.read_csv(os.path.join(root, f"sodir_{transect}_wells.csv"))
    T = pd.read_csv(os.path.join(root, f"sodir_{transect}_group_tops.csv"))
    Lk = lookup if lookup is not None else pd.read_csv(os.path.join(root, "group_lookup.csv"))
    Lk = Lk.set_index("group")
    w = W[W.well == well].iloc[0]
    t = T[T.well == well].sort_values("top_md_m")
    t = t[t.group.isin(Lk.index) | t.group.isin([g.replace(" (Barents)", "") for g in Lk.index])]
    off = w.kb_m + w.water_depth_m
    rows = []
    barents = transect.startswith("barents")
    for _, r in t.iterrows():
        key = r.group + " (Barents)" if barents and (r.group + " (Barents)") in Lk.index else r.group
        L = Lk.loc[key]
        base = min(r.base_md_m, w.td_md_m)
        rows.append(dict(unit=r.group.replace(" GP", ""), age_top_Ma=L.age_top_Ma, age_top_err_Ma=L.age_top_err_Ma,
                         age_base_Ma=L.age_base_Ma, thickness_m=base - r.top_md_m,
                         thickness_err_m=max(5.0, 0.02 * (base - r.top_md_m)),
                         water_depth_m=w.water_depth_m if np.isnan(L.water_depth_m) else L.water_depth_m,
                         water_depth_err_m=10.0 if np.isnan(L.water_depth_err_m) else L.water_depth_err_m,
                         f_shale=L.f_shale, f_sandstone=L.f_sandstone, f_chalk=L.f_chalk, eroded=0))
    if barents:   # reconstructed eroded overburden between Nordland and Sotbakken
        L = Lk.loc["ERODED CENOZOIC"]
        rows.insert(1, dict(unit="ERODED CENOZOIC", age_top_Ma=L.age_top_Ma, age_top_err_Ma=L.age_top_err_Ma,
                            age_base_Ma=L.age_base_Ma, thickness_m=1100.0, thickness_err_m=400.0,
                            water_depth_m=L.water_depth_m, water_depth_err_m=L.water_depth_err_m,
                            f_shale=L.f_shale, f_sandstone=L.f_sandstone, f_chalk=L.f_chalk, eroded=1,
                            eroded_at_Ma=2.7, water_depth_at_erosion_m=0.0))
    last = rows[-1]
    rows.append(dict(unit="base (TD)", age_top_Ma=last["age_base_Ma"], age_top_err_Ma=last["age_top_err_Ma"],
                     thickness_m=0.0, thickness_err_m=0.0, water_depth_m=last["water_depth_m"],
                     water_depth_err_m=last["water_depth_err_m"], f_shale=1.0, f_sandstone=0.0, f_chalk=0.0, eroded=0))
    u = pd.DataFrame(rows)
    # Sea level: Miller et al. (2020) for <66 Ma, 0 +- 100 m (assumed) before that
    from .sealevel import sea_level_at
    sl, sle = sea_level_at(u.age_top_Ma.to_numpy(float), u.age_top_err_Ma.to_numpy(float))
    u["sea_level_m"] = sl; u["sea_level_err_m"] = sle
    return Column(u, lithology_table(), name=well, lat=w.lat, lon=w.lon,
                  meta=dict(transect=transect, water_depth_m=w.water_depth_m, kb_m=w.kb_m,
                            td_md_m=w.td_md_m, bht_degC=w.bht_degC, td_bsf_m=w.td_md_m - off))
