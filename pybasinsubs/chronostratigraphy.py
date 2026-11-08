# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Offline Phanerozoic age lookup from official ICS chart 2026/06.

Cohen, K.M., Harper, D.A.T., Gibbard, P.L. & Car, N. (2025), Episodes
48(1):105–115, https://doi.org/10.18814/epiiugs/2025/025001.
Source: https://stratigraphy.org/chart and its official static 2026/06 PDF.
Explicit numeric ages ALWAYS win; this module never revises a Section's ages.
"""
import json
import re

import numpy as np
import pandas as pd

from ._paths import data_path


def chronostratigraphic_dataset():
    """Return fresh metadata and records, including source, chart version and year."""
    with open(data_path("chronostratigraphy", "ics_2026_06.json"), encoding="utf-8") as stream:
        return json.load(stream)


def age_bracket(name):
    """Look up (youngest, oldest) Ma, or the union of hyphen-separated ICS names.

    British Palaeocene/Palaeogene and Early/Late synonyms are recognized.
    Formation qualifiers in parentheses are ignored. Regional stages (e.g.
    Ryazanian), unknown formations and Precambrian units require explicit ages;
    no regional-to-ICS equivalence is invented.
    """
    data = chronostratigraphic_dataset()
    records = {row["name"].casefold(): row for row in data["units"]}
    n = re.sub(r"\([^)]*\)", "", str(name)).strip().casefold()
    n = n.replace("palaeocene", "paleocene").replace("palaeogene", "paleogene")
    n = n.replace("late ", "upper ").replace("early ", "lower ")
    if n in records:
        row = records[n];return float(row["age_top_Ma"]), float(row["age_base_Ma"])
    parts = [v.strip() for v in re.split(r"[–—-]", n)]
    if len(parts) > 1:
        values = [(0., 0.) if v in ("present", "recent") else age_bracket(v) for v in parts]
        return min(v[0] for v in values), max(v[1] for v in values)
    raise ValueError(f"No official ICS 2026/06 bracket for {name!r}; supply explicit numeric ages")


def resolve_unit_ages(names, age_top=None, age_base=None):
    """Build an ages table for name-only inputs; preserve every provided number.

    The returned ``unit, age_top_Ma, age_base_Ma`` columns can be joined to a
    section-units input table. Source provenance is per row; dataset metadata is
    in DataFrame.attrs. This is opt-in, not a silent replacement of model ages.
    """
    names = list(names);n = len(names)
    tops = np.full(n, np.nan) if age_top is None else np.broadcast_to(np.asarray(age_top, float), (n,)).copy()
    bases = np.full(n, np.nan) if age_base is None else np.broadcast_to(np.asarray(age_base, float), (n,)).copy()
    rows = []
    for name, top, base in zip(names, tops, bases):
        source = "provided numeric ages"
        if np.isnan(top) or np.isnan(base):
            t, b = age_bracket(name)
            source = "ICS 2026/06" if np.isnan(top) and np.isnan(base) else "provided + ICS 2026/06 missing bound"
            top = t if np.isnan(top) else top;base = b if np.isnan(base) else base
        if not np.isfinite([top, base]).all() or top < 0 or base <= top:
            raise ValueError(f"Invalid positive-duration age bracket for {name!r}")
        rows.append(dict(unit=name, age_top_Ma=float(top), age_base_Ma=float(base), age_source=source))
    out = pd.DataFrame(rows);out.attrs.update(chronostratigraphic_dataset()["metadata"])
    return out
