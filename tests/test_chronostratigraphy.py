# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pytest

from pybasinsubs.chronostratigraphy import age_bracket, chronostratigraphic_dataset, resolve_unit_ages
from pybasinsubs.fault_tracer import vg1_example


def test_official_dataset_metadata_and_brackets():
    data = chronostratigraphic_dataset()
    assert data["metadata"]["chart_version"] == "2026/06"
    assert data["metadata"]["chart_year"] == 2026
    assert "025001" in data["metadata"]["citation"]
    assert data["metadata"]["numeric_source"].startswith("https://stratigraphy.org/")
    assert len(data["units"]) == 153
    assert age_bracket("Lower Cretaceous") == (100.5, 143.1)
    assert age_bracket("Palaeocene") == (56., 66.)
    assert age_bracket("Late Jurassic") == (143.1, 161.5)
    assert age_bracket("Bajocian-Bathonian (Brent)") == (165.3, 170.9)
    assert age_bracket("Pliocene-Recent") == (0., 5.333)
    assert age_bracket("Aquitanian") == (20.45, 23.04)  # static 2026/06 PDF, not stale interactive values
    for row in data["units"]:assert row["age_base_Ma"] > row["age_top_Ma"] >= 0
    with pytest.raises(ValueError, match="explicit numeric ages"):age_bracket("Ryazanian")


def test_numeric_vg1_ages_always_win():
    sec, _, _ = vg1_example()
    t = sec.age_top.copy();b = sec.age_base.copy()
    rows = resolve_unit_ages(sec.units, t, b)
    np.testing.assert_array_equal(rows.age_top_Ma, t);np.testing.assert_array_equal(rows.age_base_Ma, b)
    np.testing.assert_array_equal(sec.age_top, t);np.testing.assert_array_equal(sec.age_base, b)
    assert (rows.age_source == "provided numeric ages").all()
    assert rows.loc[5, "age_base_Ma"] == 140.  # MUST NOT become ICS 143.1
    assert resolve_unit_ages(["Miocene"]).iloc[0].age_base_Ma == 23.04
