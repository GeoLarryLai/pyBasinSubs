# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Notebook display palettes must not enter the legacy VG1 image detector."""
import ast
import json
from pathlib import Path

import numpy as np

from pybasinsubs import faults as F
from pybasinsubs.fault_tracer import vg1_example
from pybasinsubs.material_colors import VG1_COLORS

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ["#f08068", "#60a0c0", "#f8f000", "#b840a8", "#00a050", "#e81820", "#8070c0", "#908888", "#60a0c0", "#f8f000"]


def test_notebook_display_detection_split():
    nb = json.loads((ROOT / "notebooks" / "pyBasinSubs_9_fault_interpretation.ipynb").read_text())
    setup = "".join(nb["cells"][1]["source"])
    assignments = {node.targets[0].id: node.value for node in ast.parse(setup).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    frozen = ast.literal_eval(assignments["DETECTION_COLORS"])
    assert frozen == LEGACY and tuple(frozen) != VG1_COLORS
    assert "VCOL = section_material_colors(VUNITS)" in setup
    raster_cell = next("".join(c["source"]) for c in nb["cells"] if "img, ext = F.render_section_image" in "".join(c["source"]))
    assert "colors=DETECTION_COLORS" in raster_cell and "colors=VCOL" not in raster_cell
    editor_cell = next("".join(c["source"]) for c in nb["cells"] if "display_img, display_ext = F.render_section_image" in "".join(c["source"]))
    assert "image=display_img" in editor_cell and "colors=VCOL" in editor_cell
    sec, tops, base = vg1_example()
    before, ext = F.render_section_image(sec.x, tops, base, colors=LEGACY, px=(1600, 400))
    after, ext_after = F.render_section_image(sec.x, tops, base, colors=frozen, px=(1600, 400))
    assert ext == ext_after
    np.testing.assert_array_equal(before, after)
    old_faults = F.detect_faults_from_image(before, ext, sec.x, tops)
    new_faults = F.detect_faults_from_image(after, ext_after, sec.x, tops)
    assert len(old_faults) == len(new_faults)
    for old, new in zip(old_faults, new_faults):
        assert (old.x0, old.dip, old.confidence) == (new.x0, new.dip, new.confidence)
        np.testing.assert_array_equal(old.stick, new.stick)
        np.testing.assert_array_equal(old.throw, new.throw)
    nb8 = json.loads((ROOT / "notebooks" / "pyBasinSubs_8_2d_flexural_backstripping.ipynb").read_text())
    assert any("VCOL = section_material_colors(VUNITS)" in "".join(c["source"]) for c in nb8["cells"])
