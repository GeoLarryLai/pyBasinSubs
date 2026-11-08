# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Shared material colors for the ten-unit digitized VG1 section (not fault-line colors)."""

VG1_MATERIAL_COLORS = {
    "Pliocene-Recent": "#1f77b4",
    "Miocene": "#aec7e8",
    "Eocene-Oligocene": "#ff7f0e",
    "Palaeocene": "#ffbb78",
    "Upper Cretaceous": "#2ca02c",
    "Lower Cretaceous": "#98df8a",
    "Kimmeridgian-Ryazanian": "#d62728",
    "Callovian-Oxfordian": "#ff9896",
    "Bajocian-Bathonian (Brent)": "#9467bd",
    "Triassic-Lower Jurassic": "#c5b0d5",
}
VG1_COLORS = tuple(VG1_MATERIAL_COLORS.values())



def material_color(unit, fallback):
    """Resolve a VG1 name (including notebook formation qualifiers), else use fallback."""
    aliases = {"Kimmeridgian-Ryazanian (Draupne)": "Kimmeridgian-Ryazanian",
               "Callovian-Oxfordian (Heather)": "Callovian-Oxfordian"}
    name = str(unit)
    return VG1_MATERIAL_COLORS.get(aliases.get(name, name), fallback)


def section_material_colors(units):
    """VG1 colors by unit name; other sections retain the section app's tab20 defaults.

    Explicit caller palettes remain the responsibility of the caller.
    """
    from matplotlib import colormaps

    return [material_color(name, colormaps["tab20"](i % 20)) for i, name in enumerate(units)]
