# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Click-to-place polylines for the Streamlit fault tracer (no Streamlit import; plain state logic).

State is a dict ``{"done": [[(x, z), ...], ...], "cur": [(x, z), ...]}``: clicks add vertices to ``cur``;
``finish`` closes ``cur`` (kept when it has >= 2 vertices) and starts the next polyline; ``undo`` drops the
last vertex (or reopens the last finished polyline when ``cur`` is empty); ``clear`` removes everything.
"""

import numpy as np


def new_state():
    return {"done": [], "cur": []}


def add_vertex(state, x, z):
    p = (float(x), float(z))
    if not state["cur"] or state["cur"][-1] != p:   # a repeated click on the same spot adds nothing
        state["cur"].append(p)
    return state


def undo(state):
    if state["cur"]:
        state["cur"].pop()
    elif state["done"]:
        state["cur"] = list(state["done"].pop())[:-1]
    return state


def finish(state):
    if len(state["cur"]) >= 2:
        state["done"].append(list(state["cur"]))
    state["cur"] = []
    return state


def clear(state):
    state["done"], state["cur"] = [], []
    return state


def polylines(state, include_current=True):
    """Polylines with >= 2 vertices; the open one last (when ``include_current``)."""
    out = [list(p) for p in state["done"]]
    if include_current and len(state["cur"]) >= 2:
        out.append(list(state["cur"]))
    return out


def to_rows(state, prefix="drawn", include_current=True):
    """Rows ``{fault_id, x, z}`` (one fault stick per polyline), same columns as the fault table."""
    return [dict(fault_id=f"{prefix}-{n + 1}", x=x, z=z)
            for n, line in enumerate(polylines(state, include_current)) for x, z in line]


def px_to_xz(px, py, shape, extent):
    """Image pixel (column, row) -> section (x, z); ``extent = (xmin, xmax, zmax, zmin)`` as from
    ``faults.render_section_image`` (row 0 = zmin at the top)."""
    h, w = shape[:2]
    return extent[0] + px / w * (extent[1] - extent[0]), extent[3] + py / h * (extent[2] - extent[3])


def xz_to_px(x, z, shape, extent):
    h, w = shape[:2]
    return (x - extent[0]) / (extent[1] - extent[0]) * w, (z - extent[3]) / (extent[2] - extent[3]) * h


def click_grid(extent, nx=160, nz=64):
    """Cell centres covering the section; a click picks the nearest one (one cell = 1/nx of the width)."""
    xs = extent[0] + (np.arange(nx) + 0.5) / nx * (extent[1] - extent[0])
    zs = extent[3] + (np.arange(nz) + 0.5) / nz * (extent[2] - extent[3])
    gx, gz = np.meshgrid(xs, zs)
    return gx.ravel(), gz.ravel()


def clicked_point(selection, name="click"):
    """``(x, z)`` from a Streamlit Altair selection event, or None."""
    try:
        pts = selection[name]
    except (KeyError, TypeError):
        return None
    if not pts:
        return None
    p = pts[-1]
    return float(p["x"]), float(p["z"])
