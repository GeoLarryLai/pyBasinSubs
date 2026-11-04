"""Click-to-place fault polylines (Streamlit tracer): state logic, coordinates, downstream faults, app smoke."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybasinsubs import faults as F
from pybasinsubs.apps import polyline as P
from pybasinsubs.fault_tracer import vg1_example

ROOT = Path(__file__).resolve().parents[1]


def test_add_undo_finish_clear():
    s = P.new_state()
    P.add_vertex(s, 1, 2); P.add_vertex(s, 1, 2); P.add_vertex(s, 3, 4)
    assert s["cur"] == [(1.0, 2.0), (3.0, 4.0)]            # repeated click ignored
    P.undo(s); assert s["cur"] == [(1.0, 2.0)]
    P.finish(s); assert s == {"done": [], "cur": []}          # a 1-vertex line is dropped
    for p in [(0, 0), (1, 1), (2, 3)]:
        P.add_vertex(s, *p)
    P.finish(s); P.add_vertex(s, 5, 5); P.add_vertex(s, 6, 7)
    assert len(s["done"]) == 1 and len(s["cur"]) == 2
    assert [r["fault_id"] for r in P.to_rows(s)] == ["drawn-1"] * 3 + ["drawn-2"] * 2
    assert len(P.to_rows(s, include_current=False)) == 3
    P.undo(s); P.undo(s); P.undo(s)                           # empty cur: reopens drawn-1 minus its last vertex
    assert s["done"] == [] and s["cur"] == [(0.0, 0.0), (1.0, 1.0)]
    P.clear(s); assert s == {"done": [], "cur": []} and P.to_rows(s) == []


def test_pixel_section_roundtrip():
    sec, tops, base = vg1_example()
    img, ext = F.render_section_image(sec.x, tops, base, px=(780, 300))
    x, z = P.px_to_xz(0, 0, img.shape, ext)
    assert (x, z) == (ext[0], ext[3])                         # top-left = xmin, zmin
    x, z = P.px_to_xz(img.shape[1], img.shape[0], img.shape, ext)
    assert np.isclose(x, ext[1]) and np.isclose(z, ext[2])
    px, py = P.xz_to_px(*P.px_to_xz(123.5, 77.25, img.shape, ext), img.shape, ext)
    assert np.isclose(px, 123.5) and np.isclose(py, 77.25)
    gx, gz = P.click_grid(ext, nx=10, nz=4)
    assert gx.size == 40 and ext[0] < gx.min() < gx.max() < ext[1] and ext[3] < gz.min() < gz.max() < ext[2]


def test_clicked_point():
    assert P.clicked_point({"click": [{"x": 1.5, "z": 2}]}) == (1.5, 2.0)
    assert P.clicked_point({"click": []}) is None and P.clicked_point({}) is None and P.clicked_point(None) is None


def test_polylines_feed_faults():
    """Clicked polylines go through the same table -> sticks -> user_sticks_to_faults path as before."""
    sec, tops, base = vg1_example()
    x = sec.x
    ref = F.fault_polylines(F.auto_faults("curved_midramp", x, tops, base), 8e3, n=12, x=x, ref_top=tops[-1])
    s = P.new_state()
    for k, v in list(ref.items())[:2]:
        for a, b in v:
            P.add_vertex(s, a, b)
        P.finish(s)
    rows = pd.DataFrame(P.to_rows(s), columns=["fault_id", "x", "z"])
    sticks = {str(k): g[["x", "z"]].to_numpy(float) for k, g in rows.groupby("fault_id", sort=False) if len(g) >= 2}
    got = F.user_sticks_to_faults(sticks, x, tops)
    want = F.user_sticks_to_faults({f"drawn-{i + 1}": np.asarray(v, float) for i, v in enumerate(list(ref.values())[:2])}, x, tops)
    assert len(got) == 2 and [(f.x0, f.dip, f.down) for f in got] == [(f.x0, f.dip, f.down) for f in want]


def test_tracer_app_smoke():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pybasinsubs" / "apps" / "tracer_streamlit.py"), default_timeout=240)
    at.run()
    assert not at.exception
    assert [b.label for b in at.button][:3] == ["Undo", "Finish", "Clear"]
    at.sidebar.radio[0].set_value("empty").run()
    at.session_state.poly = {"done": [[(20000.0, 500.0), (21000.0, 3000.0)]], "cur": [(40000.0, 800.0), (41000.0, 2500.0)]}
    at.run()
    assert not at.exception
    tab = at.dataframe[-1].value
    assert len(tab) == 2                                       # both drawn polylines became faults
    at.button(key="poly_finish").click().run()
    assert len(at.session_state.poly["done"]) == 2 and at.session_state.poly["cur"] == []
    at.button(key="poly_undo").click().run()
    assert len(at.session_state.poly["cur"]) == 1
    at.button(key="poly_clear").click().run()
    assert at.session_state.poly == {"done": [], "cur": []} and not at.exception
