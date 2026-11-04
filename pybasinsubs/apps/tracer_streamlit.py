# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Streamlit fault tracer.  Launch:

    pybasinsubs-tracer --streamlit          (or: streamlit run pybasinsubs/apps/tracer_streamlit.py)

Trace faults by clicking vertices on the section image (Undo / Finish / Clear); each polyline is one fault stick.
Faults can also be added / edited / deleted in the editable table (fault_id, x, z), which always works.
"""

import pandas as pd
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pybasinsubs import flex2d, faults as F
from pybasinsubs.fault_tracer import vg1_example
from pybasinsubs.apps import polyline as P


def _trace_chart(img, ext, poly):
    """Section image + polylines + an invisible click grid (Altair ships with Streamlit)."""
    import base64, io
    import altair as alt
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    h, w = img.shape[:2]
    xs = alt.Scale(domain=[ext[0], ext[1]], nice=False, zero=False)
    zs = alt.Scale(domain=[ext[2], ext[3]], nice=False, zero=False)   # depth down
    bg = alt.Chart(pd.DataFrame({"x": [ext[0]], "x2": [ext[1]], "z": [ext[3]], "z2": [ext[2]], "url": [url]})).mark_image(
        aspect=False).encode(x=alt.X("x:Q", scale=xs, axis=None), x2="x2", y=alt.Y("z:Q", scale=zs, axis=None), y2="z2", url="url:N")
    gx, gz = P.click_grid(ext)
    sel = alt.selection_point(name="click", on="click", nearest=True, fields=["x", "z"], toggle=False, empty=False)
    grid = alt.Chart(pd.DataFrame({"x": gx, "z": gz})).mark_point(opacity=0.001, size=20).encode(
        x=alt.X("x:Q", scale=xs, axis=None), y=alt.Y("z:Q", scale=zs, axis=None)).add_params(sel)
    lines = [dict(id=f"drawn-{n + 1}", i=i, x=a, z=b) for n, ln in enumerate(poly["done"] + [poly["cur"]]) for i, (a, b) in enumerate(ln)]
    layers = [bg]
    if lines:
        d = alt.Chart(pd.DataFrame(lines)).encode(x=alt.X("x:Q", scale=xs, axis=None), y=alt.Y("z:Q", scale=zs, axis=None), detail="id:N", order="i:Q")
        layers += [d.mark_line(color="red", strokeWidth=2), d.mark_point(color="red", filled=True, size=30)]
    return alt.layer(*layers, grid).properties(width=w, height=h, padding=0).configure_view(stroke=None)   # click grid on top


def _on_click():
    ev = st.session_state.get(f"trace_{st.session_state.chart_n}")
    p = P.clicked_point(getattr(ev, "selection", None) or (ev or {}).get("selection"))
    if p is not None and p != st.session_state.last_click:
        P.add_vertex(st.session_state.poly, *p)
        st.session_state.last_click = p

st.set_page_config(page_title="pyBasinSubs fault tracer", layout="wide")
st.title("Fault tracer + restoration (pyBasinSubs)")
sec, tops, base = vg1_example()
x = sec.x
st.caption("Section: line Viking Graben 1, digitized from Roberts et al. (1998) Fig. 4a. x = distance (m), z = depth (m).")

with st.sidebar:
    src = st.radio("Start from", ["stored user faults (vg1_user_faults.json / .csv)", "curved mid-ramp (auto, default)",
                                  "curved footwall crest (auto)", "straight (auto planar picks)", "import file", "empty"])
    up = st.file_uploader("Fault sticks (CSV fault_id,x,z / GeoJSON / Petrel-Charisma / Kingdom / OpendTect ASCII)")
    method = st.radio("Restoration method", ["cantilever", "listric", "domino"], index=2)
    alpha = st.slider("listric shear angle α (deg, antithetic, inclined only)", 45, 80, 65)
    te = st.number_input("Te (m)", 0.0, 50000.0, 1500.0, 500.0)

key = f"{src}-{getattr(up, 'name', '')}"
if st.session_state.get("key") != key:
    if src.startswith("stored") and F.stored_user_faults() is not None:
        sticks, _ = F.read_user_faults()
    elif src.startswith(("stored", "curved", "straight")):
        kind = {"curved m": "curved_midramp", "curved f": "curved_crest", "straight": "straight"}.get(src[:8], "curved_midramp")
        sticks = F.fault_polylines(F.auto_faults(kind, x, tops, base), 8e3, n=12, x=x, ref_top=tops[-1])
    elif src == "import file" and up is not None:
        import tempfile, pathlib
        p = pathlib.Path(tempfile.mkdtemp()) / up.name; p.write_bytes(up.getvalue()); sticks = F.read_fault_sticks(p)
    else:
        sticks = {}
    st.session_state.table = pd.DataFrame([dict(fault_id=k, x=float(a), z=float(b)) for k, v in sticks.items() for a, b in v],
                                          columns=["fault_id", "x", "z"])
    st.session_state.key = key

img, ext = F.render_section_image(x, tops, base, px=(780, 300))
st.subheader("Trace faults: click to add vertices (one polyline = one fault stick)")
if "poly" not in st.session_state:
    st.session_state.poly, st.session_state.last_click, st.session_state.chart_n = P.new_state(), None, 0
poly = st.session_state.poly
b1, b2, b3, _ = st.columns([1, 1, 1, 5])
for col, label, fn in ((b1, "Undo", P.undo), (b2, "Finish", P.finish), (b3, "Clear", P.clear)):
    if col.button(label, key=f"poly_{label.lower()}"):
        fn(poly); st.session_state.last_click = None; st.session_state.chart_n += 1   # new chart = no stale click
st.altair_chart(_trace_chart(img, ext, poly), on_select=_on_click, selection_mode="click",
                key=f"trace_{st.session_state.chart_n}", width="content")
st.caption(f"{len(P.polylines(poly, False))} finished, current {len(poly['cur'])} vertices. "
           "Finish closes the current fault and starts the next; Undo removes the last vertex.")
drawn = P.to_rows(poly)

st.subheader("Fault sticks (add / edit / delete rows)")
table = st.data_editor(st.session_state.table, num_rows="dynamic", use_container_width=True, key="editor")
allp = pd.concat([table, pd.DataFrame(drawn, columns=["fault_id", "x", "z"])]).dropna()
sticks = {str(k): g[["x", "z"]].to_numpy(float) for k, g in allp.groupby("fault_id", sort=False) if len(g) >= 2}
faults = F.user_sticks_to_faults(sticks, x, tops)
sfmt = st.radio("Save format", ["json", "csv"], horizontal=True)
if st.button("Save to stored user faults (vg1_user_faults.<format>)"):
    st.success(f"saved {len(sticks)} faults to {F.write_user_faults(sticks, F.user_faults_path('vg1_user_faults.' + sfmt))}")

fig, ax = plt.subplots(figsize=(14, 3.6))
ax.imshow(img, extent=ext, aspect="auto")
for k, v in sticks.items():
    ax.plot(v[:, 0], v[:, 1], "r-o", ms=3); ax.text(v[0, 0], v[0, 1] - 80, k, color="r", fontsize=8)
ax.set_xlabel("x (m)"); ax.set_ylabel("depth (m)"); st.pyplot(fig)
st.download_button("Download sticks CSV", table.to_csv(index=False).encode() if not drawn else allp.to_csv(index=False).encode(),
                   "fault_sticks.csv", "text/csv")
st.write(pd.DataFrame([dict(fault=f.label, x0_km=round(f.x0 / 1e3, 2), dip=round(f.dip, 1), down=f.down) for f in faults]))

if st.button("Restore with " + method, type="primary"):
    kw = {"alpha": float(alpha)} if method == "listric" else {}
    r = flex2d.restore(sec, faults, method=method, Te=te, beta=1.15, **kw)
    nu = len(r["ages"])
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
    for a_, i in zip(axs, [nu - 3, nu - 2, nu - 1]):
        for k in range(i, nu):
            a_.fill_between(x / 1e3, r["top"][i][k], r["base"][i][k], color=f"C{k % 10}", lw=0)
        for f in faults:
            a_.axvline(f.x0 / 1e3, color="k", lw=.4, ls="--")
        a_.set_title(f"{method}: {r['ages'][i]:g} Ma"); a_.set_ylim(6000, -500)
    st.pyplot(fig)
    m = flex2d.restoration_metrics(r, nu - 1, x, faults)
    st.success(f"Oldest top at {r['ages'][-1]:g} Ma: std {m['std']:.0f} m, P95-P5 relief {m['relief_p95_p5']:.0f} m")
