# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Streamlit fault tracer.  Launch:

    pybasinsubs-tracer --streamlit          (or: streamlit run pybasinsubs/apps/tracer_streamlit.py)

Draw fault segments with the mouse over the section image (streamlit-drawable-canvas).
Faults can also be added / edited / deleted in the editable table (fault_id, x, z), which always works.
"""

import pandas as pd
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pybasinsubs import flex2d, faults as F
from pybasinsubs.fault_tracer import vg1_example

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
try:
    from streamlit_drawable_canvas import st_canvas
    from PIL import Image
    st.subheader("Draw faults (each straight line = one fault stick; use the table to add more vertices)")
    cv = st_canvas(background_image=Image.fromarray(img), drawing_mode="line", stroke_width=3, stroke_color="#ff0000",
                   width=img.shape[1], height=img.shape[0], key="canvas")
    drawn = []
    if cv.json_data:
        for n, o in enumerate(o for o in cv.json_data["objects"] if str(o.get("type", "")).lower() == "line"):
            cx, cy = o["left"], o["top"]
            for px_, py_ in [(cx + o["x1"], cy + o["y1"]), (cx + o["x2"], cy + o["y2"])]:
                drawn.append(dict(fault_id=f"drawn-{n + 1}", x=ext[0] + px_ / img.shape[1] * (ext[1] - ext[0]),
                                  z=ext[3] + py_ / img.shape[0] * (ext[2] - ext[3])))
except Exception as e:  # optional dependency missing or incompatible
    drawn = []
    st.info(f"Mouse drawing unavailable ({type(e).__name__}); edit faults in the table below.")

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
