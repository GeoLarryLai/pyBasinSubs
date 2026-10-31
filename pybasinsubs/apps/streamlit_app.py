# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Streamlit app:  pybasinsubs-ui --streamlit   (or: streamlit run pybasinsubs/apps/streamlit_app.py)"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # works without pip install
import matplotlib
matplotlib.use("Agg")
import streamlit as st
from pybasinsubs import app_core as core

st.set_page_config(page_title="pyBasinSubs", layout="wide")
st.title("pyBasinSubs - backstripping & geohistory")
with st.sidebar:
    src = st.selectbox("Data", core.SOURCES)
    well = None; up = None
    if src.startswith("North"):
        well = st.selectbox("Well", core.wells("viking_graben"))
    elif src.startswith("Barents"):
        well = st.selectbox("Well", core.wells("barents_hammerfest"))
    elif src == "My CSV":
        up = st.file_uploader("Column CSV (see example_data/template)", type="csv")
    mixing = st.radio("Lithology mixing", ["paper", "mixed"], horizontal=True)
    n = st.slider("Monte Carlo draws", 0, 4000, 1000, 250)
    wscale = st.slider("Water-depth scale", 0.5, 1.5, 1.0, 0.05)
    hf = st.slider("Heat flow for maturity (mW/m², 0 = off)", 0, 120, 0)
tab1, tab2 = st.tabs(["1-D column", "2-D section"])
with tab1:
    col = core.load_case(src, well, up.getvalue() if up else None)
    col = core.apply_overrides(col, wscale)
    out = core.run_case(col, n=n, mixing=mixing, heat_flow_mW=hf or None)
    st.pyplot(out["fig_geohistory"])
    if "fig_maturity" in out:
        st.pyplot(out["fig_maturity"])
        st.dataframe(out["maturity"])
    st.dataframe(out["summary"])
    st.download_button("Download results CSV", out["summary"].to_csv(index=False), "pybasinsubs_results.csv")

with tab2:
    st.caption("Upload a section (horizons + units CSV, see example_data/template/my_section_*.csv) and optional "
               "fault sticks; blank = the bundled Viking Graben template.")
    c1, c2, c3 = st.columns(3)
    hz = c1.file_uploader("Horizons CSV (x_m, unit, depth_m)", type="csv", key="hz")
    un = c2.file_uploader("Units CSV (unit, ages, f_<lith>, water_depth_m)", type="csv", key="un")
    fl = c3.file_uploader("Fault sticks CSV (fault_id, x, z) - optional", type="csv", key="fl")
    d1, d2, d3, d4, d5 = st.columns(5)
    method = d1.selectbox("Method", core.SECTION_METHODS, index=1, key="sec_method")
    te = d2.number_input("Te (km)", 0.0, 50.0, 1.5, 0.5)
    hd = d3.number_input("H_d (km)", 2.0, 30.0, 8.0, 1.0)
    nmc = d4.number_input("MC draws (0 = off)", 0, 500, 0, 10)
    cut = d5.checkbox("Re-place planes on block cutoffs", value=False)
    if (hz is None) != (un is None):
        st.warning("Upload both the horizons and the units CSV (or neither for the template).")
    else:
        try:
            sec, fs = core.load_section(hz.getvalue() if hz else None, un.getvalue() if un else None,
                                        fl.getvalue() if fl else None, cutoff_planes=cut)
            o = core.run_section(sec, fs, method, Te=te * 1e3, Hd=hd * 1e3, n_mc=int(nmc), seed=0)
            st.write(f"{len(sec.units)} units, {len(sec.x)} columns, {len(fs)} faults; method = {o['method']}")
            st.pyplot(o["fig"])
            st.dataframe(o["table"].round(1))
            st.download_button("Download restoration metrics CSV", o["table"].to_csv(index=False), "section_restoration.csv")
        except Exception as e:      # show input errors in the app
            st.error(f"Section error: {e}")
