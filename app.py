"""BodyComp: CT body composition workstation in Python/Streamlit (the ct-scan-test workflow).

    terminal 1: cd inference && python server.py   segmentation server (Mazurowski nnU-Net + TotalSegmentator vertebrae)
    terminal 2: streamlit run app.py               ->  http://localhost:8501

Studies, masks and reports live in this browser session; only signed reports are written (DATA_DIR/reports).
Only the de-identified metrics payload (Transmitted JSON tab) is sent to the language model.
"""
from __future__ import annotations

import time

import streamlit as st

import config
from ui import sidebar, tabs
from ui import state as S

st.set_page_config(page_title="BodyComp · CT body composition", page_icon="🩻", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(config.BASE_DIR / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

S.init()
ss = st.session_state

# Long actions queued by buttons (DICOM conversion, report generation) run here, before anything is drawn.
if ss.action:
    action, ss.action = ss.action, None
    S.run_action(action)


def header(a: dict) -> None:
    v, mask, rep, sig = a["vol"], a["mask"], a["report"], a["signature"]
    if v:
        nx, ny, nz = v["shape"]
        sx, sy, sz = v["spacing"]
        case = f"<strong title='{tabs.h(v['name'])}'>{tabs.h(v['name'])}</strong><span>{nx}×{ny}×{nz} · {sx:.2f}×{sy:.2f}×{sz:.2f} mm</span>"
    else:
        case = "<span>No study open</span>"
    seg = ({"sample": "Sample mask"}.get(mask["source"], "Mask loaded") if mask
           else "Running…" if ss.job else "No mask")
    repst = ("Signed" if sig else ("Draft · out of date" if rep["payloadKey"] != a["payload_key"] else "Draft") if rep
             else "Not started")
    secs = time.time() - ss.job["t0"] if ss.job else ss.seg_secs
    took = "–" if secs is None else f"{int(secs) // 60} min {int(secs) % 60:02d} s" if secs >= 60 else f"{int(secs)} s"
    st.markdown(
        "<header class='top'><div class='brand'><b>BodyComp</b><span>CT body composition</span></div>"
        f"<div class='case'>{case}</div><div class='chips-top'>"
        f"<span class='chip-s {'on' if mask else 'mid' if ss.job else ''}'><em>Segmentation</em>{seg}</span>"
        f"<span class='chip-s {'mid' if ss.job else 'on' if ss.seg_secs is not None else ''}'><em>Seg time</em>{took}</span>"
        f"<span class='chip-s {'on' if sig else 'mid' if rep else ''}'><em>Report</em>{repst}</span></div></header>",
        unsafe_allow_html=True)


@st.fragment(run_every=1)
def live_header(a: dict) -> None:
    """Header while a segmentation runs: the Seg time chip ticks each second and the result is picked up when the
    thread ends; the rest of the page stays usable meanwhile."""
    job = ss.job
    if job is not None and not job["thread"].is_alive():
        S.finish_job()
        st.rerun()
    header(a)


sidebar.render()
a = S.analysis()
if ss.job:
    live_header(a)
else:
    header(a)

if a["vol"] is None:
    st.info("Open a CT volume to begin.")
    st.stop()

# Every tab is drawn on each run and the browser switches between them, so changing tab needs no rerun and the 3D model
# and charts stay mounted. A callback opens a tab by setting tab_default and bumping tab_nonce (new key = fresh widget).
panes = st.tabs(S.TABS, key=f"tabs_{ss.tab_nonce}", default=ss.tab_default)
for pane, render in zip(panes, (tabs.overview, tabs.viewer, tabs.metrics, tabs.report, tabs.transmitted, tabs.runlog)):
    with pane:
        render(a)

st.markdown("<div class='footer'>Research and decision-support use only. Not a cleared medical device. A qualified reader must "
            "verify the segmentation and the L3 level, and review any generated text, before clinical use. Model: Mazurowski Lab "
            "muscle/fat nnU-Net (CC BY-NC 4.0, arXiv:2502.09779).</div>", unsafe_allow_html=True)
