"""BodyComp: CT body composition workstation in Python/Streamlit (the ct-scan-test workflow).

    terminal 1: cd server_segment && python server.py   segmentation server (Mazurowski nnU-Net + TotalSegmentator vertebrae)
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

st.set_page_config(page_title="BodyComp · Body composition analysis", page_icon="🩻", layout="wide",
                   initial_sidebar_state="expanded")
S.init()
ss = st.session_state

# dark.css only redefines the colour variables and restyles Streamlit's own widgets (the sidebar theme button)
css = ["style.css", "dark.css"] if ss.dark else ["style.css"]
st.markdown("<style>" + "".join((config.BASE_DIR / "assets" / f).read_text(encoding="utf-8") for f in css) + "</style>",
            unsafe_allow_html=True)

# Long actions queued by buttons (DICOM conversion, report generation) run here, before anything is drawn.
if ss.action:
    action, ss.action = ss.action, None
    S.run_action(action)


def took_text(secs) -> str:
    return "–" if secs is None else f"{int(secs) // 60} min {int(secs) % 60:02d} s" if secs >= 60 else f"{int(secs)} s"


def steps(a: dict) -> tuple[list[tuple[str, str]], str]:
    """The five workflow steps as (label, done|now|todo) and a plain sentence on what to do next."""
    v, mask, rep, sig = a["vol"], a["mask"], a["report"], a["signature"]
    stale = bool(rep) and rep["payloadKey"] != a["payload_key"]
    done = [v is not None, mask is not None, a["l3"] is not None, bool(rep) and not stale, bool(sig)]
    labels = ["Open CT", "Segment", "Check L3", "Report", "Sign off"]
    now = next((i for i, d in enumerate(done) if not d), None)
    hint = {
        0: "Open a CT study in the sidebar (NIfTI file or DICOM).",
        1: ("Segmentation is running. You can keep scrolling through the slices meanwhile." if ss.job else
            "Run segmentation in the sidebar, or open a mask you already have."),
        2: "Scroll to the L3 vertebra and press <b>Set L3 here</b> in the sidebar.",
        3: ("The measurements changed after the report was written. Generate it again." if stale else
            "Generate the report. Only the eight numbers on the <b>Transmitted JSON</b> tab are sent."),
        4: ("This is the sample study, so it cannot be signed. Open a real CT to sign a report." if v and v["sample"] else
            "Review the draft on the <b>AI report</b> tab, then sign it."),
        None: "Report signed. Export it as PDF or CSV from the Overview.",
    }[now]
    return [(lab, "done" if d else "now" if i == now else "todo") for i, (lab, d) in enumerate(zip(labels, done))], hint


def header(a: dict) -> None:
    v, mask, rep, sig = a["vol"], a["mask"], a["report"], a["signature"]
    if v:
        nx, ny, nz = v["shape"]
        sx, sy, sz = v["spacing"]
        sub = (f"Study <b title='{tabs.h(v['name'])}'>{tabs.h(v['name'])}</b><span>{nz} axial slices</span>"
               f"<span class='mono'>{nx}×{ny} · {sx:.2f}×{sy:.2f}×{sz:.2f} mm</span>")
        if ss.seg_secs is not None and mask:
            sub += f"<span>segmented in <b>{took_text(ss.seg_secs)}</b></span>"
    else:
        sub = "<span>No study open</span>"
    if ss.job:
        seg = ("mid", f"Segmenting · {took_text(time.time() - ss.job['t0'])}")
    elif mask:
        seg = ("ok", "Segmentation OK" if mask["source"] == "inference" else "Mask loaded")
    else:
        seg = ("", "Not segmented")
    if rep:
        stale = rep["payloadKey"] != a["payload_key"]
        repc = ("mid", "Report out of date") if stale else ("ok", "Report · " + ("AI" if rep["source"] == "llm" else "rule-based"))
    else:
        repc = ("", "No report")
    signc = ("ok", "Signed") if sig else ("mid", "Not signed") if rep else ("", "Not signed")
    chips = "".join(f"<span class='chip {c}'>{t}</span>" for c, t in (seg, repc, signc))
    st.markdown(f"<header class='top'><div class='ttl'><h1>Body Composition Analysis</h1><p>{sub}</p></div>"
                f"<div class='chips-top'>{chips}</div></header>", unsafe_allow_html=True)
    rows, hint = steps(a)
    # Connectors are their own items between the steps, so the highlighted step never runs into a line
    items = []
    for n, (lab, state) in enumerate(rows, 1):
        if n > 1:
            items.append(f"<li class='bar{' done' if rows[n - 2][1] == 'done' else ''}' aria-hidden='true'></li>")
        items.append(f"<li class='{state}'{' aria-current=step' if state == 'now' else ''}>"
                     f"<i>{'✓' if state == 'done' else n}</i>{lab}</li>")
    st.markdown(f"<nav class='steps' aria-label='Workflow'><ol>{''.join(items)}</ol>"
                f"<p class='next'><b>Next</b>{hint}</p></nav>", unsafe_allow_html=True)


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
