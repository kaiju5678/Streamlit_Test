"""Sidebar: data input, segmentation, viewer controls, L3, patient, report endpoint, device status."""
from __future__ import annotations

import streamlit as st

import config
from core import analysis as an
from core import labels as lb
from core import llm_client, seg_client
from ui import live
from ui import state as S

ss = st.session_state


@st.cache_data(ttl=10, show_spinner=False)
def inference_health():
    return seg_client.health()


@st.cache_data(ttl=60, show_spinner=False)
def _llm_reachable(url: str) -> str | None:
    """Checked once a minute (url is the cache key): off the lab network every check waits for the 2 s timeout."""
    return llm_client.reachable()


def sec(title: str) -> None:
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> None:
    st.markdown("<dl class='kv'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>", unsafe_allow_html=True)


def render() -> None:
    v, mask = ss.vol, ss.mask
    health = inference_health()
    ready = health is not None
    busy = ss.job is not None
    with st.sidebar:
        st.markdown("<div class='brand'><b>BodyComp</b><span>CT body composition</span></div>", unsafe_allow_html=True)

        # ---- 1 · Data input
        sec("1 · Data input")
        live.file_buttons(busy=busy, has_ct=v is not None, dicom=ready and not busy)
        # The buttons above open the file pickers of these hidden uploaders (Streamlit's upload endpoint, up to 2 GB).
        with st.container(key="bc_uploads"):
            with st.container(key="bc_up_ct"):
                st.file_uploader("CT", type=["nii", "gz"], accept_multiple_files=True, key=f"up_ct_{ss.up_nonce}",
                                 on_change=S.cb_upload_ct, label_visibility="collapsed")
            with st.container(key="bc_up_mask"):
                st.file_uploader("Mask files", type=["nii", "gz"], accept_multiple_files=True, key=f"up_mask_files_{ss.up_nonce}",
                                 on_change=S.cb_upload_mask, label_visibility="collapsed")
            with st.container(key="bc_up_dir"):
                st.file_uploader("DICOM folder", accept_multiple_files="directory", key=f"up_dcm_dir_{ss.up_nonce}",
                                 on_change=S.cb_upload_dicom, args=("dir",), label_visibility="collapsed")
            with st.container(key="bc_up_zip"):
                st.file_uploader("DICOM .zip", type=["zip"], key=f"up_dcm_zip_{ss.up_nonce}",
                                 on_change=S.cb_upload_dicom, args=("zip",), label_visibility="collapsed")
        if ss.status["text"]:
            st.markdown(f"<p class='status{' err' if ss.status['error'] else ''}'>{ss.status['text']}</p>", unsafe_allow_html=True)

        # ---- 2 · Segmentation
        sec("2 · Segmentation")
        can_segment = v is not None and bool(v["bytes"]) and ready and not busy
        st.button("Working…" if busy else "Run segmentation", type="primary", icon=":material/play_arrow:",
                  on_click=S.cb_segment, disabled=not can_segment, width="stretch")
        if not seg_client.configured():
            note = "No inference server configured (INFERENCE_URL)."
        elif health is None:
            note = f"Inference server offline at {config.INFERENCE_BASE}. Start it with python inference/server.py."
        elif not health.get("repo_found", True):
            note = "Server cannot find CT-Muscle-and-Fat-Segmentation (set SEG_REPO)."
        else:
            model = "Mazurowski nnU-Net" if health.get("model") == "mazurowski" else health["model"]
            note = f"Model: {model}" + (" + TotalSegmentator" if health.get("totalsegmentator") else "")
        st.markdown(f"<p class='note'>{note}</p>", unsafe_allow_html=True)
        if mask:
            # names and roles come from the mask when it is loaded; show / hide is the checkbox row under the 2D view
            st.markdown("<div class='legend side'>" + "".join(
                f"<span><i style='background:{lb.rgb(d['color'])}'></i>{d['name']}</span>" for d in ss.labels.values()) + "</div>",
                unsafe_allow_html=True)

        # ---- 3 · Viewer controls
        sec("3 · Viewer controls")
        if v is not None:
            live.sliders(v)
        live.window_presets()
        c1, c2 = st.columns(2)
        ss.w_ww, ss.w_wl = ss.wl["w"], ss.wl["l"]
        c1.number_input("Width (HU)", min_value=1, step=10, key="w_ww", on_change=S.cb_wl_number)
        c2.number_input("Level (HU)", step=5, key="w_wl", on_change=S.cb_wl_number)
        st.segmented_control("Tool", ["Cross", "Caliper"], key="tool", required=True,
                             help="Cross: click a slice to move the crosshair. Caliper: click two points on one slice to measure.")
        st.button("Clear measurements", on_click=S.cb_clear_meas, disabled=not ss.meas and not ss.pending, width="stretch")
        st.markdown("<p class='note'>On a slice: wheel = slice, Ctrl+wheel = zoom, drag = pan, double-click = reset.</p>",
                    unsafe_allow_html=True)

        # ---- L3 landmark
        sec("L3 landmark")
        c1, c2 = st.columns(2)
        c1.button("Set L3 at current axial", on_click=S.cb_set_l3, disabled=v is None, width="stretch")
        c2.button("Go to L3", on_click=S.cb_go_l3, disabled=ss.l3 is None, width="stretch")
        st.markdown(f"<p class='status'>{f'L3 at axial slice {ss.l3 + 1}' if ss.l3 is not None else 'Not set'}</p>",
                    unsafe_allow_html=True)

        # ---- Patient
        sec("Patient")
        c1, c2 = st.columns(2)
        c1.selectbox("Sex", ["", "M", "F"], key="p_sex", format_func=lambda s: {"": "–", "M": "Male", "F": "Female"}[s])
        c2.number_input("Age", min_value=0, max_value=120, step=1, key="p_age", placeholder="–")
        c1, c2 = st.columns(2)
        c1.number_input("Height cm", min_value=50.0, max_value=250.0, step=1.0, key="p_h", placeholder="–")
        c2.number_input("Weight kg", min_value=10.0, max_value=400.0, step=0.1, key="p_w", placeholder="–")
        b = an.bmi_of(S.patient())
        st.markdown(f"<p class='status'>{f'BMI {b:.1f} kg/m²' if b else 'Sex, height and weight are needed for index cut-offs.'}</p>",
                    unsafe_allow_html=True)

        # ---- 4 · Report endpoint
        sec("4 · Report endpoint")
        on = llm_client.configured()
        why = _llm_reachable(config.LLM_URL) if on else None
        st.markdown(f"<p class='status{' err' if why else ''}'><span class='led {'bad' if why else 'on' if on else ''}'></span>"
                    f"{config.LLM_MODEL if on else 'No language model · rule-based drafts'}"
                    f"{f' · {why}' if why else ''}</p>", unsafe_allow_html=True)
        st.button("Generate AI report" if on else "Generate draft report", on_click=S.cb_generate,
                  disabled=v is None, width="stretch")

        # ---- Device status
        sec("Device status")
        device_panel()


@st.fragment(run_every=10)
def device_panel() -> None:
    d = seg_client.device()
    if not d.get("configured"):
        st.markdown("<p class='note'>No inference server configured.</p>", unsafe_allow_html=True)
        return
    if not d.get("online"):
        st.markdown(f"<p class='status err'><span class='led bad'></span>Offline{' · ' + d['error'] if d.get('error') else ''}</p>",
                    unsafe_allow_html=True)
        return
    st.markdown(f"<p class='status'><span class='led {'mid' if d.get('busy') else 'on'}'></span>{d['device']} · "
                f"{'running' if d.get('busy') else 'ready'}</p>", unsafe_allow_html=True)
    mem = (f"{d['mem_used_mb'] / 1024:.1f} / {(d.get('mem_total_mb') or 0) / 1024:.1f} GB"
           if d.get("mem_used_mb") is not None else "–")
    util = d.get("gpu_util") if d.get("gpu_util") is not None else d.get("cpu_util")
    kv([("Memory", mem), ("GPU" if d.get("gpu_util") is not None else "CPU", f"{util:.0f}%" if util is not None else "–"),
        ("Temp", f"{d['temp_c']:.0f} °C" if d.get("temp_c") is not None else "–"),
        ("Power", f"{d['power_w']:.1f} W" if d.get("power_w") is not None else "–")])
    if d.get("mem_used_mb") and d.get("mem_total_mb"):
        pct = d["mem_used_mb"] / d["mem_total_mb"] * 100
        st.markdown(f"<div class='meter'><span style='width:{pct:.0f}%'></span></div>", unsafe_allow_html=True)
