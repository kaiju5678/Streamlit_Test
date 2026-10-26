"""Sidebar: data input, segmentation, viewer controls, L3, patient, report endpoint, device status."""
from __future__ import annotations

import html

import streamlit as st

import config
from core import analysis as an
from core import llm_client, seg_client
from ui import live
from ui import state as S

ss = st.session_state


@st.cache_data(ttl=10, show_spinner=False)
def inference_health():
    return seg_client.health()


@st.cache_data(ttl=8, show_spinner=False)
def device_status() -> dict:
    """Cached so a rerun (a click, a slider) does not wait on the server; the panel below refreshes every 10 s anyway."""
    return seg_client.device()


@st.cache_data(ttl=60, show_spinner=False)
def _llm_reachable(url: str) -> str | None:
    """Checked once a minute (url is the cache key): off the lab network every check waits for the 2 s timeout."""
    return llm_client.reachable()


def sec(title: str, n: int | None = None, done: bool = False) -> None:
    """Section heading; workflow sections carry their step number, or a tick once done."""
    mark = f"<i class='{'done' if done else ''}'>{'✓' if done else n}</i>" if n else ""
    st.markdown(f"<div class='sec'>{mark}{title}</div>", unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> None:
    st.markdown("<dl class='kv'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>", unsafe_allow_html=True)


def render() -> None:
    v = ss.vol
    health = inference_health()
    ready = health is not None
    busy = ss.job is not None
    with st.sidebar:
        with st.container(horizontal=True, vertical_alignment="center", key="brand_row"):
            st.markdown("<div class='brand'><span class='logo'><svg width='16' height='16' viewBox='0 0 24 24' fill='none' "
                        "stroke='currentColor' stroke-width='2' aria-hidden='true'><ellipse cx='12' cy='12' rx='9' ry='7'/>"
                        "<circle cx='12' cy='15' r='2.5'/></svg></span><div><b>CT Body Composition</b>"
                        "<span>BodyComp analysis console</span></div></div>", unsafe_allow_html=True)
            st.button("Light" if ss.dark else "Dark", key="theme_btn", on_click=S.cb_theme,
                      icon=":material/light_mode:" if ss.dark else ":material/dark_mode:",
                      help="Switch to the light theme" if ss.dark else "Switch to the dark theme")
        live.theme_memory()

        # ---- 1 · Data input
        sec("Data input", 1, v is not None)
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
        sec("Segmentation", 2, ss.mask is not None)
        can_segment = v is not None and bool(v["bytes"]) and ready and not busy
        st.button("Segmenting…" if busy else "Run segmentation again" if ss.mask is not None else "Run segmentation",
                  type="secondary" if ss.mask is not None else "primary", icon=":material/play_arrow:",
                  on_click=S.cb_segment, disabled=not can_segment, width="stretch")
        if not seg_client.configured():
            note = "Segmentation server not set (INFERENCE_URL in .env)."
        elif health is None:
            note = (f"Segmentation server is not running ({config.INFERENCE_BASE}). Start it with "
                    "<span class='mono'>python server_segment/server.py</span>, or open a mask you already have.")
        elif not health.get("repo_found", True):
            note = "Server cannot find CT-Muscle-and-Fat-Segmentation (set SEG_REPO)."
        elif health.get("error"):
            note = f"Server could not load the model: {html.escape(health['error'][-160:])}"
        elif health.get("ready") is False:
            note = "Server is loading the model; a run started now waits for it."
        else:
            model = "Mazurowski nnU-Net" if health.get("model") == "mazurowski" else health["model"]
            note = f"Model: {model}" + (" + TotalSegmentator" if health.get("totalsegmentator") else "")
        st.markdown(f"<p class='note{' warn' if ready is False and seg_client.configured() else ' model'}'>{note}</p>",
                    unsafe_allow_html=True)

        # ---- 3 · Viewer controls
        sec("Viewer")
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
        sec("L3 level", 3, ss.l3 is not None)
        c1, c2 = st.columns(2)
        c1.button("Set L3 here", help="Mark the current axial slice as the L3 level", on_click=S.cb_set_l3, disabled=v is None, width="stretch")
        c2.button("Go to L3", on_click=S.cb_go_l3, disabled=ss.l3 is None, width="stretch")
        st.markdown(f"<p class='status'>{f'L3 at axial slice {ss.l3 + 1}' if ss.l3 is not None else 'Not set'}</p>",
                    unsafe_allow_html=True)

        # ---- Patient
        sec("Patient <small>optional</small>")
        c1, c2 = st.columns(2)
        c1.selectbox("Sex", ["", "M", "F"], key="p_sex", format_func=lambda s: {"": "–", "M": "Male", "F": "Female"}[s])
        c2.number_input("Age", min_value=0, max_value=120, step=1, key="p_age", placeholder="–")
        c1, c2 = st.columns(2)
        c1.number_input("Height cm", min_value=50.0, max_value=250.0, step=1.0, key="p_h", placeholder="–")
        c2.number_input("Weight kg", min_value=10.0, max_value=400.0, step=0.1, key="p_w", placeholder="–")
        b = an.bmi_of(S.patient())
        st.markdown(f"<p class='note'>{f'BMI {b:.1f} kg/m². ' if b else 'Sex, height and weight enable the L3 cut-offs. '}"
                    "Kept on this computer, never sent.</p>", unsafe_allow_html=True)

        # ---- 4 · Report endpoint
        sec("Report", 4, ss.report is not None)
        on = llm_client.configured()
        why = _llm_reachable(config.LLM_URL) if on else None
        # Off the lab network is normal (MedGemma is only reachable there): amber, not an error
        st.markdown(f"<p class='status'><span class='led {'mid' if why else 'on' if on else ''}'></span>"
                    f"{html.escape(config.LLM_MODEL) if on else 'No language model · rule-based drafts'}"
                    f"{' · not reachable' if why else ' · online' if on else ''}</p>"
                    + (f"<p class='note warn' title='{html.escape(why)}'>Only reachable from the lab network. "
                       "Reports will use the rule-based draft for now.</p>" if why else ""), unsafe_allow_html=True)
        st.button("Create AI report" if on else "Create draft report", on_click=S.cb_generate, icon=":material/description:",
                  type="primary" if ss.mask is not None and ss.report is None else "secondary",
                  disabled=v is None, width="stretch")


        # ---- Device status
        sec("Device status")
        device_panel()
        st.markdown("<div class='privacy'><b>Privacy mode</b>CT images and masks never leave this computer. Only eight "
                    "de-identified numbers are sent for the report, checked against a whitelist first.</div>",
                    unsafe_allow_html=True)


@st.fragment(run_every=10)
def device_panel() -> None:
    d = device_status()
    if not d.get("configured"):
        st.markdown("<p class='note'>Segmentation server not set.</p>", unsafe_allow_html=True)
        return
    if not d.get("online"):
        st.markdown(f"<p class='status' title='{html.escape(d.get('error') or '')}'><span class='led mid'></span>"
                    "Segmentation server not connected</p>", unsafe_allow_html=True)
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
