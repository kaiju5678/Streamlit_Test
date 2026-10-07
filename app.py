import time
from pathlib import Path
import streamlit as st
import core

WORK = Path("workdir"); WORK.mkdir(exist_ok=True)   # CT + masks stay on local disk only
st.set_page_config(page_title="CT Body Composition", layout="wide")
ss = st.session_state
for k, v in dict(log=[], study=None, study_name=None, mask=None).items():
    ss.setdefault(k, v)


def log(msg):
    ss.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")


@st.cache_data(show_spinner="Loading volume...")
def get_volume(p): return core.load_volume(p)


@st.cache_data(show_spinner=False)
def get_mask(p): return core.load_mask(p)


# ---------------- Sidebar: workflow ----------------
with st.sidebar:
    st.header("1 · Data input")
    up = st.file_uploader("CT study (NIfTI)", type=["nii", "gz"])
    if up and ss.study_name != up.name:
        dst = WORK / Path(up.name).name
        dst.write_bytes(up.getbuffer())
        ss.update(study=str(dst), study_name=up.name, mask=None)
        log(f"Uploaded {up.name} ({up.size/1e6:.1f} MB)")

    st.header("2 · Inference")
    if st.button("▶ Run segmentation", type="primary", disabled=ss.study is None, use_container_width=True):
        out = str(WORK / (Path(ss.study).name.split(".")[0] + "_mask.nii.gz"))
        t0 = time.time()
        try:
            with st.spinner("Segmenting..."):
                core.run_segmentation(ss.study, out, log)
            ss.mask = out
            log(f"Segmentation finished in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"ERROR: {e}"); st.error(str(e))

    vol = zooms = mask = None
    if ss.study:
        vol, zooms = get_volume(ss.study)
        if ss.mask:
            mask = get_mask(ss.mask)

    st.header("3 · Viewer")
    if vol is not None:
        n = vol.shape[2]
        k = st.slider("Axial slice", 0, n - 1, n // 2)
        width = st.slider("Window width", 50, 2000, 400)
        level = st.slider("Window level", -500, 500, 50)
        opacity = st.slider("Mask opacity", 0.0, 1.0, 0.4)
    else:
        st.caption("Upload a CT to enable the viewer.")

# ---------------- Main ----------------
st.title("Body Composition Analysis")
if ss.study_name:
    st.caption(f"Study **{ss.study_name}** · {vol.shape[2]} axial slices" + (" · mask loaded" if mask is not None else ""))

t_view, t_metrics, t_3d, t_log = st.tabs(["Slice viewer", "Metrics", "3D", "Run log"])

with t_view:
    if vol is None:
        st.info("Upload a NIfTI CT file in the sidebar to begin.")
    else:
        st.image(core.render_slice(vol, k, width, level, mask, opacity), use_container_width=True,
                 caption=f"Slice {k+1}/{vol.shape[2]} · W {width} / L {level}")
        st.markdown(" ".join(f"<span style='color:rgb{c}'>■</span> {n}" for n, c in core.LABELS.values()),
                    unsafe_allow_html=True)

with t_metrics:
    if mask is None:
        st.info("Run segmentation to see metrics.")
    else:
        st.table(core.mask_metrics(vol, mask, zooms))

with t_3d:
    st.info("3D rendering is being developed by the DCU team.")

with t_log:
    st.code("\n".join(ss.log) or "(empty)")
