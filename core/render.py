"""Slice rendering and Plotly figures.

Volumes are RAS-canonical (x → right, y → anterior, z → superior). Displayed radiologically:
axial anterior up / patient right on screen left; coronal superior up; sagittal superior up, anterior left.
"""
from __future__ import annotations

import base64
import io
import math
import re

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from core.labels import ROLE_COLOR, rgb

INK = "#dbe4f0"
MUTED = "#8a9ab0"
GRID = "rgba(138,154,176,0.18)"
FONT = dict(color=INK, family="IBM Plex Sans, Segoe UI, sans-serif", size=12)


def window(hu: np.ndarray, w: float, l: float) -> np.ndarray:
    lo = l - w / 2
    return (np.clip((hu.astype(np.float32) - lo) / max(w, 1), 0, 1) * 255).astype(np.uint8)


def view_slice(arr: np.ndarray, view: str, cur: dict) -> np.ndarray:
    """2D plane of a RAS volume in display orientation (rows top→bottom, cols left→right)."""
    if view == "axial":
        s = arr[:, :, cur["k"]]          # (x, y)
    elif view == "coronal":
        s = arr[:, cur["j"], :]          # (x, z)
    else:
        s = arr[cur["i"], :, :]          # (y, z)
    return s.T[::-1, ::-1]


def view_geom(shape, spacing, view: str) -> dict:
    """Display size and the scale from voxel grid to the (isotropic) display grid."""
    nx, ny, nz = shape
    sx, sy, sz = spacing
    cols, rows, csp, rsp = {"axial": (nx, ny, sx, sy), "coronal": (nx, nz, sx, sz), "sagittal": (ny, nz, sy, sz)}[view]
    px = min(csp, rsp)
    fx, fy = csp / px, rsp / px
    return {"cols": cols, "rows": rows, "fx": fx, "fy": fy, "w": max(1, round(cols * fx)), "h": max(1, round(rows * fy)),
            "csp": csp, "rsp": rsp}


def compose(hu2d: np.ndarray, mask2d: np.ndarray | None, labels: dict, wl: dict, opacity: float) -> np.ndarray:
    g = window(hu2d, wl["w"], wl["l"])
    img = np.repeat(g[..., None], 3, axis=2).astype(np.float32)
    if mask2d is not None and opacity > 0:
        for L, d in labels.items():
            if not d["visible"]:
                continue
            m = mask2d == L
            if m.any():
                img[m] = img[m] * (1 - opacity) + np.array(d["color"], np.float32) * opacity
    return img.astype(np.uint8)










def png_uri(img: np.ndarray, size: tuple[int, int] | None = None) -> str:
    im = Image.fromarray(img)
    if size and size != im.size:
        im = im.resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def measure_mm(view: str, spacing, a, b) -> float:
    csp, rsp = {"axial": (spacing[0], spacing[1]), "coronal": (spacing[0], spacing[2]), "sagittal": (spacing[1], spacing[2])}[view]
    return math.hypot((b[0] - a[0]) * csp, (b[1] - a[1]) * rsp)


# ---------------------------------------------------------------- charts
def _base(fig: go.Figure, height: int) -> go.Figure:
    # Explicit hover box: with a transparent paper Plotly falls back to a light box that the light text can't be read on
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=FONT,
                      hoverlabel=dict(bgcolor="rgba(17,23,31,0.94)", bordercolor="#2a3646", font=dict(color=INK, size=12)))
    return fig


def donut(parts: list[dict], centre: str, sub: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=[p["name"] for p in parts], values=[p["value"] for p in parts], hole=0.66, sort=False,
                           direction="clockwise", marker=dict(colors=[p["color"] for p in parts], line=dict(color="#111a2b", width=2)),
                           textinfo="none", hovertemplate="%{label}: %{value:,.0f} cm³ (%{percent})<extra></extra>"))
    fig.update_layout(showlegend=False, annotations=[dict(text=f"<b>{centre}</b><br><span style='font-size:11px;color:{MUTED}'>{sub}</span>",
                                                          showarrow=False, font=dict(size=18, color=INK))])
    return _base(fig, 210)


def line_chart(series: list[dict], unit: str, x_label: str, markers: list[dict]) -> go.Figure:
    fig = go.Figure()
    for s in series:
        n = len(s["values"])
        # WebGL: hover stays smooth with hundreds of slices × several series (SVG redraws every point)
        fig.add_trace(go.Scattergl(x=np.arange(n), y=s["values"], name=s["name"], mode="lines+markers",
                                   line=dict(color=s["color"], width=2), marker=dict(size=5, opacity=0.01),
                                   customdata=np.arange(n), hovertemplate=f"{s['name']} %{{y:.1f}} {unit}<extra></extra>"))
    for m in markers:
        fig.add_vline(x=m["x"], line=dict(color=m.get("color", MUTED), dash="dot", width=1),
                      annotation=dict(text=m["label"], font=dict(color=m.get("color", MUTED), size=11)),
                      annotation_position=m.get("pos", "top"))
    # No zoom or pan: fixedrange is what locks it (st.plotly_chart turns a falsy dragmode into "pan"); a click on a point
    # still selects it and moves the "current" slice
    # Legend sits above the slice-marker labels (L3, current) so the two rows never overlap
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0), hovermode="x unified", clickmode="event+select",
                      xaxis=dict(title=x_label, gridcolor=GRID, zeroline=False, hoverformat="d", showspikes=True, fixedrange=True,
                                 spikemode="across", spikesnap="cursor", spikecolor=MUTED, spikethickness=1, spikedash="dot"),
                      yaxis=dict(title=unit, gridcolor=GRID, zeroline=False, fixedrange=True), dragmode=False)
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)
    # Margins after _base (it sets 8 px all round): room for the y tick labels, the x ticks + title and the legend row
    return _base(fig, 360).update_layout(margin=dict(l=64, r=16, t=56, b=52))


def area_chart(t: list, v: list, unit: str, color: str, vmax: float | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=t, y=v, mode="lines", fill="tozeroy", line=dict(color=color, width=1.5),
                               connectgaps=False, hovertemplate=f"%{{x:.1f}} s · %{{y:.1f}} {unit}<extra></extra>"))
    vals = [x for x in v if x is not None]
    if vals:
        i = int(np.nanargmax([x if x is not None else -np.inf for x in v]))
        fig.add_trace(go.Scatter(x=[t[i]], y=[v[i]], mode="markers", marker=dict(color=color, size=7), hoverinfo="skip"))
    fig.update_layout(showlegend=False, xaxis=dict(title="s", gridcolor=GRID, zeroline=False),
                      yaxis=dict(gridcolor=GRID, zeroline=False, range=[0, vmax] if vmax else None))
    return _base(fig, 180)




MESH_STYLE = {  # display name, colour, opacity; drawn outside-in so the translucent shells sit over the solid parts
    "skin": ("Body surface (CT)", "rgb(222,184,160)", 0.15),  # CT only, before a mask exists
    "sat": ("Subcutaneous fat", "rgb(240,198,96)", 0.13),
    # muscle and VAT see-through, so the skeleton shows inside (as in the three.js Mesh3D reference)
    "muscle": ("Skeletal muscle", "rgb(214,86,82)", 0.5),
    "vat": ("Visceral fat", "rgb(72,196,178)", 0.4),
    "imat": ("Intermuscular fat", "rgb(176,136,255)", 0.7),
    "other": ("Other", "rgb(236,150,64)", 0.5),
    "bone": ("Bone", "rgb(226,230,240)", 1.0),
    "skeleton": ("Bone (CT)", "rgb(226,230,240)", 1.0),
    "L3 vertebra": ("L3 vertebra", "rgb(126,224,138)", 1.0),
}
LANDMARK = re.compile(r"\b(T12|L3|L4)\b")


def _surface(b: np.ndarray, sp, step: int, sigma) -> tuple[np.ndarray, np.ndarray] | None:
    """Smooth surface of a binary volume: blurring the mask first removes the voxel staircase of marching cubes."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    f = gaussian_filter(np.pad(b, 2).astype(np.float32), sigma) if sigma else np.pad(b, 2).astype(np.float32)
    try:
        verts, faces, _, _ = marching_cubes(f, level=0.5, spacing=sp, step_size=step)
    except (ValueError, RuntimeError):
        return None
    return verts - 2 * np.array(sp), faces  # undo the padding offset


def _block_max(hu: np.ndarray, f: int) -> np.ndarray:
    """Highest HU of each f×f block in x/y: thin cortex and ribs survive the downsampling (a strided pick misses them)."""
    if f == 1:
        return hu
    nx, ny, nz = hu.shape
    dx, dy = -(-nx // f), -(-ny // f)
    p = np.pad(hu, ((0, dx * f - nx), (0, dy * f - ny), (0, 0)), constant_values=-1024)
    return p.reshape(dx, f, dy, f, nz).max(axis=(1, 3))


def _body(mask: np.ndarray) -> np.ndarray:
    """Body outline per slice: inside when between the outermost mask voxels of both its row and its column.
    Robust to a broken fat ring (no hole filling needed) and leaves out the curved table under and beside the patient."""
    m = mask > 0
    inside = lambda ax: (np.maximum.accumulate(m, axis=ax)  # noqa: E731
                         & np.flip(np.maximum.accumulate(np.flip(m, ax), axis=ax), ax))
    return inside(0) & inside(1)


def _skeleton(hu: np.ndarray, mask: np.ndarray, sp, bone_hu: float, exclude: np.ndarray) -> np.ndarray:
    """Bone from the CT itself (HU threshold) inside the body outline, in connected pieces big enough to be skeleton."""
    from scipy import ndimage as ndi

    b = (hu >= bone_hu) & _body(mask) & ~exclude
    lab, n = ndi.label(b)
    if n == 0:
        return b
    # drop specks (calcifications, contrast-filled vessels cut by the threshold): keep pieces of at least 0.3 cm³
    size = np.bincount(lab.ravel())
    keep = size >= 300 / (sp[0] * sp[1] * sp[2])
    keep[0] = False
    return keep[lab]


def _ct_body(hu: np.ndarray) -> np.ndarray:
    """Body from the CT alone: tissue above -500 HU, largest connected piece (drops the table and cushions), with the
    air inside it (lungs, bowel gas) filled slice by slice so only the skin surface is drawn."""
    from scipy import ndimage as ndi

    b = hu > -500
    lab, n = ndi.label(b)
    if n == 0:
        return b
    size = np.bincount(lab.ravel())
    size[0] = 0
    b = lab == size.argmax()
    for k in range(b.shape[2]):
        b[:, :, k] = ndi.binary_fill_holes(b[:, :, k])
    return b


def build_meshes(mask: np.ndarray | None, labels: dict, spacing, max_voxels: int, hu: np.ndarray | None = None,
                 bone_hu: float = 250) -> list[dict]:
    """Surface of each tissue role from the mask, plus the skeleton from the CT (marching cubes on a downsampled grid).
    Every group is built whatever its 2D visibility: the 3D view has its own on/off checkboxes.
    Without a mask (CT just loaded, not segmented yet) the model is the body surface and the skeleton from the CT."""
    if mask is None and hu is None:
        return []
    nx, ny, nz = (mask if mask is not None else hu).shape
    f = max(1, math.ceil(math.sqrt(nx * ny * nz / max_voxels)))
    sp = (spacing[0] * f, spacing[1] * f, spacing[2])
    if mask is None:
        small = _ct_body(hu[::f, ::f, :])
        labels = {}
    else:
        small = mask[::f, ::f, :]
    groups: dict[str, list[int]] = {"skin": []} if mask is None else {}
    for L, d in labels.items():
        key = "L3 vertebra" if "L3" in d["name"] and d["role"] == "bone" else d["role"]
        groups.setdefault(key, []).append(L)
    # Most body-composition masks have no bone label: then take it from the CT. Vertebra landmarks (T12, L3, L4) are
    # not a skeleton, so they do not stop the CT bone.
    if hu is not None and not any(d["role"] == "bone" and not LANDMARK.search(d["name"]) for d in labels.values()):
        groups["skeleton"] = []
    # sigma in voxels, softer across the (usually thicker) slices
    sig = (1.0, 1.0, max(0.5, min(1.0, sp[0] / sp[2])))
    out = []
    for key in sorted(groups, key=lambda k: list(MESH_STYLE).index(k) if k in MESH_STYLE else 99):
        if key == "skeleton":
            b = _skeleton(_block_max(hu, f), small, sp, bone_hu, np.isin(small, groups.get("L3 vertebra", [])))
        elif key == "skin":
            b = small
        else:
            b = np.isin(small, groups[key])
        if b.sum() < 20:
            continue
        fine = key in ("imat", "L3 vertebra", "skeleton", "bone")
        # IMAT is scattered specks and ribs are thin: a strong blur would melt them away
        surf = _surface(b, sp, 1 if fine else 2, 0 if key == "imat" else tuple(x * 0.6 for x in sig) if fine else sig)
        if surf is None:
            continue
        verts, faces = surf
        # the camera looks from anterior, so the patient's right (+x) is on the viewer's left, as on the slices
        name, color, opacity = MESH_STYLE.get(key, (key, rgb(ROLE_COLOR.get(key, (200, 200, 200))), 0.6))
        if key == "skeleton":
            name = f"Bone (CT ≥ {bone_hu:g} HU)"
        out.append({"name": name, "x": verts[:, 0], "y": verts[:, 1], "z": verts[:, 2],
                    "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2], "color": color, "opacity": opacity})
    return out
