"""Slice rendering and Plotly figures.

Volumes are RAS-canonical (x → right, y → anterior, z → superior). Displayed radiologically:
axial anterior up / patient right on screen left; coronal superior up; sagittal superior up, anterior left.
"""
from __future__ import annotations

import base64
import io
import math

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
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=FONT)
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
        fig.add_trace(go.Scatter(x=np.arange(n), y=s["values"], name=s["name"], mode="lines+markers",
                                 line=dict(color=s["color"], width=2), marker=dict(size=5, opacity=0.01),
                                 customdata=np.arange(n),
                                 hovertemplate=f"slice %{{x}}<br>{s['name']} %{{y:.1f}} {unit}<extra></extra>"))
    for m in markers:
        fig.add_vline(x=m["x"], line=dict(color=m.get("color", MUTED), dash="dot", width=1),
                      annotation=dict(text=m["label"], font=dict(color=m.get("color", MUTED), size=11)), annotation_position="top")
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified", clickmode="event+select",
                      xaxis=dict(title=x_label, gridcolor=GRID, zeroline=False),
                      yaxis=dict(title=unit, gridcolor=GRID, zeroline=False), dragmode="zoom")
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8))
    return _base(fig, 320)


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




def build_meshes(mask: np.ndarray, labels: dict, spacing, max_voxels: int) -> list[dict]:
    """Surface of each tissue role from the mask (marching cubes on a downsampled grid)."""
    from skimage.measure import marching_cubes

    nx, ny, nz = mask.shape
    f = max(1, math.ceil(math.sqrt(nx * ny * nz / max_voxels)))
    small = mask[::f, ::f, :]
    sp = (spacing[0] * f, spacing[1] * f, spacing[2])
    groups: dict[str, list[int]] = {}
    for L, d in labels.items():
        if not d["visible"]:
            continue
        key = "L3 vertebra" if "L3" in d["name"] and d["role"] == "bone" else d["role"]
        groups.setdefault(key, []).append(L)
    style = {"sat": 0.16, "vat": 0.55, "muscle": 0.9, "imat": 0.95, "bone": 0.9, "other": 0.5, "L3 vertebra": 1.0}
    names = {"sat": "Subcutaneous fat", "vat": "Visceral fat", "muscle": "Skeletal muscle", "imat": "Intermuscular fat",
             "bone": "Bone", "other": "Other", "L3 vertebra": "L3 vertebra"}
    out = []
    for key, ls in groups.items():
        b = np.isin(small, ls)
        if b.sum() < 20:
            continue
        b = np.pad(b, 1).astype(np.float32)
        try:
            verts, faces, _, _ = marching_cubes(b, level=0.5, spacing=sp, step_size=1 if key in ("imat", "L3 vertebra") else 2)
        except (ValueError, RuntimeError):
            continue
        verts -= np.array(sp)  # undo the padding offset
        # the camera looks from anterior, so the patient's right (+x) is on the viewer's left, as on the slices
        color = "rgb(126,224,138)" if key == "L3 vertebra" else rgb(ROLE_COLOR.get(key, (200, 200, 200)))
        out.append({"name": names.get(key, key), "x": verts[:, 0], "y": verts[:, 1], "z": verts[:, 2],
                    "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2], "color": color, "opacity": style.get(key, 0.6)})
    return out
