"""Browser-side (real-time) parts of the UI, built with st.components.v2.

- Slice viewers (axial / coronal / sagittal): every slice of a view is sent to the browser once (windowed grey JPEGs
  + label PNGs, cached in `window.__bcMpr`), so the wheel, the sliders and the opacity redraw instantly.
- Sidebar live sliders: axial scrub and mask opacity.
- File buttons: Open CT / Open mask / Sample / DICOM ▾ open the browser's file picker of hidden st.file_uploaders,
  so files still go through Streamlit's upload endpoint (large CTs are fine).
- 3D model: Plotly mesh drawn in the browser; the axial-slice plane follows the scrub live.

The pieces talk to each other in the page with window events:
  bodycomp:cur      {axis: 'i'|'j'|'k', value, src}
  bodycomp:opacity  {value, src}
Python hears about a change only when the movement stops (debounced trigger values).
"""
from __future__ import annotations

import base64
import hashlib
import io
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import config
from core import render
from ui import state as S

ss = st.session_state

AXES = {"axial": ("i", "j", "k"), "coronal": ("i", "k", "j"), "sagittal": ("j", "k", "i")}   # column, row, slice


def _ensure_plotly_js() -> str:
    """plotly.js from the installed plotly package, served by Streamlit static serving (works offline)."""
    dst = config.BASE_DIR / "static" / "plotly.min.js"
    if not dst.exists():
        import plotly
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", dst)
    return "app/static/plotly.min.js"


PLOTLY_URL = _ensure_plotly_js()

# ================================================================= slice viewer
VIEWER_CSS = """
.wrap { display: flex; flex-direction: column; gap: 6px; font: 12px ui-monospace, Consolas, monospace; color: #8696a8; }
.stage { position: relative; background: #000; border-radius: 4px; overflow: hidden; display: flex;
         align-items: center; justify-content: center; touch-action: none; user-select: none; }
canvas { display: block; transform-origin: center center; cursor: crosshair; }
.hud { position: absolute; left: 8px; top: 6px; color: #d9e1ea; text-shadow: 0 0 3px #000; pointer-events: none; }
.probe { position: absolute; right: 8px; top: 6px; color: #d9e1ea; text-shadow: 0 0 3px #000; pointer-events: none; text-align: right; }
.tag { position: absolute; left: 8px; bottom: 6px; color: #7ee08a; text-shadow: 0 0 3px #000; pointer-events: none; }
.hint { position: absolute; right: 8px; bottom: 6px; color: #66768a; pointer-events: none; }
.bar { display: flex; align-items: center; gap: 10px; }
input[type=range] { flex: 1; accent-color: #f0b44c; }
.num { min-width: 74px; text-align: right; color: #d9e1ea; }
"""

VIEWER_JS = r"""
const COLOR = { axial: '#f0b44c', coronal: '#4cc3e0', sagittal: '#e273c8' };      // core/labels.VIEW_COLOR
const VIEW_OF = { i: 'sagittal', j: 'coronal', k: 'axial' };
const TITLE = { axial: 'Axial', coronal: 'Coronal', sagittal: 'Sagittal' };
const cache = () => (window.__bcMpr = window.__bcMpr || { stacks: {}, img: {}, lab: {}, frac: {} });
function img(G, uri) {
  let im = G.img[uri];
  if (!im) {                                 // decoded ahead, so scrolling never waits for a decode on draw
    im = new Image(); im.decoding = 'async'; im.src = uri; G.img[uri] = im;
    if (im.decode) im.decode().catch(() => {});
  }
  return im;
}

function build(root) {
  const S = { id: Math.random().toString(36).slice(2), cur: { i: 0, j: 0, k: 0 }, opacity: 0.45,
              local: { i: 0, j: 0, k: 0, o: 0 }, z: 1, px: 0, py: 0, ov: new Map(), asked: new Set(), timers: {} };
  const wrap = document.createElement('div'); wrap.className = 'wrap';
  wrap.innerHTML = `<div class="stage"><canvas></canvas><div class="hud"></div><div class="probe"></div>
    <div class="tag"></div><div class="hint">wheel: slice · ctrl+wheel: zoom · drag: pan · double-click: reset</div></div>
    <div class="bar"><input type="range" min="0" step="1"><span class="num"></span></div>`;
  root.appendChild(wrap);
  S.stage = wrap.querySelector('.stage'); S.cv = wrap.querySelector('canvas'); S.ctx = S.cv.getContext('2d');
  S.hud = wrap.querySelector('.hud'); S.probe = wrap.querySelector('.probe'); S.tag = wrap.querySelector('.tag');
  S.range = wrap.querySelector('input'); S.num = wrap.querySelector('.num');

  // Python gets the whole crosshair once the movement stops (one trigger, so linked axes are not lost).
  S.commit = () => { clearTimeout(S.timers.cur);
    S.timers.cur = setTimeout(() => S.trigger('cur', { ...S.cur, t: Date.now() }), 280); };
  const size = (axis) => ({ i: S.data.dims[0], j: S.data.dims[1], k: S.data.dims[2] }[axis]);
  S.setCur = (axis, v) => {
    v = Math.max(0, Math.min(size(axis) - 1, Math.round(v)));
    if (v === S.cur[axis]) return false;
    S.cur[axis] = v; S.local[axis] = Date.now(); S.draw();
    window.dispatchEvent(new CustomEvent('bodycomp:cur', { detail: { axis, value: v, src: S.id } }));
    return true;
  };
  // Scroll / slider of this view. Linked (Slice viewer tab): the other two planes move by the same fraction of
  // their extent, so all three views change together.
  S.seek = (axis, v) => {
    const d = S.data, f = cache().frac, prev = S.cur[axis], span = (a) => Math.max(1, size(a) - 1);
    v = Math.max(0, Math.min(size(axis) - 1, Math.round(v)));
    if (v === prev) return;
    if (d.link) {
      const dt = (v - prev) / span(axis);
      for (const a of ['i', 'j', 'k']) {
        if (a === axis) continue;
        const base = f[a] !== undefined && Math.round(f[a] * span(a)) === S.cur[a] ? f[a] : S.cur[a] / span(a);
        f[a] = Math.max(0, Math.min(1, base + dt));
        S.setCur(a, f[a] * span(a));
      }
      f[axis] = v / span(axis);
    }
    S.setCur(axis, v); S.commit();
  };
  S.range.addEventListener('input', () => S.seek(S.data.axes[2], +S.range.value));

  let acc = 0;
  S.stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const z = Math.max(1, Math.min(8, S.z * Math.exp(-e.deltaY * 0.0015)));
      const r = S.stage.getBoundingClientRect(), cx = e.clientX - r.left - r.width / 2, cy = e.clientY - r.top - r.height / 2;
      S.px = cx - (cx - S.px) * (z / S.z); S.py = cy - (cy - S.py) * (z / S.z); S.z = z;
      if (S.z === 1) { S.px = 0; S.py = 0; }
      S.applyZoom(); return;
    }
    acc += e.deltaY;
    if (Math.abs(acc) >= 40 || e.deltaMode === 1) { const ax = S.data.axes[2]; S.seek(ax, S.cur[ax] + (acc > 0 ? 1 : -1)); acc = 0; }
  }, { passive: false });

  let down = null;
  S.cv.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, px: S.px, py: S.py, moved: false };
    S.cv.setPointerCapture(e.pointerId); });
  S.cv.addEventListener('pointermove', (e) => {
    if (down) {
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.hypot(dx, dy) > 4) down.moved = true;
      if (down.moved && S.z > 1) { S.px = down.px + dx; S.py = down.py + dy; S.applyZoom(); }
    }
    S.hover(e);
  });
  S.cv.addEventListener('pointerup', (e) => {
    if (down && !down.moved) { const v = S.voxel(e); if (v) S.trigger('click', { ...v.ijk, t: Date.now() }); }
    down = null;
  });
  S.cv.addEventListener('pointerleave', () => { S.probe.textContent = ''; });
  S.stage.addEventListener('dblclick', () => { S.z = 1; S.px = 0; S.py = 0; S.applyZoom(); });

  S.voxel = (e) => {
    const r = S.cv.getBoundingClientRect(), d = S.data;
    const c = Math.floor((e.clientX - r.left) / r.width * d.ncol), rr = Math.floor((e.clientY - r.top) / r.height * d.nrow);
    if (c < 0 || rr < 0 || c >= d.ncol || rr >= d.nrow) return null;
    const [ca, ra, sa] = d.axes, ijk = { ...S.cur };
    ijk[ca] = d.ncol - 1 - c; ijk[ra] = d.nrow - 1 - rr; ijk[sa] = S.cur[sa];
    return { c, r: rr, ijk };
  };
  S.hover = (e) => {
    const v = S.voxel(e), d = S.data; if (!v) { S.probe.textContent = ''; return; }
    const lab = S.labels(S.cur[d.axes[2]]); const L = lab ? lab[v.r * d.ncol + v.c] : 0;
    S.probe.textContent = `${v.ijk.i}, ${v.ijk.j}, ${v.ijk.k}` + (L && d.names[L] ? ` · ${d.names[L]}` : '');
  };
  S.applyZoom = () => { S.cv.style.transform = `translate(${S.px}px, ${S.py}px) scale(${S.z})`; S.draw(); };
  S.layout = () => {
    const d = S.data; if (!d) return;
    S.stage.style.height = d.height + 'px';
    if (S.cv.width !== d.ncol || S.cv.height !== d.nrow) { S.cv.width = d.ncol; S.cv.height = d.nrow; S.ov.clear(); }
    const W = S.stage.clientWidth || 400, H = d.height, pw = d.ncol * d.csp, ph = d.nrow * d.rsp;
    const s = Math.min(W / pw, H / ph);
    S.cv.style.width = (pw * s) + 'px'; S.cv.style.height = (ph * s) + 'px';
  };
  new ResizeObserver(() => { S.layout(); S.draw(); }).observe(S.stage);

  S.labels = (idx) => {
    const G = cache(), d = S.data, uris = G.stacks[d.mask_id]; if (!uris) return null;
    const key = d.mask_id + '|' + idx; if (G.lab[key]) return G.lab[key];
    const im = img(G, uris[idx]); if (!im.complete || !im.naturalWidth) { im.onload = () => S.draw(); return null; }
    const oc = document.createElement('canvas'); oc.width = d.ncol; oc.height = d.nrow;
    const ox = oc.getContext('2d', { willReadFrequently: true }); ox.drawImage(im, 0, 0);
    const px = ox.getImageData(0, 0, d.ncol, d.nrow).data, out = new Uint8Array(d.ncol * d.nrow);
    for (let p = 0; p < out.length; p++) out[p] = px[p * 4];
    G.lab[key] = out; return out;
  };
  S.overlay = (idx) => {                     // full-alpha label colours; opacity is applied when drawing
    const d = S.data, key = `${d.mask_id}|${idx}|${d.color_sig}`;
    if (S.ov.has(key)) return S.ov.get(key);
    const lab = S.labels(idx); if (!lab) return null;
    if (S.lutSig !== d.color_sig) {          // label → RGBA as one 32-bit word (little-endian ABGR)
      S.lut = new Uint32Array(256); S.lutSig = d.color_sig;
      for (const L in d.colors) { const c = d.colors[L]; S.lut[L] = ((255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0]) >>> 0; }
    }
    const oc = document.createElement('canvas'); oc.width = d.ncol; oc.height = d.nrow;
    const ox = oc.getContext('2d'), id = ox.createImageData(d.ncol, d.nrow), px = new Uint32Array(id.data.buffer), lut = S.lut;
    for (let p = 0; p < lab.length; p++) px[p] = lut[lab[p]];
    ox.putImageData(id, 0, 0);
    if (S.ov.size > 400) S.ov.clear();
    S.ov.set(key, oc); return oc;
  };
  // At most one paint per frame, however many scroll / slider / event updates arrive in between.
  S.draw = () => { if (!S.raf) S.raf = requestAnimationFrame(() => { S.raf = 0; S.paint(); }); };
  S.paint = () => {
    const d = S.data, G = cache(); if (!d) return;
    const [ca, ra, sa] = d.axes, idx = S.cur[sa], n = d.dims[{ i: 0, j: 1, k: 2 }[sa]];
    const ctx = S.ctx, uris = G.stacks[d.gray_id];
    S.range.max = n - 1; if (+S.range.value !== idx) S.range.value = idx;
    S.num.textContent = `${idx + 1} / ${n}`;
    S.hud.innerHTML = `<b style="color:${COLOR[d.view]}">${TITLE[d.view]}</b> ${idx + 1} / ${n}`;
    S.tag.textContent = sa === 'k' && d.l3 === idx ? 'L3' : '';
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, d.ncol, d.nrow);
    if (!uris) return;
    const im = img(G, uris[idx]);
    if (!im.complete || !im.naturalWidth) { im.onload = () => S.draw(); return; }
    ctx.drawImage(im, 0, 0, d.ncol, d.nrow);
    if (d.mask_id && S.opacity > 0) { const ov = S.overlay(idx); if (ov) { ctx.globalAlpha = S.opacity; ctx.drawImage(ov, 0, 0); ctx.globalAlpha = 1; } }
    const sx = d.ncol / Math.max(1, S.cv.getBoundingClientRect().width), sy = d.nrow / Math.max(1, S.cv.getBoundingClientRect().height);
    const X = (a) => d.ncol - 1 - a + 0.5, Y = (b) => d.nrow - 1 - b + 0.5;
    if (d.crosshair) {
      ctx.globalAlpha = 0.75;
      ctx.strokeStyle = COLOR[VIEW_OF[ca]]; ctx.lineWidth = sx; ctx.beginPath(); ctx.moveTo(X(S.cur[ca]), 0); ctx.lineTo(X(S.cur[ca]), d.nrow); ctx.stroke();
      ctx.strokeStyle = COLOR[VIEW_OF[ra]]; ctx.lineWidth = sy; ctx.beginPath(); ctx.moveTo(0, Y(S.cur[ra])); ctx.lineTo(d.ncol, Y(S.cur[ra])); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (ra === 'k' && d.l3 !== null && d.l3 !== undefined) {
      ctx.strokeStyle = ctx.fillStyle = '#7ee08a'; ctx.lineWidth = sy; ctx.setLineDash([6 * sx, 4 * sx]);
      ctx.beginPath(); ctx.moveTo(0, Y(d.l3)); ctx.lineTo(d.ncol, Y(d.l3)); ctx.stroke(); ctx.setLineDash([]);
    }
    const P = (a) => [X(a[0]), Y(a[1])];
    ctx.strokeStyle = ctx.fillStyle = '#ffd166'; ctx.lineWidth = 2 * sx;
    for (const m of d.meas) {
      if (m.slice !== idx) continue;
      const [x0, y0] = P(m.a), [x1, y1] = P(m.b);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.save(); ctx.translate((x0 + x1) / 2, (y0 + y1) / 2); ctx.scale(sx, sy); ctx.font = '13px sans-serif';
      ctx.fillText(`${m.length_mm.toFixed(1)} mm`, 6, -6); ctx.restore();
    }
    if (d.pending && d.pending.slice === idx) {
      const [x, y] = P(d.pending.a); ctx.beginPath(); ctx.ellipse(x, y, 4 * sx, 4 * sy, 0, 0, 7); ctx.stroke();
    }
  };
  return S;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const G = cache();
  if (data.gray) G.stacks[data.gray_id] = data.gray;
  if (data.mask) G.stacks[data.mask_id] = data.mask;
  const S = parentElement.__bc || (parentElement.__bc = build(parentElement));
  S.trigger = setTriggerValue; S.data = data;
  for (const id of [data.gray_id, data.mask_id]) {
    if (id && !G.stacks[id] && !S.asked.has(id)) { S.asked.add(id); setTriggerValue('missing', id); }
  }
  for (const id of [data.gray_id, data.mask_id]) if (id && G.stacks[id]) G.stacks[id].forEach((u) => img(G, u));
  const now = Date.now();
  for (const a of ['i', 'j', 'k']) if (now - S.local[a] > 1000) S.cur[a] = data.cur[a];
  if (now - S.local.o > 1000) S.opacity = data.opacity;
  S.layout(); S.paint();
  if (S.onCur) { window.removeEventListener('bodycomp:cur', S.onCur); window.removeEventListener('bodycomp:opacity', S.onOp); }
  S.onCur = (e) => { if (e.detail.src !== S.id) { S.cur[e.detail.axis] = e.detail.value; S.local[e.detail.axis] = Date.now(); S.draw(); } };
  S.onOp = (e) => { S.opacity = e.detail.value; S.local.o = Date.now(); S.draw(); };
  window.addEventListener('bodycomp:cur', S.onCur); window.addEventListener('bodycomp:opacity', S.onOp);
  return () => { window.removeEventListener('bodycomp:cur', S.onCur); window.removeEventListener('bodycomp:opacity', S.onOp); };
}
"""

# ================================================================= sidebar live sliders
SLIDERS_CSS = """
.sl { display: flex; flex-direction: column; gap: 10px; font: 14px "Source Sans Pro", sans-serif; color: #d9e1ea; }
.lbl b { color: #f0b44c; font-weight: 600; }
input[type=range] { width: 100%; accent-color: #f0b44c; margin: 4px 0 0; }
"""

SLIDERS_JS = r"""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let S = parentElement.__bcs;
  if (!S) {
    S = parentElement.__bcs = { id: 'sliders', k: 0, o: 45, local: { k: 0, o: 0 }, timers: {} };
    const box = document.createElement('div'); box.className = 'sl';
    box.innerHTML = `<div class="k"><div class="lbl"></div><input type="range" min="0" step="1"></div>
                     <div class="o"><div class="lbl"></div><input type="range" min="0" max="100" step="1"></div>`;
    parentElement.appendChild(box);
    S.kl = box.querySelector('.k .lbl'); S.kr = box.querySelector('.k input');
    S.ol = box.querySelector('.o .lbl'); S.or = box.querySelector('.o input');
    S.show = () => { const d = S.data; S.kr.max = d.nz - 1; S.kr.value = S.k; S.or.value = S.o;
      S.kl.innerHTML = `Axial scrub · <b>${S.k + 1}</b> / ${d.nz}${d.l3 === S.k ? ' · L3' : ''}`;
      S.ol.innerHTML = `Mask opacity · <b>${S.o}%</b>`; };
    const later = (name, value) => { clearTimeout(S.timers[name]); S.timers[name] = setTimeout(() => S.trigger(name, value), 280); };
    S.kr.addEventListener('input', () => {
      S.k = +S.kr.value; S.local.k = Date.now(); S.show();
      window.dispatchEvent(new CustomEvent('bodycomp:cur', { detail: { axis: 'k', value: S.k, src: S.id } }));
      later('k', S.k);
    });
    S.or.addEventListener('input', () => {
      S.o = +S.or.value; S.local.o = Date.now(); S.show();
      window.dispatchEvent(new CustomEvent('bodycomp:opacity', { detail: { value: S.o / 100, src: S.id } }));
      later('opacity', S.o);
    });
  }
  S.trigger = setTriggerValue; S.data = data;
  const now = Date.now();
  if (now - S.local.k > 1000) S.k = data.k;
  if (now - S.local.o > 1000) S.o = data.opacity;
  S.show();
  if (S.onCur) window.removeEventListener('bodycomp:cur', S.onCur);
  S.onCur = (e) => { if (e.detail.src !== S.id && e.detail.axis === 'k') { S.k = e.detail.value; S.local.k = Date.now(); S.show(); } };
  window.addEventListener('bodycomp:cur', S.onCur);
  return () => window.removeEventListener('bodycomp:cur', S.onCur);
}
"""

# ================================================================= file buttons
FILES_CSS = """
.fb { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font: 14px "Source Sans Pro", sans-serif; }
button { font: inherit; color: #d9e1ea; background: #11171f; border: 1px solid #2e3b4b; border-radius: 8px;
         padding: 7px 10px; cursor: pointer; width: 100%; display: flex; align-items: center; justify-content: center; gap: 6px; }
button:hover:not(:disabled) { border-color: #f0b44c; color: #f0b44c; }
button:disabled { opacity: .45; cursor: not-allowed; }
.dd { position: relative; }
.menu { position: absolute; z-index: 10; top: calc(100% + 4px); left: 0; right: 0; background: #161e28; border: 1px solid #2e3b4b;
        border-radius: 8px; padding: 4px; display: none; box-shadow: 0 8px 24px rgba(0,0,0,.5); }
.menu.open { display: block; }
.menu button { border: 0; background: transparent; justify-content: flex-start; padding: 7px 10px; }
.menu button:hover { background: #232e3b; color: #f0b44c; }
.note { grid-column: 1 / -1; font: 12px ui-monospace, Consolas, monospace; color: #f0b44c; min-height: 0; }
.caret { font-size: 10px; opacity: .8; }
"""

FILES_HTML = """
<div class="fb">
  <button data-pick="bc_up_ct">Open CT</button>
  <button data-pick="bc_up_mask" title="Segmentation file(s); its CT and vertebrae_*.nii.gz are found automatically when they are in the repo">Open mask</button>
  <button data-act="sample">Sample</button>
  <div class="dd"><button class="drop" data-need="dicom">DICOM <span class="caret">▼</span></button>
    <div class="menu"><button data-pick="bc_up_dir">DICOM folder</button><button data-pick="bc_up_zip">DICOM .zip</button></div>
  </div>
  <div class="note"></div>
</div>
"""

FILES_JS = r"""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement;
  const q = (s) => root.querySelector(s), qa = (s) => [...root.querySelectorAll(s)];
  const note = q('.note'), closeAll = () => qa('.menu').forEach((m) => m.classList.remove('open'));
  const input = (key) => document.querySelector(`.st-key-${key} input[type=file]`);
  const need = { dicom: data.dicom };
  q('[data-pick=bc_up_ct]').disabled = data.busy;
  q('[data-pick=bc_up_mask]').disabled = data.busy;
  q('[data-act=sample]').disabled = data.busy;
  qa('.drop').forEach((b) => { b.disabled = !need[b.dataset.need]; });
  q('[data-need=dicom]').title = data.dicom ? 'Converted on the inference server' : 'Inference server offline';
  note.textContent = '';
  if (root.__bcf) { root.__bcf.trigger = setTriggerValue; return; }
  const F = (root.__bcf = { trigger: setTriggerValue });
  qa('[data-pick]').forEach((b) => b.addEventListener('click', (e) => {
    e.stopPropagation(); closeAll();
    const el = input(b.dataset.pick);
    if (!el) { note.textContent = 'Uploader not ready, try again.'; return; }
    el.onchange = () => { const f = el.files; if (f && f.length) note.textContent = `Uploading ${f.length > 1 ? f.length + ' files' : f[0].name}…`; };
    el.click();
  }));
  q('[data-act=sample]').addEventListener('click', () => F.trigger('action', 'sample'));
  qa('.drop').forEach((b) => b.addEventListener('click', (e) => {
    e.stopPropagation(); const m = b.parentElement.querySelector('.menu'), was = m.classList.contains('open');
    closeAll(); if (!was) m.classList.add('open');
  }));
  document.addEventListener('click', closeAll);
}
"""

# ================================================================= 3D model
MESH_JS = r"""
const cache = () => (window.__bcMesh = window.__bcMesh || {});
function loadPlotly(url) {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (!window.__bcPlotlyP) window.__bcPlotlyP = new Promise((res, rej) => {
    const s = document.createElement('script'); s.src = new URL(url, document.baseURI).href;
    s.onload = () => res(window.Plotly); s.onerror = rej; document.head.appendChild(s);
  });
  return window.__bcPlotlyP;
}
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const C = cache();
  if (data.meshes) C[data.mesh_id] = data.meshes;
  let M = parentElement.__bcm;
  if (!M) {
    M = parentElement.__bcm = { div: document.createElement('div'), sig: '', local: 0 };
    M.div.style.width = '100%'; parentElement.appendChild(M.div);
  }
  M.data = data;
  M.div.style.height = data.height + 'px';
  const meshes = C[data.mesh_id];
  if (!meshes) { if (M.asked !== data.mesh_id) { M.asked = data.mesh_id; setTriggerValue('missing', data.mesh_id); } return; }
  const zOf = (k) => (k + 0.5) * data.sz;
  if (Date.now() - M.local > 1000) M.k = data.k;
  // Move only the slice plane: Plotly.restyle would rebuild every tissue mesh (normals included) on each step.
  M.movePlane = (Plotly) => {
    const z = zOf(M.k), zs = [z, z, z, z], gd = M.div;
    const fd = gd._fullData && gd._fullData[M.plane], sc = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
    const obj = fd && sc && sc.traces && sc.traces[fd.uid];
    if (obj && obj.update && sc.glplot) {
      gd.data[M.plane].z = zs; fd.z = zs;
      obj.update(fd); sc.glplot.redraw();
    } else Plotly.restyle(gd, { z: [zs] }, [M.plane]);
  };
  loadPlotly(data.plotly_url).then((Plotly) => {
    const sig = data.mesh_id + '|' + data.height;
    if (M.sig !== sig) {
      M.sig = sig;
      const traces = meshes.map((m) => ({ type: 'mesh3d', x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k, color: m.color,
        opacity: m.opacity, name: m.name, showlegend: true, hoverinfo: 'name', flatshading: false,
        lighting: { ambient: 0.45, diffuse: 0.8, specular: 0.25, roughness: 0.6 } }));
      const z = zOf(M.k), ex = data.ex, ey = data.ey;
      traces.push({ type: 'mesh3d', x: [0, ex, ex, 0], y: [0, 0, ey, ey], z: [z, z, z, z], i: [0, 0], j: [1, 2], k: [2, 3],
        color: '#f0b44c', opacity: 0.22, name: 'Axial slice', hoverinfo: 'name', showlegend: true });
      M.plane = traces.length - 1;
      // fixed ranges (the whole volume), so the plane is never clipped when it leaves the segmented range
      const ax = (hi) => ({ visible: false, showbackground: false, range: [0, hi], autorange: false });
      Plotly.react(M.div, traces, {
        margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: 'rgba(0,0,0,0)', uirevision: 'mesh',
        font: { color: '#d9e1ea', size: 11 }, legend: { orientation: 'h', y: 0.02, x: 0 },
        scene: { xaxis: ax(ex), yaxis: ax(ey), zaxis: ax(data.ez), aspectmode: 'data', bgcolor: 'rgba(0,0,0,0)',
                 camera: { eye: { x: 0, y: 2.0, z: 0.6 }, up: { x: 0, y: 0, z: 1 } } },
      }, { displaylogo: false, responsive: true });
    } else M.movePlane(Plotly);
    if (M.onCur) window.removeEventListener('bodycomp:cur', M.onCur);
    M.onCur = (e) => {
      if (e.detail.axis !== 'k') return;
      M.k = e.detail.value; M.local = Date.now();
      if (M.raf) return;
      M.raf = requestAnimationFrame(() => { M.raf = 0; M.movePlane(Plotly); });
    };
    window.addEventListener('bodycomp:cur', M.onCur);
  });
  return () => { if (M.onCur) window.removeEventListener('bodycomp:cur', M.onCur); };
}
"""

_viewer = st.components.v2.component("bodycomp_slice_viewer", css=VIEWER_CSS, js=VIEWER_JS)
_sliders = st.components.v2.component("bodycomp_live_sliders", css=SLIDERS_CSS, js=SLIDERS_JS)
_files = st.components.v2.component("bodycomp_file_buttons", html=FILES_HTML, css=FILES_CSS, js=FILES_JS)
_mesh = st.components.v2.component("bodycomp_mesh3d", js=MESH_JS, isolate_styles=False)


# ================================================================= stacks (sent once per browser session)
def _uri(img: np.ndarray, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        Image.fromarray(img).save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _stack(sid: str, build) -> list[str]:
    cache = ss.setdefault("live_stacks", {})
    if sid not in cache:
        with st.spinner("Preparing slices for the viewer…"):
            cache[sid] = build()
        family = sid.rsplit(":", 1)[0].split(":")[:2]      # kind + view: keep only the newest of each
        for old in [k for k in cache if k != sid and k.split(":")[:2] == family]:
            cache.pop(old)
    return cache[sid]


def gray_stack(v: dict, view: str, wl: dict) -> tuple[str, list[str]]:
    hu, axis = v["hu"], AXES[view][2]
    n = hu.shape["ijk".index(axis)]
    sid = f"g:{view}:{v['id']}:{wl['w']}:{wl['l']}"
    return sid, _stack(sid, lambda: [_uri(render.window(render.view_slice(hu, view, {axis: s}), wl["w"], wl["l"]), "JPEG")
                                     for s in range(n)])


def mask_stack(mask: dict, view: str) -> tuple[str, list[str]]:
    m, axis = mask["data"], AXES[view][2]
    n = m.shape["ijk".index(axis)]
    sid = f"m:{view}:{mask['id']}"
    return sid, _stack(sid, lambda: [_uri(render.view_slice(m, view, {axis: s}), "PNG") for s in range(n)])


def _once(sid: str | None, payload):
    """The payload only the first time; afterwards the browser has it (it asks again via `missing` if not)."""
    if sid is None:
        return None
    sent = ss.setdefault("live_sent", set())
    if sid in sent:
        return None
    sent.add(sid)
    return payload


# ================================================================= callbacks
def _value(key: str, name: str):
    res = ss.get(key)
    return res.get(name) if isinstance(res, dict) else getattr(res, name, None)


def _on_cur(key: str) -> None:
    c = _value(key, "cur")
    if c and ss.vol is not None:
        ss.cur = {**ss.cur, **{a: max(0, min(n - 1, int(c[a]))) for a, n in zip("ijk", ss.vol["shape"]) if a in c}}


def _on_k(key: str) -> None:
    k = _value(key, "k")
    if k is not None and ss.vol is not None:
        ss.cur = {**ss.cur, "k": max(0, min(ss.vol["shape"][2] - 1, int(k)))}


def _on_opacity(key: str) -> None:
    o = _value(key, "opacity")
    if o is not None:
        ss.opacity = max(0, min(100, int(o))) / 100


def _on_click(view: str, key: str) -> None:
    c = _value(key, "click")
    if c and ss.vol is not None:
        S.apply_click(view, int(c["i"]), int(c["j"]), int(c["k"]))


def _on_missing(key: str) -> None:
    sid = _value(key, "missing")
    if sid:
        ss.setdefault("live_sent", set()).discard(sid)


def _on_action(key: str) -> None:
    if _value(key, "action") == "sample":
        S.load_sample()


# ================================================================= mounts
def viewer(a: dict, view: str, key: str, height: int, crosshair: bool = True, link: bool = False) -> None:
    v, mask, labels = a["vol"], a["mask"], a["labels"]
    ca, ra, sa = AXES[view]
    gid, guris = gray_stack(v, view, a["wl"])
    mid, muris = mask_stack(mask, view) if mask else (None, None)
    colors = {int(L): list(d["color"]) for L, d in labels.items() if d["visible"]} if mask else {}
    sp = dict(zip("ijk", v["spacing"]))
    dims = list(v["shape"])
    data = {
        "view": view, "axes": [ca, ra, sa], "dims": dims, "ncol": dims["ijk".index(ca)], "nrow": dims["ijk".index(ra)],
        "csp": sp[ca], "rsp": sp[ra], "height": height,
        "gray_id": gid, "gray": _once(gid, guris), "mask_id": mid, "mask": _once(mid, muris),
        "cur": a["cur"], "l3": a["l3"], "crosshair": crosshair, "link": link, "opacity": a["opacity"],
        "colors": colors, "color_sig": hashlib.md5(repr(sorted(colors.items())).encode()).hexdigest()[:8],
        "names": {int(L): d["name"] for L, d in labels.items()},
        "meas": [{"slice": m["slice"], "a": list(m["a"]), "b": list(m["b"]), "length_mm": m["length_mm"]}
                 for m in ss.meas if m["view"] == view],
        "pending": ({"slice": ss.pending["slice"], "a": list(ss.pending["a"])}
                    if ss.pending and ss.pending["view"] == view else None),
    }
    _viewer(key=key, data=data, on_cur_change=lambda: _on_cur(key), on_click_change=lambda: _on_click(view, key),
            on_missing_change=lambda: _on_missing(key))


def sliders(v: dict) -> None:
    _sliders(key="live_sliders", data={"k": ss.cur["k"], "nz": v["shape"][2], "l3": ss.l3, "opacity": round(ss.opacity * 100)},
             on_k_change=lambda: _on_k("live_sliders"), on_opacity_change=lambda: _on_opacity("live_sliders"))


def file_buttons(*, busy: bool, has_ct: bool, dicom: bool) -> None:
    _files(key="file_buttons", data={"busy": busy, "has_ct": has_ct, "dicom": dicom},
           on_action_change=lambda: _on_action("file_buttons"))


def mesh(a: dict, height: int = 420) -> None:
    v, mask = a["vol"], a["mask"]
    sig = (mask["id"], tuple((L, d["role"], d["visible"], "L3" in d["name"]) for L, d in a["labels"].items()))
    mesh_id = "mesh:" + hashlib.md5(repr(sig).encode()).hexdigest()[:12]
    cache = ss.setdefault("live_meshes", {})
    if mesh_id not in cache:
        with st.spinner("Building 3D surfaces…"):
            meshes = render.build_meshes(mask["data"], a["labels"], v["spacing"], config.MESH_MAX_VOXELS)
        cache.clear()
        cache[mesh_id] = [{"name": m["name"], "color": m["color"], "opacity": m["opacity"],
                           "x": np.round(m["x"], 1).tolist(), "y": np.round(m["y"], 1).tolist(), "z": np.round(m["z"], 1).tolist(),
                           "i": m["i"].tolist(), "j": m["j"].tolist(), "k": m["k"].tolist()} for m in meshes]
    nx, ny, nz = v["shape"]
    sx, sy, sz = v["spacing"]
    _mesh(key="mesh3d", data={"mesh_id": mesh_id, "meshes": _once(mesh_id, cache[mesh_id]), "ex": nx * sx, "ey": ny * sy,
                              "ez": nz * sz, "sz": sz, "k": a["cur"]["k"], "height": height, "plotly_url": PLOTLY_URL},
          on_missing_change=lambda: _on_missing("mesh3d"))
