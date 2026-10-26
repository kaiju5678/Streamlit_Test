"""The six tabs of the workstation (Overview, Slice viewer, Metrics table, AI report, Transmitted JSON, Run log)."""
from __future__ import annotations

import html
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import config
from core import analysis as an
from core import labels as lb
from core import biomarkers, llm_client, render
from ui import live
from ui import state as S

ss = st.session_state
ROLE_CSS = {"muscle": lb.rgb(lb.ROLE_COLOR["muscle"]), "vat": lb.rgb(lb.ROLE_COLOR["vat"]),
            "sat": lb.rgb(lb.ROLE_COLOR["sat"]), "imat": lb.rgb(lb.ROLE_COLOR["imat"])}


def f1(v) -> str:
    return "–" if v is None or not math.isfinite(v) else f"{v:.1f}"


def f0(v) -> str:
    return "–" if v is None or not math.isfinite(v) else f"{round(v):,}"


def h(s) -> str:
    return html.escape(str(s))


def md(x: str) -> None:
    st.markdown(x, unsafe_allow_html=True)


def panel_head(title: str, sub: str = "") -> None:
    md(f"<div class='panel-head'><h3>{title}</h3><p>{sub}</p></div>")


def pill(flag, yes: str, no: str) -> str:
    if flag is None:
        return "<span class='pill na'>no reference</span>"
    return f"<span class='pill bad'>▲ {h(yes)}</span>" if flag else f"<span class='pill ok'>● {h(no)}</span>"


def viewport(a: dict, view: str, key: str, height: int, crosshair: bool = True, link: bool = False,
             checks: bool = False) -> None:
    live.viewer(a, view, key, height, crosshair, link, checks)


def role_parts(rv: dict) -> list[dict]:
    return [{"name": "Muscle", "value": rv["muscle"]["cm3"], "color": ROLE_CSS["muscle"]},
            {"name": "Visceral fat", "value": rv["vat"]["cm3"], "color": ROLE_CSS["vat"]},
            {"name": "Subcutaneous fat", "value": rv["sat"]["cm3"], "color": ROLE_CSS["sat"]},
            {"name": "Intermuscular fat", "value": rv["imat"]["cm3"], "color": ROLE_CSS["imat"]}]


def indices(a: dict) -> list[dict]:
    ev, p = a["ev"], a["patient"]
    if not ev:
        return []
    rv = an.role_volumes(a["stats"], a["labels"], a["vol"]["spacing"]) if a["stats"] else None
    adipose = rv["sat"]["cm3"] + rv["vat"]["cm3"] + rv["imat"]["cm3"] if rv else None
    tissue = adipose + rv["muscle"]["cm3"] if rv else None
    src = lambda c: c["source"] if c else ""  # noqa: E731
    return [
        {"k": "Skeletal muscle index", "v": f1(ev["smi"]), "u": "cm²/m²", "ref": f"≥ {ev['smi_cut']}" if ev["smi_cut"] else "–",
         "flag": ev["low_muscle"], "yes": "Low", "no": "Normal", "src": src(an.ref_smi(p))},
        {"k": "Muscle attenuation", "v": f1(ev["muscle_hu"]), "u": "HU", "ref": f"≥ {ev['hu_cut']}" if ev["hu_cut"] else "–",
         "flag": ev["myosteatosis"], "yes": "Low", "no": "Normal", "src": src(an.ref_smra(p))},
        {"k": "Visceral fat area", "v": f1(ev["vat"]), "u": "cm²", "ref": f"≤ {ev['vat_cut']}" if ev["vat_cut"] else "–",
         "flag": ev["visceral_obesity"], "yes": "High", "no": "Normal", "src": src(an.ref_vat_area(p))},
        {"k": "VAT / SAT ratio", "v": "–" if ev["vat_sat"] is None else f"{ev['vat_sat']:.2f}", "u": "", "ref": "–",
         "flag": None, "yes": "", "no": "", "src": ""},
        {"k": "Adipose share", "v": f1(adipose / tissue * 100) if tissue else "–", "u": "%", "ref": "–",
         "flag": None, "yes": "", "no": "", "src": ""},
    ]




# ================================================================= Overview
OV_VIEW_H = 340  # px, image area of the Overview axial view (the 3D stage beside it follows)
VIEW_H = 300     # px, image area of each Slice viewer view


def overview(a: dict) -> None:
    v, mask, labels, stats, ev = a["vol"], a["mask"], a["labels"], a["stats"], a["ev"]
    rv = an.role_volumes(stats, labels, v["spacing"]) if stats else None
    adipose = rv["sat"]["cm3"] + rv["vat"]["cm3"] + rv["imat"]["cm3"] if rv else None
    tissue = adipose + rv["muscle"]["cm3"] if rv else None
    extent = round((stats["k_max"] - stats["k_min"] + 1) * v["spacing"][2], 1) if stats and stats["k_min"] >= 0 else None

    vs_s = f"L3 VAT/SAT ratio {ev['vat_sat']:.2f}" if ev and ev["vat_sat"] is not None else "VAT / SAT ratio at L3 –"
    hu_s = ("set L3 to compute" if not ev else
            pill(ev["myosteatosis"], f"below {ev['hu_cut']} HU", f"≥ {ev['hu_cut']} HU") if ev["hu_cut"] else "needs height and weight")
    md("<section class='kpis'>"
       f"<div class='kpi'><span class='k'>Muscle volume</span><span class='v'>{f0(rv['muscle']['cm3'] if rv else None)}<small>cm³</small></span>"
       f"<span class='s'>{f'over {extent} mm segmented range' if extent else 'segmented range'}</span></div>"
       f"<div class='kpi'><span class='k'>Muscle attenuation · L3</span><span class='v'>{f1(ev['muscle_hu'] if ev else None)}<small>HU</small></span>"
       f"<span class='s'>{hu_s}</span></div>"
       f"<div class='kpi'><span class='k'>Visceral fat volume</span><span class='v'>{f0(rv['vat']['cm3'] if rv else None)}<small>cm³</small></span>"
       f"<span class='s'>{vs_s}</span></div>"
       f"<div class='kpi'><span class='k'>Total adipose</span><span class='v'>{f0(adipose)}<small>cm³</small></span>"
       f"<span class='s'>{f'{f1(adipose / tissue * 100)}% of segmented tissue' if tissue else 'VAT + SAT + IMAT'}</span></div>"
       "</section>")

    # Each row sits in an "eqrow" container: assets/style.css stretches both boxes of the row to the same height
    c1, c2 = st.container(key="eqrow_ov_view").columns(2, gap="medium")
    with c1.container(border=True):
        panel_head("Axial slice with segmentation", "Wheel or slider to change slice · click to move the crosshair")
        viewport(a, "axial", "ov_axial", OV_VIEW_H, checks=True)
    with c2.container(border=True):
        panel_head("3D model", "Surface of each label from the segmentation mask" if mask is not None
                   else "Body surface and bone from the CT · tissue surfaces appear after segmentation")
        # the 2D box also has a slider row under its image, so the 3D stage is that much taller to line up
        live.mesh(a, OV_VIEW_H + 30)

    c1, c2 = st.container(key="eqrow_ov_info").columns(2, gap="medium")
    rep = a["report"]
    with c1.container(border=True):
        if rep:
            sub = (f"{rep['model']} · {rep['latencyMs'] / 1000:.1f} s · {rep['generatedAt'][:19].replace('T', ' ')}" if rep["source"] == "llm"
                   else "Rule-based draft · the model failed, see AI report" if rep.get("fallbackReason")
                   else "Rule-based draft · no language model configured")
        else:
            sub = "No draft yet"
        badge = "<span class='pill ok'>Signed</span>" if a["signature"] else "<span class='pill na'>Draft</span>" if rep else ""
        md(f"<div class='panel-head'><h3>Assessment {badge}</h3><p>{h(sub)}</p></div>")
        if rep:
            n = rep["narrative"]
            if rep["payloadKey"] != a["payload_key"]:
                md("<p class='banner warn'>Metrics changed after this draft was written. Regenerate before signing.</p>")
            out = f"<div class='prose'><p>{h(n['summary'])}</p>"
            if n["lifestyle"]:
                out += "<h4>Lifestyle considerations</h4><ul>" + "".join(f"<li>{h(r)}</li>" for r in n["lifestyle"]) + "</ul>"
            md(out + "<p class='disc'>Draft for the reading physician. Review and sign it on the AI report tab.</p></div>")
        else:
            md(f"<p class='empty'>{'Generate a draft from the eight body-composition biomarkers. Only those eight numbers are sent.' if a['payload'] else 'Load a segmentation first.'}</p>")
            st.button("Generate AI report" if llm_client.configured() else "Generate rule-based draft", type="primary",
                      on_click=S.cb_generate, key="ov_gen", disabled=a["payload"] is None)
    with c2.container(border=True):
        panel_head("Tissue composition", "Volumes over the segmented range, and L3 indices against the configured references")
        if rv and tissue:
            parts = role_parts(rv)
            d1, d2 = st.columns([2, 3], vertical_alignment="center")
            d1.plotly_chart(render.donut(parts, f0(tissue), "cm³ total"), key="ov_donut", config={"displayModeBar": False}, theme=None)
            d2.markdown("<ul class='comp-list'>" + "".join(
                f"<li><span><i style='background:{p['color']}'></i>{p['name']}</span><b>{f0(p['value'])} cm³</b>"
                f"<em>{f1(p['value'] / tissue * 100)}%</em><span class='comp-bar'><span style='width:{p['value'] / tissue * 100:.1f}%;"
                f"background:{p['color']}'></span></span></li>" for p in parts) + "</ul>", unsafe_allow_html=True)
        else:
            md("<p class='empty'>No segmentation loaded.</p>")
        rows = indices(a)
        if rows:
            md("<table class='m idx'><thead><tr><th>L3 index</th><th>Value</th><th>Reference</th><th>Status</th></tr></thead><tbody>"
               + "".join(f"<tr title='{h(r['src'])}'><td>{r['k']}</td><td>{r['v']} <small>{r['u']}</small></td><td>{r['ref']}</td><td>"
                         + (f"<span class='pill na'>{'no reference' if r['ref'] == '–' else 'needs patient data'}</span>" if r["flag"] is None
                            else pill(r["flag"], r["yes"], r["no"])) + "</td></tr>" for r in rows) + "</tbody></table>")
        e1, e2 = st.columns(2)
        e1.download_button("Export report (HTML / PDF)", data=report_html(a) if rep else "", file_name=f"{ss.ref}_report.html",
                           mime="text/html", disabled=not rep, key="ov_dl_rep", width="stretch", on_click="ignore")
        e2.download_button("Export metrics (CSV)", data=csv_text(a) if mask else "", file_name=f"{ss.ref}_metrics.csv",
                           mime="text/csv", disabled=not mask, key="ov_dl_csv", width="stretch", on_click="ignore")


def csv_text(a: dict) -> str:
    v = a["vol"]
    return an.metrics_csv(ref=ss.ref, shape=v["shape"], spacing=v["spacing"], mask=a["mask"], labels=a["labels"],
                          stats=a["stats"], patient=a["patient"], l3=a["l3"], hu=v["hu"], body=a["body"])


# ================================================================= Slice viewer
def viewer(a: dict) -> None:
    # two eqrow rows (axial | coronal, sagittal | study) so the boxes side by side are the same height
    c1, c2 = st.container(key="eqrow_v_top").columns(2, gap="small")
    for col, view in ((c1, "axial"), (c2, "coronal")):
        with col.container(border=True):
            viewport(a, view, f"v_{view}", VIEW_H, checks=view == "axial")
    c1, c2 = st.container(key="eqrow_v_bottom").columns(2, gap="small")
    with c1.container(border=True):
        viewport(a, "sagittal", "v_sagittal", VIEW_H)
    with c2.container(border=True):
        info_panel(a)


def info_panel(a: dict) -> None:
    v, mask, labels, cur = a["vol"], a["mask"], a["labels"], a["cur"]
    hu = int(v["hu"][cur["i"], cur["j"], cur["k"]])
    L = int(mask["data"][cur["i"], cur["j"], cur["k"]]) if mask else 0
    sx, sy, sz = v["spacing"]
    panel_head("Study", h(v["name"]))
    rows = [("Cursor (i, j, k)", f"{cur['i']}, {cur['j']}, {cur['k']}"),
            ("Position", f"{cur['i'] * sx:.1f}, {cur['j'] * sy:.1f}, {cur['k'] * sz:.1f} mm"),
            ("HU at cursor", f"{hu}"), ("Label at cursor", f"{L} · {h(labels[L]['name'])}" if L in labels else "none"),
            ("L3", f"axial slice {a['l3'] + 1}" if a["l3"] is not None else "not set"),
            ("Mask", f"{h(mask['name'])} ({mask['source']})" if mask else "none")]
    md("<dl class='kv'>" + "".join(f"<dt>{k}</dt><dd>{val}</dd>" for k, val in rows) + "</dl>")
    md(f"<h4 class='sub'>Measurements ({len(ss.meas)})</h4>")
    if ss.pending:
        md(f"<p class='note'>Caliper: first point set on {ss.pending['view']} slice {ss.pending['slice'] + 1}. Click the second point.</p>")
    if not ss.meas:
        md("<p class='note'>Choose Tool → Caliper in the sidebar, then click two points on one slice.</p>")
    for m in ss.meas:
        c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="center")
        c1.markdown(f"<span class='mono'>#{m['id']} {m['view']} {m['slice'] + 1} · <b>{m['length_mm']:.1f} mm</b></span>",
                    unsafe_allow_html=True)
        c2.button("Go", key=f"go_{m['id']}", on_click=S.cb_go_meas, args=(m,))
        c3.button("✕", key=f"del_{m['id']}", on_click=S.cb_delete_meas, args=(m["id"],))


# ================================================================= Metrics table
def metrics(a: dict) -> None:
    v, mask, labels, stats, l3, cur = a["vol"], a["mask"], a["labels"], a["stats"], a["l3"], a["cur"]
    if not mask or not stats:
        md("<p class='empty'>Load a CT and a segmentation to see per-label metrics.</p>")
        return
    sp = v["spacing"]
    px = sp[0] * sp[1] / 100
    l3s = an.slice_stats(v["hu"], mask["data"], l3) if l3 is not None else None
    cs = an.slice_stats(v["hu"], mask["data"], cur["k"])
    with st.container(border=True):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        with c1:
            panel_head("Per-label metrics", f"Volume and attenuation over the whole mask; areas on the current slice ({cur['k'] + 1})"
                                            f" and at L3{f' ({l3 + 1})' if l3 is not None else ''}")
        c2.download_button("Export CSV", csv_text(a), file_name=f"{ss.ref}_metrics.csv", mime="text/csv", key="mt_csv",
                           on_click="ignore", width="stretch")
        rows = []
        for L, d in labels.items():
            lv = an.label_volume(stats, L, sp)
            n3 = l3s["n"][L] if l3s else 0
            rows.append({"Label": f"{L} · {d['name']}", "Role": d["role"], "Voxels": lv["voxels"], "Volume cm³": lv["cm3"],
                         "Mean HU": lv["mean_hu"], "SD HU": lv["sd_hu"], f"Slice {cur['k'] + 1} cm²": cs["n"][L] * px,
                         "L3 cm²": n3 * px if l3s else None, "L3 mean HU": l3s["sum"][L] / n3 if l3s and n3 else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                    ["Volume cm³", "Mean HU", "SD HU", f"Slice {cur['k'] + 1} cm²", "L3 cm²", "L3 mean HU"]})
        md(f"<p class='disc'>Voxel spacing {sp[0]:.3f} × {sp[1]:.3f} × {sp[2]:.3f} mm. Segmented slices "
           f"{stats['k_min'] + 1}–{stats['k_max'] + 1} of {v['shape'][2]}.</p>")

    body = a["body"]
    if body and (body.get("metrics_2d") or body.get("metrics_3d")):
        with st.container(border=True):
            s = body["slices"]
            panel_head("predict_muscle_fat.py metrics",
                       "Same formulas as the Mazurowski repo: 2D at the L3 peak slice, 3D between the T12 and L4 peak slices "
                       f"(T12 {s.get('T12', 0) + 1 if s.get('T12') is not None else '–'}, L3 "
                       f"{s['L3'] + 1 if s.get('L3') is not None else '–'}, L4 {s['L4'] + 1 if s.get('L4') is not None else '–'}). "
                       "Body = voxels above −500 HU.")
            c1, c2 = st.columns(2)
            for col, kind, title in ((c1, "metrics_2d", "2D at L3"), (c2, "metrics_3d", "3D T12–L4")):
                m = body.get(kind)
                col.markdown(f"**{title}**")
                if not m:
                    col.caption("Not available (vertebra not in the scan).")
                    continue
                col.dataframe(pd.DataFrame([{"Metric": k, "Value": val if isinstance(val, (int, float)) else str(val)}
                                            for k, val in m.items() if k not in ("slice_index", "slice_range")]),
                              hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.2f")})

    with st.container(border=True):
        panel_head("Cross-sectional area by axial slice", "Click a point to jump the viewers to that slice")
        series = [{"name": n, "color": ROLE_CSS[r], "values": an.role_profile(stats, labels, sp, r)}
                  for n, r in (("Muscle", "muscle"), ("Visceral fat", "vat"), ("Subcutaneous fat", "sat"), ("Intermuscular fat", "imat"))]
        series = [s for s in series if s["values"].any()]
        if series:
            # Labels on opposite sides of their lines so "L3" and "current" stay readable when the slices are close
            marks = (([{"x": l3, "label": "L3", "color": "#7ee08a", "pos": "top left"}] if l3 is not None else [])
                     + [{"x": cur["k"], "label": "current", "pos": "top right"}])
            st.plotly_chart(render.line_chart(series, "cm²", "Axial slice", marks), key="mt_profile",
                            on_select=lambda: S.cb_pick_slice("mt_profile"), selection_mode="points",
                            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False}, theme=None)
            with st.expander("Show as table"):
                df = pd.DataFrame({"Slice": np.arange(1, v["shape"][2] + 1), "z mm": np.arange(v["shape"][2]) * sp[2],
                                   **{f"{s['name']} cm²": s["values"] for s in series}})
                st.dataframe(df, hide_index=True, width="stretch", height=320,
                             column_config={c: st.column_config.NumberColumn(format="%.1f") for c in df.columns if c != "Slice"})
        else:
            md("<p class='empty'>No muscle or fat labels in this mask to plot.</p>")


# ================================================================= AI report
SECTIONS = [("muscle", "Muscle characteristics"), ("fat", "Fat distribution"), ("metabolic", "Potential metabolic implications"),
            ("sarcopenia", "Sarcopenia-related observations")]


def cutoff_bar(value, cutoff, lo, hi, flag_when, unit) -> str:
    if value is None or not math.isfinite(value):
        return "<div class='cbar'></div>"
    pos = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))  # noqa: E731
    zone = ""
    if cutoff is not None:
        c = pos(cutoff)
        zone = (f"<span class='zone' style='left:0;width:{c:.1f}%'></span>" if flag_when == "below"
                else f"<span class='zone' style='left:{c:.1f}%;width:{100 - c:.1f}%'></span>")
        zone += f"<span class='cut' style='left:{c:.1f}%'></span>"
    return (f"<div class='cbar'>{zone}<span class='val' style='left:{pos(value):.1f}%'></span></div>"
            f"<div class='cbar-ax'><span>{lo}</span><span>{hi} {unit}</span></div>")


def risk_card(title, metric, value, unit, cutoff, flag, flag_when, yes, no, lo, hi, source) -> str:
    cls = "bad" if flag else "ok" if flag is False else ""
    sub = f"{metric} · cut-off {'<' if flag_when == 'below' else '>'} {cutoff}" if cutoff is not None else f"{metric} · needs sex, height, weight"
    return (f"<div class='riskcard {cls}'><div class='rc-head'><b>{title}</b>{pill(flag, yes, no)}</div>"
            f"<div class='rc-val'>{f1(value)} <small>{unit}</small></div><div class='rc-sub'>{sub}</div>"
            f"{cutoff_bar(value, cutoff, lo, hi, flag_when, unit)}{f'<div class=rc-src>{h(source)}</div>' if source else ''}</div>")


def report(a: dict) -> None:
    v, mask, labels, l3, p, rep, sig, ev = a["vol"], a["mask"], a["labels"], a["l3"], a["patient"], a["report"], a["signature"], a["ev"]
    stale = bool(rep) and rep["payloadKey"] != a["payload_key"]
    signed = bool(sig)

    c = st.columns([2, 1.3, 2, 3], vertical_alignment="center")
    c[0].button("Regenerate" if rep else ("Generate AI report" if llm_client.configured() else "Generate rule-based draft"),
                type="primary", on_click=S.cb_generate, disabled=not a["payload"] or signed, key="rp_gen", width="stretch")
    c[1].button("Done editing" if ss.editing else "Edit text", on_click=S.cb_toggle_edit, disabled=not rep or signed,
                key="rp_edit", width="stretch")
    c[2].download_button("Print / save PDF (HTML)", data=report_html(a) if rep else "", file_name=f"{ss.ref}_report.html",
                         mime="text/html", disabled=not rep, key="rp_dl", on_click="ignore", width="stretch",
                         help="Open the downloaded page and print it to PDF")
    badge = (f"<span class='pill ok'>Signed by {h(sig['reader'])}</span>" if signed
             else "<span class='pill na'>Draft · pending sign-off</span>" if rep else "")
    c[3].markdown(f"<div style='text-align:right'>{badge}</div>", unsafe_allow_html=True)

    run = a["run"]
    seg = ("server model" if mask and mask["source"] == "inference" else mask["source"]) if mask else "none"
    seg_sub = f"run {run['id']} · {run['wall_s']}s · {run['model']}" if run else (mask["name"] if mask else "")
    narr = (rep["model"] if rep["source"] == "llm" else "Rule-based") if rep else "–"
    narr_sub = (f"temperature {rep['temperature']} · {rep['latencyMs'] / 1000:.1f} s{' · after retry' if rep.get('attempts', 1) > 1 else ''}"
                if rep and rep["source"] == "llm" else "model failed · fallback" if rep and rep.get("fallbackReason")
                else "no language model" if rep else "")
    state = "Signed" if signed else "Out of date" if stale else ("Draft · edited" if rep["edited"] else "Draft") if rep else "Not started"
    md("<section class='prov'>"
       f"<div><span class='k'>Case</span><b>{h(ss.ref or '–')}</b><small>{'synthetic sample' if v['sample'] else 'identifier hashed'}</small></div>"
       f"<div><span class='k'>Segmentation</span><b>{h(seg)}</b><small>{h(seg_sub)}</small></div>"
       f"<div><span class='k'>Narrative</span><b>{h(narr)}</b><small>{h(narr_sub)}</small></div>"
       f"<div><span class='k'>Prompt SHA-256</span><b class='mono'>{(rep or {}).get('promptSha256', '')[:12] + '…' if rep and rep.get('promptSha256') else '–'}</b>"
       f"<small>{rep['generatedAt'][:19].replace('T', ' ') if rep else ''}</small></div>"
       f"<div><span class='k'>Status</span><b>{state}</b><small>{sig['signedAt'][:19].replace('T', ' ') if signed else ''}</small></div>"
       "</section>")

    if ev:
        md("<section class='risk'>"
           + risk_card("Low muscle mass", "Skeletal muscle index", ev["smi"], "cm²/m²", ev["smi_cut"], ev["low_muscle"], "below",
                       "Below cut-off", "Above cut-off", 20, 80, (an.ref_smi(p) or {}).get("source"))
           + risk_card("Myosteatosis", "Mean muscle attenuation", ev["muscle_hu"], "HU", ev["hu_cut"], ev["myosteatosis"], "below",
                       "Below cut-off", "Above cut-off", 0, 70, (an.ref_smra(p) or {}).get("source"))
           + risk_card("Visceral obesity", "Visceral fat area", ev["vat"], "cm²", ev["vat_cut"], ev["visceral_obesity"], "above",
                       "Above cut-off", "Below cut-off", 0, 350, (an.ref_vat_area(p) or {}).get("source"))
           + "</section>")
    else:
        md("<p class='banner'>Load a segmentation and set the L3 slice to compute the indices.</p>")

    main, side = st.columns([3, 2], gap="medium")
    with main:
        if not rep:
            with st.container(border=True):
                md(f"<p class='empty'>{f'Generate a draft with {h(config.LLM_MODEL)}. Only the de-identified numbers shown on the Transmitted JSON tab are sent.' if llm_client.configured() else 'No language model is configured (LLM_URL). A rule-based draft can be written from the measured values instead.'}</p>")
        else:
            n = rep["narrative"]
            if rep.get("fallbackReason"):
                md(f"<p class='banner warn'>{h(config.LLM_MODEL)} did not return a usable draft, so this is the rule-based draft instead. "
                   f"Reason: <span class='mono'>{h(rep['fallbackReason'][:240])}</span>. Choose Regenerate to try the model again.</p>")
            if stale and not signed:
                md("<p class='banner warn'>The metrics changed after this draft was written. Regenerate before signing.</p>")
            if rep["unverifiedNumbers"]:
                md(f"<p class='banner warn'>These numbers in the draft do not match any measured value: "
                   f"<b class='mono'>{h(', '.join(rep['unverifiedNumbers']))}</b>. Check or remove them before signing.</p>")
            section(n, "summary", "Body composition summary")
            pl, d = a["payload"] or {}, rep.get("derived") or {}
            cm3 = lambda k: f1(pl[k] / 1000) if pl.get(k) is not None else "–"  # noqa: E731
            facts_of = {
                "muscle": (f"<div class='facts'><span>Volume <b>{cm3('muscle_volume_mm3')} cm³</b></span>"
                           f"<span>Density <b>{f1(pl.get('muscle_density_hu'))} HU</b></span>"
                           f"<span>Muscle index <b>{f1(d.get('muscle_volume_index_cm2'))} cm²</b></span></div>"),
                "fat": (f"<div class='facts'><span>SAT <b>{cm3('sfat_volume_mm3')} cm³</b></span><span>VAT <b>{cm3('vfat_volume_mm3')} cm³</b></span>"
                        f"<span>IMAT <b>{cm3('mfat_volume_mm3')} cm³</b></span>"
                        f"<span>VAT/SAT <b>{'–' if d.get('vat_sat_ratio') is None else d['vat_sat_ratio']}</b></span></div>"),
            } if pl else {}
            for i, (key, title) in enumerate(SECTIONS, 1):
                section(n, key, f"{i}. {title}", facts_of.get(key, ""))
            with st.container(border=True):
                md(f"<h3 class='fh'>{len(SECTIONS) + 1}. Lifestyle considerations</h3>")
                if ss.editing:
                    st.text_area("Lifestyle considerations (one per line)", key="ed_lifestyle", height=130,
                                 on_change=S.cb_edit, args=("lifestyle",), label_visibility="collapsed")
                else:
                    md("<ul>" + "".join(f"<li>{h(r)}</li>" for r in n["lifestyle"]) + "</ul>" if n["lifestyle"] else "<p>None.</p>")
            section(n, "limitations", "Limitations")
            if signed and ss.get("signed_remarks"):
                with st.container(border=True):
                    md(f"<h3 class='fh'>Reader remarks</h3><p>{h(ss.signed_remarks)}</p>")

    with side:
        with st.container(border=True):
            panel_head("Key anatomical evidence", f"L3 · axial slice {l3 + 1} of {v['shape'][2]}" if l3 is not None else "L3 not set")
            if l3 is not None:
                st.image(l3_snapshot(a), caption=f"W {a['wl']['w']} / L {a['wl']['l']} · {'mask overlay' if mask else 'no mask'}",
                         width="stretch")
                if mask:
                    l3s = an.slice_stats(v["hu"], mask["data"], l3)
                    sp = v["spacing"]
                    md("<ul class='ev-list'>" + "".join(
                        f"<li><i style='background:{lb.rgb(d['color'])}'></i>{h(d['name'])}<b>{f1(l3s['n'][L] * sp[0] * sp[1] / 100)} cm²</b>"
                        f"<em>{mean_hu(l3s, L)} HU</em></li>"
                        for L, d in labels.items()) + "</ul>")
        with st.container(border=True):
            panel_head("Physician attestation", "Signed and stored on the server" if signed else "Review the draft, then sign")
            if signed:
                md("<dl class='kv sig'>"
                   f"<dt>Reader</dt><dd>{h(sig['reader'])}</dd><dt>Login</dt><dd>{h(sig['authenticatedUser'] or 'not provided by proxy')}</dd>"
                   f"<dt>Signed</dt><dd>{sig['signedAt'][:19].replace('T', ' ')} UTC</dd><dt>SHA-256</dt><dd class='mono'>{sig['sha256']}</dd>"
                   f"<dt>Ed25519</dt><dd class='mono'>{sig['signature'][:44]}…</dd><dt>Key</dt><dd class='mono'>{sig['keyFingerprint']}</dd>"
                   f"<dt>Stored as</dt><dd class='mono'>{h(sig['file'])}</dd></dl>")
            else:
                st.text_input("Reading physician", key="reader", placeholder="Full name and credentials")
                st.text_area("Remarks or corrections", key="remarks", height=100, placeholder="Optional addendum stored with the signed report")
                if v["sample"]:
                    md("<p class='note warn'>The synthetic sample cannot be signed.</p>")
                reader = (ss.get("reader") or "").strip()
                if ss.confirming:
                    md(f"<div class='confirm'>Sign this report as <b>{h(reader)}</b>? It cannot be changed afterwards.</div>")
                    b1, b2 = st.columns(2)
                    b1.button("Confirm and sign", type="primary", on_click=S.cb_sign, args=(a["payload"],), width="stretch")
                    b2.button("Cancel", on_click=lambda: ss.update(confirming=False), width="stretch")
                else:
                    st.button("Sign and approve", type="primary", on_click=lambda: ss.update(confirming=True), width="stretch",
                              disabled=not rep or stale or not reader or v["sample"])
                if ss.sign_error:
                    md(f"<p class='status err'>{h(ss.sign_error)}</p>")
                md(f"<p class='note'>The metrics, narrative, model provenance and your remarks are signed with an Ed25519 key "
                   f"and stored in <span class='mono'>{h(config.DATA_DIR / 'reports')}</span>.</p>")
        md("<p class='disc'>Research and decision-support use. Not a cleared medical device. Automated measurements and generated "
           "text require review by a qualified reader before clinical use.</p>")


def mean_hu(st_: dict, L: int) -> str:
    return f"{st_['sum'][L] / st_['n'][L]:.0f}" if st_["n"][L] else "–"


def section(n: dict, key: str, title: str, facts: str = "") -> None:
    with st.container(border=True):
        md(f"<h3 class='fh'>{title}</h3>")
        if ss.editing:
            st.text_area(title, key=f"ed_{key}", height=110, on_change=S.cb_edit, args=(key,), label_visibility="collapsed")
        else:
            md(f"<p>{h(n[key] or 'Not stated.')}</p>")
        if facts:
            md(facts)


def l3_snapshot(a: dict) -> np.ndarray:
    v = a["vol"]
    cur = {**a["cur"], "k": a["l3"]}
    mask = a["mask"]["data"] if a["mask"] else None
    g = render.view_geom(v["shape"], v["spacing"], "axial")
    img = render.compose(render.view_slice(v["hu"], "axial", cur), None if mask is None else render.view_slice(mask, "axial", cur),
                         a["labels"], a["wl"], a["opacity"])
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((g["w"], g["h"]), Image.BILINEAR))


def report_html(a: dict) -> str:
    """Standalone printable report (the reference app's print view)."""
    rep, sig, pl, v = a["report"], a["signature"], a["payload"], a["vol"]
    n = rep["narrative"]
    snap = render.png_uri(l3_snapshot(a)) if a["l3"] is not None else ""
    rows = "".join(f"<tr><td>{r['k']}</td><td>{r['v']} {r['u']}</td><td>{r['ref']}</td><td>"
                   f"{'–' if r['flag'] is None else (r['yes'] if r['flag'] else r['no'])}</td></tr>" for r in indices(a))
    secs = "".join(f"<h2>{i}. {t}</h2><p>{h(n[k] or 'Not stated.')}</p>" for i, (k, t) in enumerate(SECTIONS, 1))
    recs = "".join(f"<li>{h(r)}</li>" for r in n["lifestyle"]) or "<li>None.</li>"
    bio = "".join(f"<tr><td>{h(biomarkers.FIELDS[k])}</td><td><code>{k}</code></td><td>{'–' if x is None else f'{x:,}'}</td></tr>"
                  for k, x in pl.items())
    rng = a["bio_range"] or {}
    signed = (f"<h2>Physician attestation</h2><p>Signed by <b>{h(sig['reader'])}</b> at {sig['signedAt']}<br>"
              f"SHA-256 <code>{sig['sha256']}</code><br>Ed25519 key {sig['keyFingerprint']} · stored as {h(sig['file'])}</p>"
              + (f"<p><b>Remarks:</b> {h(ss.get('signed_remarks', ''))}</p>" if ss.get("signed_remarks") else "")
              if sig else "<p class='draft'>DRAFT — not signed</p>")
    prov = (f"{rep['model']} (temperature {rep['temperature']})" if rep["source"] == "llm" else "rule-based draft")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>BodyComp report {h(ss.ref)}</title>
<style>body{{font:14px/1.5 system-ui,Segoe UI,sans-serif;max-width:820px;margin:24px auto;color:#111;padding:0 16px}}
h1{{font-size:20px;margin:0}}h2{{font-size:15px;margin:18px 0 4px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #eee;padding:4px 6px;text-align:left}}.meta{{color:#555}}.draft{{color:#b00;font-weight:600}}
img{{max-width:320px;float:right;margin:0 0 8px 12px;border:1px solid #ccc}}.disc{{color:#666;font-size:12px;margin-top:24px}}
@media print{{body{{margin:0}}}}</style></head><body>
<h1>CT body composition report</h1>
<p class="meta">Case {h(ss.ref)} · {v['shape'][2]} axial slices · biomarkers over {h(rng.get('source', '–'))} · L3 slice {a['l3'] + 1 if a['l3'] is not None else '–'} ·
narrative: {h(prov)} · generated {rep['generatedAt'][:19].replace('T', ' ')}{' · synthetic sample' if v['sample'] else ''}</p>
{f'<img src="{snap}" alt="L3 axial slice">' if snap else ''}
<h2>Body composition summary</h2><p>{h(n['summary'])}</p>{secs}
<h2>{len(SECTIONS) + 1}. Lifestyle considerations</h2><ul>{recs}</ul>
<h2>Limitations</h2><p>{h(n['limitations'])}</p>
<h2>Transmitted biomarkers</h2><table><tr><th>Biomarker</th><th>Field</th><th>Value</th></tr>{bio}</table>
<h2>L3 indices</h2><table><tr><th>Index</th><th>Value</th><th>Reference</th><th>Status</th></tr>{rows}</table>
{signed}
<p class="disc">Research and decision-support use. Not a cleared medical device. Automated measurements and generated text
require review by a qualified reader before clinical use. Printed {datetime.now():%Y-%m-%d %H:%M}.</p>
<script>window.addEventListener('load',()=>setTimeout(()=>window.print(),300))</script></body></html>"""


# ================================================================= Transmitted JSON
def block(title: str, sub: str, value, file: str, key: str) -> None:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    with st.container(border=True):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            panel_head(title, sub)
        c2.download_button("Download", text, file_name=file, mime="application/json", key=key, on_click="ignore", width="stretch")
        st.code(text, language="json" if not isinstance(value, str) else None)


def transmitted(a: dict) -> None:
    pl, rep = a["payload"], a["report"]
    md("<p class='banner'>This is everything that leaves the app for report generation: the eight whitelisted "
       "body-composition biomarkers. CT images, masks, file names, patient details and identifiers are never included.</p>")
    if not pl:
        md("<p class='empty'>No payload yet. Load a segmentation first.</p>")
        return
    sent = bool(rep) and rep["source"] == "llm"
    bad = biomarkers.violations(pl)
    rng = a["bio_range"] or {}
    where = h(rng.get("source", "–")) + (f" (axial slices {rng['slices'][0] + 1}–{rng['slices'][1] + 1})" if rng.get("slices") else "")
    md(f"<p class='banner warn'>Whitelist check failed, the payload would be blocked: {h('; '.join(bad))}</p>" if bad else
       f"<p class='status'>Whitelist check passed: exactly the {len(biomarkers.FIELDS)} allowed fields, all numeric · "
       f"{len(json.dumps(pl).encode())} bytes · measured over {where}</p>")
    block("Biomarker payload (as sent)" if sent else "Biomarker payload",
          f"Sent to {rep['model']} at {rep['generatedAt'][:19].replace('T', ' ')}" if sent else
          ("Will be sent when you generate a report" if llm_client.configured() else "No language model configured; nothing is sent"),
          pl, f"{ss.ref}_payload.json", "js_pl")
    block("Derived indices (added in the prompt)", "Computed from the eight values only, so they add no information about the patient",
          (rep or {}).get("derived") or biomarkers.derived(pl), f"{ss.ref}_derived.json", "js_der")
    if rep and rep.get("request"):
        block("Language model request", f"OpenAI-compatible chat completion · prompt SHA-256 {rep['promptSha256']}", rep["request"],
              f"{ss.ref}_llm_request.json", "js_req")
    if rep and rep.get("rawResponse"):
        block("Raw model response", f"{rep['latencyMs']} ms", rep["rawResponse"], f"{ss.ref}_llm_response.txt", "js_raw")
    if a["signature"]:
        block("Signature receipt", "Returned after signing", a["signature"], f"{ss.ref}_signature.json", "js_sig")


# ================================================================= Run log
LEVELS = ["ALL", "DEBUG", "INFO", "WARN", "ERROR"]
RANK = {"DEBUG": 0, "INFO": 1, "SUCCESS": 1, "WARN": 2, "ERROR": 3}


def runlog(a: dict) -> None:
    run, v = a["run"], a["vol"]
    with st.container(border=True):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            panel_head("Segmentation run", f"{h(run['filename'])} · {run['model']} on {h(run['device'])} · started {run['started']}"
                       if run else "No server run yet. Open a CT and choose Run segmentation.")
        if run:
            c2.button("Refresh", key="rl_refresh", width="stretch",
                      on_click=lambda: ss.update(run=S.seg_client.run_record(run["id"]) or ss.run))
            samples = run.get("samples") or []

            def peak(k):
                vals = [s[k] for s in samples if s.get(k) is not None]
                return max(vals) if vals else None

            mem_total = next((s["mem_total_mb"] for s in samples if s.get("mem_total_mb")), None)
            infer = run["stages"].get("inference")
            has_gpu = any(s.get("gpu_util") is not None for s in samples)
            tput = f"{v['shape'][2] / infer:.2f}" if infer else "–"
            pm, pg, pc, pt, pw = peak("mem_used_mb"), peak("gpu_util"), peak("cpu_util"), peak("temp_c"), peak("power_w")
            md("<section class='kpis small'>"
               f"<div class='kpi'><span class='k'>Total wall time</span><span class='v'>{run['wall_s']:.1f}<small>s</small></span>"
               f"<span class='s'>{' · '.join(f'{k} {val}s' for k, val in run['stages'].items()) or '–'}</span></div>"
               f"<div class='kpi'><span class='k'>Peak memory</span><span class='v'>{f'{pm / 1024:.1f}' if pm else '–'}"
               f"<small>{f'/ {mem_total / 1024:.1f} GB' if mem_total else 'GB'}</small></span><span class='s'>sampler: {run['sampler']}</span></div>"
               f"<div class='kpi'><span class='k'>Throughput</span><span class='v'>{tput}"
               f"<small>slices/s</small></span><span class='s'>{v['shape'][2]} slices{f' in {infer}s inference' if infer else ''}</span></div>"
               f"<div class='kpi'><span class='k'>{'Peak GPU / temperature' if has_gpu else 'Peak CPU'}</span>"
               f"<span class='v'>{f'{pg:.0f}' if has_gpu and pg is not None else f'{pc:.0f}' if pc is not None else '–'}%"
               f"<small>{f'{pt:.0f} °C' if pt is not None else ''}</small></span>"
               f"<span class='s'>{f'peak {pw:.1f} W' if pw is not None else 'power not reported'} · status {run['status']}</span></div>"
               "</section>")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 1.3, 1.2, 1.2], vertical_alignment="center")
        level = c1.segmented_control("Minimum level", LEVELS, default="ALL", key="rl_level", label_visibility="collapsed") or "ALL"
        server = [] if ss.hide_server or not run else [{**l, "origin": "server"} for l in run["logs"]]
        lines = sorted((l for l in server + ss.weblog if level == "ALL" or RANK.get(l["level"], 1) >= RANK[level]), key=lambda l: l["t"])
        text = "\n".join(f"[{l['t']}] [{l['origin']}] [{l['level']}] [{l['tag']}] {l['msg']}" for l in lines) + "\n"
        c2.download_button("Export .log", text, file_name=f"{run['id'] if run else 'session'}_bodycomp.log", key="rl_dl",
                           on_click="ignore", disabled=not lines, width="stretch")
        c3.button("Clear view", key="rl_clear", on_click=lambda: ss.update(weblog=[], hide_server=True), width="stretch")
        c4.button("Show server", key="rl_show", on_click=lambda: ss.update(hide_server=False), disabled=not ss.hide_server,
                  width="stretch")
        body = "".join(f"<div class='tl lv-{l['level'].lower()}'><span class='tt'>{l['t'][11:23]}</span>"
                       f"<span class='to'>{'srv' if l['origin'] == 'server' else 'web'}</span><span class='tv'>{l['level']}</span>"
                       f"<span class='tg'>{h(l['tag'])}</span><span class='tm'>{h(l['msg'])}</span></div>"
                       for l in reversed(lines[-1500:]))
        md(f"<div class='term'>{body or '<div class=term-empty>No log lines.</div>'}</div>")

    if run and run.get("samples"):
        with st.container(border=True):
            panel_head("Resource profile", f"Sampled every 0.5 s during the run ({run['sampler']}). Dot marks the peak.")
            samples = run["samples"]
            t = [s["t"] for s in samples]
            has_gpu = any(s.get("gpu_util") is not None for s in samples)
            mem_total = next((s["mem_total_mb"] for s in samples if s.get("mem_total_mb")), None)
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Memory used (MB)**")
            c1.plotly_chart(render.area_chart(t, [s.get("mem_used_mb") for s in samples], "MB", ROLE_CSS["vat"], mem_total),
                            key="rl_mem", config={"displayModeBar": False}, theme=None)
            c2.markdown(f"**{'GPU' if has_gpu else 'CPU'} utilisation (%)**")
            c2.plotly_chart(render.area_chart(t, [s.get("gpu_util" if has_gpu else "cpu_util") for s in samples], "%", ROLE_CSS["muscle"], 100),
                            key="rl_util", config={"displayModeBar": False}, theme=None)
            c3.markdown("**Power (W)**")
            c3.plotly_chart(render.area_chart(t, [s.get("power_w") for s in samples], "W", ROLE_CSS["imat"]),
                            key="rl_pow", config={"displayModeBar": False}, theme=None)
