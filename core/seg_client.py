"""Client for inference/server.py."""
from __future__ import annotations

import json
from urllib.parse import quote, unquote

import requests

import config
from core.http_log import HOOKS


def configured() -> bool:
    return bool(config.INFERENCE_BASE)


def _headers(extra: dict | None = None) -> dict:
    h = {"X-Api-Key": config.INFERENCE_API_KEY} if config.INFERENCE_API_KEY else {}
    return {**h, **(extra or {})}


def _hjson(resp, name: str):
    raw = resp.headers.get(name)
    try:
        return json.loads(unquote(raw)) if raw else None
    except ValueError:
        return None


def _error(resp) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"


def health() -> dict | None:
    if not configured():
        return None
    try:
        r = requests.get(f"{config.INFERENCE_BASE}/health", hooks=HOOKS, timeout=2)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def device() -> dict:
    if not configured():
        return {"configured": False}
    try:
        r = requests.get(f"{config.INFERENCE_BASE}/device", hooks=HOOKS, headers=_headers(), timeout=3)
        if not r.ok:
            return {"configured": True, "online": False, "error": _error(r)}
        return {"configured": True, "online": True, **r.json()}
    except requests.RequestException as e:
        return {"configured": True, "online": False, "error": type(e).__name__}


def run_record(run_id: str) -> dict | None:
    try:
        r = requests.get(f"{config.INFERENCE_BASE}/runs/{run_id}", hooks=HOOKS, headers=_headers(), timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def segment(ct_bytes: bytes, filename: str, run_id: str) -> dict:
    """Send CT bytes, receive the mask and its label names. Blocking (call from a thread)."""
    r = requests.post(f"{config.INFERENCE_BASE}/segment", hooks=HOOKS, data=ct_bytes, timeout=config.INFERENCE_TIMEOUT_S,
                      headers=_headers({"Content-Type": "application/octet-stream", "X-Filename": quote(filename),
                                        "X-Run-Id": run_id}))
    if not r.ok:
        raise RuntimeError(_error(r))
    return {
        "mask_bytes": r.content,
        "labels": _hjson(r, "X-Labels"),
        "scheme": r.headers.get("X-Label-Scheme") or config.INFERENCE_SCHEME,
        "body": _hjson(r, "X-Body-Composition"),
        "run_id": r.headers.get("X-Run-Id", run_id),
    }


def convert_dicom(zip_bytes: bytes) -> tuple[bytes, list, str]:
    r = requests.post(f"{config.INFERENCE_BASE}/convert", hooks=HOOKS, data=zip_bytes, timeout=900,
                      headers=_headers({"Content-Type": "application/zip"}))
    if not r.ok:
        raise RuntimeError(f"DICOM conversion failed: {_error(r)}")
    return r.content, _hjson(r, "X-Series") or [], r.headers.get("X-Series-Chosen", "")
