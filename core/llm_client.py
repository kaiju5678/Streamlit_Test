"""Draft reports from MedGemma (or any OpenAI-compatible chat-completions server).

Only the eight whitelisted biomarkers are sent (core/biomarkers.py), and the whitelist check runs right before the
request: no images, no file names, no identifiers, no patient details. The derived indices in the prompt are computed
here from those eight values.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone

import requests

import config
from core import biomarkers
from core.http_log import HOOKS
from core.analysis import unverified_numbers

SYSTEM = """You are a medical research assistant. You write a structured, research-oriented report on CT body composition
for a physician, who will review and sign it.
The data are eight biomarkers measured on an abdominal CT segment, normally vertebrae T12 to L4 (skeletal muscle,
subcutaneous fat, visceral fat, intramuscular fat), plus indices derived from them. No patient identifiers, age, sex,
height or weight are available.
Task: assess the metabolic syndrome risk and the sarcopenia risk suggested by these measurements, and give lifestyle
guidance.
Rules:
- Use only the numbers you are given. Never invent, estimate or convert values. If a value is null, say it is not available.
- Quote volumes in cm³, using the "*_cm3" derived values exactly as given (they are the mm³ biomarkers already
  converted). Write numbers without thousands separators.
- "body_height_mm" is the craniocaudal height of the analysed T12-L4 segment, not the patient's height.
- "muscle_volume_index_cm2" is muscle volume divided by segment height (mean muscle cross-sectional area, unit cm²);
  it is not the L3 skeletal muscle index, so do not compare it with skeletal muscle index cut-offs.
- "vat_sat_ratio" is visceral fat volume divided by subcutaneous fat volume (VAT/SAT), not a fat-to-muscle ratio.
- Do not quote any threshold, cut-off or reference range except those inside the given data. There is no validated
  volume or percentage cut-off for sarcopenia in these data: describe muscle quantity without one, and base the
  sarcopenia observations on the myosteatosis indication and muscle volume index as given.
- Quote each percentage with the exact meaning given in its label.
- For myosteatosis use the given "myosteatosis_indication" only; do not apply other cut-offs.
- Without height, sex and reference ranges for volumes, describe risks as "suggested" or "possible", never as a diagnosis.
- This is decision-support information, not a definitive diagnosis.
Reply with one JSON object and nothing else:
{"summary": string, "muscle": string, "fat": string, "metabolic": string, "sarcopenia": string,
 "lifestyle": string[], "limitations": string}
summary = body composition summary; muscle = muscle characteristics; fat = fat distribution;
metabolic = potential metabolic implications (metabolic syndrome risk); sarcopenia = sarcopenia-related observations;
lifestyle = lifestyle considerations, one per item; limitations = limitations of this assessment."""

NARRATIVE_KEYS = ("summary", "muscle", "fat", "metabolic", "sarcopenia", "limitations")

RETRY_NOTE = ("Your previous reply was not a valid JSON object. Reply again with only the JSON object described in the "
              "instructions: no markdown, no code fences, no other text.")


class LlmError(RuntimeError):
    pass


def configured() -> bool:
    return bool(config.LLM_URL)


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    return headers


def reachable() -> str | None:
    """None when the server answers GET /models; otherwise why not (e.g. off the lab network)."""
    try:
        r = requests.get(f"{config.LLM_URL}/models", headers=_headers(), timeout=(2, 5))
    except requests.ConnectTimeout:
        return "no answer (connection timed out; the server may only be reachable from the lab network)"
    except requests.RequestException as e:
        return f"cannot connect ({type(e).__name__})"
    return None if r.ok else f"HTTP {r.status_code}"


def _post(body: dict) -> requests.Response:
    headers = _headers()
    try:
        return requests.post(f"{config.LLM_URL}/chat/completions", hooks=HOOKS, json=body, headers=headers, timeout=(10, config.LLM_TIMEOUT_S))
    except requests.ConnectTimeout as e:
        raise LlmError(f"Cannot reach the language model at {config.LLM_URL} (connection timed out).") from e
    except requests.Timeout as e:
        raise LlmError("The language model did not answer in time.") from e
    except requests.RequestException as e:
        raise LlmError(f"Cannot reach the language model ({e}).") from e


# Each derived index spelled out, so a small model does not mix up which percentage is a share of what.
DERIVED_LABELS = {
    "vat_sat_ratio": ("VAT/SAT ratio (visceral fat volume / subcutaneous fat volume)", ""),
    "visceral_share_of_fat_pct": ("Visceral fat as a share of total fat volume", "%"),
    "fat_share_of_body_pct": ("Total fat as a share of body volume", "%"),
    "muscle_share_of_body_pct": ("Skeletal muscle as a share of body volume", "%"),
    "intramuscular_fat_to_muscle_pct": ("Intramuscular fat as a share of skeletal muscle volume", "%"),
    "muscle_volume_index_cm2": ("Muscle volume index (muscle volume / segment height = mean muscle cross-sectional area)", " cm²"),
    "myosteatosis_indication": ("Myosteatosis indication", ""),
    "myosteatosis_reference": ("Myosteatosis reference", ""),
    "muscle_volume_cm3": ("Skeletal muscle volume", " cm³"),
    "sfat_volume_cm3": ("Subcutaneous fat volume", " cm³"),
    "vfat_volume_cm3": ("Visceral fat volume", " cm³"),
    "mfat_volume_cm3": ("Intramuscular fat volume", " cm³"),
    "total_fat_volume_cm3": ("Total fat volume", " cm³"),
    "body_volume_cm3": ("Body volume of the segment", " cm³"),
}


def user_message(payload: dict, derived: dict) -> str:
    lines = []
    for k, v in derived.items():
        label, unit = DERIVED_LABELS.get(k, (k, ""))
        lines.append(f"- {label} [{k}]: {'not available' if v is None else f'{v}{unit}'}")
    return ("Body-composition biomarkers:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
            + "\n\nDerived indices (computed from the eight values above):\n" + "\n".join(lines))


def generate(payload: dict) -> dict:
    """Returns a GeneratedReport dict (source 'llm'). Raises LlmError when no usable draft comes back, and
    biomarkers.PayloadBlocked when the payload fails the whitelist check (then nothing is sent)."""
    if not configured():
        raise LlmError("No language model configured. Set LLM_URL in .env.")
    biomarkers.check(payload)                     # nothing but the eight whitelisted numbers may leave the app
    derived = biomarkers.derived(payload)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(payload, derived)}]
    prompt_sha = hashlib.sha256(json.dumps(messages, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    started = time.time()
    state = {"json_mode": True}

    def call(msgs):
        body = {"model": config.LLM_MODEL, "messages": msgs, "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS}
        if state["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        r = _post(body)
        # Ask for JSON mode first; drop it if this server rejects the option.
        if state["json_mode"] and r.status_code in (400, 422):
            state["json_mode"] = False
            body.pop("response_format")
            r = _post(body)
        if not r.ok:
            raise LlmError(f"Language model returned HTTP {r.status_code}. {r.text[:400]}")
        j = r.json()
        return ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "", body

    raw, sent, narrative, attempts = "", {}, None, 0
    # First try, then one retry that shows the model its unreadable reply.
    for retry in (False, True):
        attempts += 1
        convo = messages if not retry else [*messages, {"role": "assistant", "content": raw[:4000]},
                                            {"role": "user", "content": RETRY_NOTE}]
        raw, sent = call(convo)
        narrative = parse_narrative(raw)
        if narrative:
            break
    if not narrative:
        raise LlmError(f"The model reply was not the expected JSON after {attempts} attempts. Reply started: {raw[:200]}")

    text = "\n".join([*(narrative[k] for k in NARRATIVE_KEYS), *narrative["lifestyle"]])
    return {
        "source": "llm",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "narrative": narrative,
        "unverifiedNumbers": unverified_numbers(text, {**payload, **derived}),
        "derived": derived,
        "model": config.LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "latencyMs": int((time.time() - started) * 1000),
        "promptSha256": prompt_sha,
        "attempts": attempts,
        "request": sent,
        "rawResponse": raw,
    }


def strip_thinking(raw: str) -> str:
    """Remove a reasoning section before the answer (MedGemma 1.5 <unused94>…<unused95>, or <think>…</think>).
    An unclosed section means the model ran out of tokens while thinking, so nothing after it is an answer."""
    s = re.sub(r"<unused94>[\s\S]*?<unused95>", "", raw)
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.I)
    m = re.search(r"<unused94>|<think>", s, flags=re.I)
    if m:
        s = s[:m.start()]
    return re.sub(r"<unused\d+>", "", s).strip()


def parse_narrative(reply: str) -> dict | None:
    """Accepts a bare object, a fenced ```json block, or an object embedded in prose."""
    raw = strip_thinking(reply)
    blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.I)
    src = blocks[-1] if blocks else raw  # the last fenced block is the final answer
    start, end = src.find("{"), src.rfind("}")
    if start < 0 or end <= start:
        return None
    chunk = src[start:end + 1]
    try:
        o = json.loads(chunk)
    except json.JSONDecodeError:
        try:  # common small-model slips: trailing commas, smart quotes
            o = json.loads(re.sub(r",\s*([}\]])", r"\1", chunk).replace("“", '"').replace("”", '"'))
        except json.JSONDecodeError:
            return None
    if not isinstance(o, dict):
        return None

    def s(v) -> str:
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        if isinstance(v, list):
            return " ".join(map(str, v))
        return str(v)

    recs = o.get("lifestyle")
    n = {k: s(o.get(k)) for k in NARRATIVE_KEYS}
    n["lifestyle"] = [x for x in map(s, recs) if x] if isinstance(recs, list) else ([s(recs)] if recs else [])
    return n if n["summary"] or n["muscle"] or n["fat"] else None
