"""Draft reports from any OpenAI-compatible chat-completions server (port of ct-scan-test/app/api/report/route.ts).

Only the bodycomp.metrics/v1 payload is sent: no images, no file names, no identifiers.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone

import requests

import config
from core.analysis import unverified_numbers

SYSTEM = """You draft CT body-composition reports for a radiologist, who will review and sign them.
Rules:
- Use only the numbers in the JSON you are given. Never invent, estimate or convert values.
- If a value is null, say it is not available.
- A finding is abnormal only if its entry in "references" has flagged=true. Do not apply other cut-offs.
- Mention every item in "quality_flags" under limitations.
- Be concise and clinical. No patient identifiers exist in the data; do not make any up.
Reply with one JSON object and nothing else:
{"summary": string, "examination": string, "muscle": string, "adipose": string, "recommendations": string[], "limitations": string}"""

RETRY_NOTE = ("Your previous reply was not a valid JSON object. Reply again with only the JSON object described in the "
              "instructions: no markdown, no code fences, no other text.")


class LlmError(RuntimeError):
    pass


def configured() -> bool:
    return bool(config.LLM_URL)


def _post(body: dict) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    try:
        return requests.post(f"{config.LLM_URL}/chat/completions", json=body, headers=headers, timeout=(10, config.LLM_TIMEOUT_S))
    except requests.ConnectTimeout as e:
        raise LlmError(f"Cannot reach the language model at {config.LLM_URL} (connection timed out).") from e
    except requests.Timeout as e:
        raise LlmError("The language model did not answer in time.") from e
    except requests.RequestException as e:
        raise LlmError(f"Cannot reach the language model ({e}).") from e


def generate(payload: dict) -> dict:
    """Returns a GeneratedReport dict (source 'llm'). Raises LlmError when no usable draft comes back."""
    if not configured():
        raise LlmError("No language model configured. Set LLM_URL in .env.")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Body-composition measurements:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"},
    ]
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

    text = "\n".join([narrative["summary"], narrative["examination"], narrative["muscle"], narrative["adipose"],
                      *narrative["recommendations"], narrative["limitations"]])
    return {
        "source": "llm",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "narrative": narrative,
        "unverifiedNumbers": unverified_numbers(text, payload),
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

    recs = o.get("recommendations")
    n = {
        "summary": s(o.get("summary")), "examination": s(o.get("examination")),
        "muscle": s(o.get("muscle")), "adipose": s(o.get("adipose")),
        "recommendations": [x for x in map(s, recs) if x] if isinstance(recs, list) else ([s(recs)] if recs else []),
        "limitations": s(o.get("limitations")),
    }
    return n if n["summary"] or n["muscle"] or n["adipose"] else None
