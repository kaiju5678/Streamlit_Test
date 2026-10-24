"""One terminal line per HTTP call the frontend makes (segmentation server, language model)."""
from __future__ import annotations

import os
from datetime import datetime
from urllib.parse import urlsplit

if os.name == "nt":
    os.system("")  # turn on ANSI colours in cmd.exe

ENABLED = os.getenv("ACCESS_LOG", "1") == "1"


def status_text(code: int) -> str:
    """Status code coloured by class: 2xx green, 3xx cyan, 4xx yellow, 5xx red."""
    colour = {2: "32", 3: "36", 4: "33"}.get(code // 100, "31")
    return f"\033[{colour}m{code}\033[0m"


def _log(resp, *args, **kwargs):
    if ENABLED:
        u = urlsplit(resp.url)
        ms = resp.elapsed.total_seconds() * 1000
        print(f"{datetime.now():%H:%M:%S}  {status_text(resp.status_code)}  {resp.request.method:<4} {u.netloc}{u.path}  {ms:.0f} ms",
              flush=True)
    return resp


HOOKS = {"response": _log}
