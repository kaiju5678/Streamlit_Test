"""Frontend settings. Every value comes from an environment variable or from `.env` next to this file
(same names as ct-scan-test/.env.example)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _path(name: str, default: str = "") -> Path | None:
    value = os.getenv(name, default).strip()
    if not value:
        return None
    p = Path(value)
    return (p if p.is_absolute() else BASE_DIR / p).resolve()


# ---- Segmentation server (inference/server.py). Empty disables "Run segmentation".
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000/segment").strip()
INFERENCE_BASE = INFERENCE_URL.rstrip("/").removesuffix("/segment") if INFERENCE_URL else ""
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
INFERENCE_SCHEME = os.getenv("INFERENCE_SCHEME", "generic")
INFERENCE_TIMEOUT_S = int(os.getenv("INFERENCE_TIMEOUT_S", "3600"))

# ---- Language model: any OpenAI-compatible /v1/chat/completions server. Empty = rule-based drafts only.
LLM_URL = os.getenv("LLM_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "medgemma-27b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_MS", "120000")) / 1000
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3000"))

# ---- Folders
DATA_DIR = _path("DATA_DIR", "./data")            # signed reports + signing key
SIGNING_KEY_PEM = os.getenv("SIGNING_KEY_PEM", "").replace("\\n", "\n")
# A mask opened without its CT: the CT is looked up here (matching NIfTI header), with vertebrae_* of the same run.
SEG_REPO = _path("SEG_REPO", "../CT-Muscle-and-Fat-Segmentation")
CT_SEARCH_DIRS = [p for p in ([SEG_REPO / "demo", SEG_REPO, SEG_REPO / "results"] if SEG_REPO else [])
                  + [_path(s) for s in os.getenv("CT_SEARCH_DIRS", "").split(";") if s.strip()] if p]

# ---- UI
MESH_MAX_VOXELS = int(os.getenv("MESH_MAX_VOXELS", "1500000"))
