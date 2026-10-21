"""Physician sign-off: Ed25519 signature over the canonical report record, stored in DATA_DIR/reports
(port of ct-scan-test/app/api/sign/route.ts and lib/server/signing.ts)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import config

_cached: dict | None = None


def canonical(v) -> str:
    """Deterministic JSON: keys sorted at every level, so a hash or signature is reproducible."""
    if v is None or not isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(canonical(x) for x in v) + "]"
    return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical(v[k]) for k in sorted(v)) + "}"


def signing_key() -> dict:
    """SIGNING_KEY_PEM (PKCS#8) wins; otherwise a key is created once in DATA_DIR/signing-key.pem."""
    global _cached
    if _cached:
        return _cached
    pem = config.SIGNING_KEY_PEM.encode() if config.SIGNING_KEY_PEM else None
    if pem is None:
        f = config.DATA_DIR / "signing-key.pem"
        if f.exists():
            pem = f.read_bytes()
        else:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            pem = Ed25519PrivateKey.generate().private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
            f.write_bytes(pem)
            try:
                os.chmod(f, 0o600)
            except OSError:
                pass
    priv = serialization.load_pem_private_key(pem, password=None)
    pub = priv.public_key()
    der = pub.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    pub_pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    _cached = {"priv": priv, "pub_pem": pub_pem, "fingerprint": hashlib.sha256(der).hexdigest()[:16]}
    return _cached


def sign(payload: dict, report: dict, reader: str, remarks: str, authenticated_user: str | None = None) -> dict:
    reader = (reader or "").strip()
    if not reader:
        raise ValueError("Enter the reading physician before signing.")
    if payload.get("schema") != "bodycomp.metrics/v1" or not report.get("narrative"):
        raise ValueError("Missing report or metrics.")
    if payload.get("synthetic_sample"):
        raise ValueError("The synthetic sample cannot be signed.")

    signed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    # What gets signed: metrics, narrative, its provenance, the reader's remarks and identity.
    record = {
        "schema": "bodycomp.signed-report/v1",
        "signed_at": signed_at,
        "reader": reader,
        "authenticated_user": authenticated_user,
        "decision": "amended" if report.get("edited") else "approved",
        "remarks": (remarks or "")[:5000],
        "metrics": payload,
        "narrative": report["narrative"],
        "provenance": {
            "source": report.get("source"),
            "model": report.get("model"),
            "prompt_sha256": report.get("promptSha256"),
            "generated_at": report.get("generatedAt"),
            "unverified_numbers": report.get("unverifiedNumbers", []),
        },
    }
    text = canonical(record)
    sha = hashlib.sha256(text.encode()).hexdigest()
    k = signing_key()
    signature = base64.b64encode(k["priv"].sign(text.encode())).decode()

    out_dir = config.DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    file = f"{payload['case_ref']}_{signed_at.replace(':', '-').replace('.', '-')}.json"
    (out_dir / file).write_text(json.dumps({"record": record, "sha256": sha, "signature": signature,
                                            "algorithm": "Ed25519", "key_fingerprint": k["fingerprint"]},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "report.signed", "case_ref": payload["case_ref"], "reader": reader, "sha256": sha, "file": file}),
          flush=True)
    return {"sha256": sha, "signature": signature, "algorithm": "Ed25519", "keyFingerprint": k["fingerprint"],
            "signedAt": signed_at, "reader": reader, "authenticatedUser": authenticated_user, "file": file}
