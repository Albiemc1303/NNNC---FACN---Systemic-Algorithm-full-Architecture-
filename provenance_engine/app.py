from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import hashlib
import json
import os
import asyncio
from pathlib import Path

API_TOKEN = os.getenv("PROVENANCE_API_TOKEN")
app = FastAPI(title="Provenance Engine v1.1")

class ProvenanceRequest(BaseModel):
    repo: str
    branch: str
    commit: str
    pr_number: int | None = None
    author: str
    timestamp: str
    files_changed: list[str]
    diff: str
    metadata: dict | None = None

# --- Internal: log decisions ---
def log_decision(request: ProvenanceRequest, status: str, reason: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "reason": reason,
        "commit": request.commit,
        "author": request.author,
        "files_changed": request.files_changed,
        "diff_hash": hashlib.sha256(request.diff.encode()).hexdigest(),
    }
    name = hashlib.sha256(f"{request.commit}{datetime.utcnow()}".encode()).hexdigest()[:16]
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{name}.json", "w") as f:
        json.dump(log_entry, f, indent=4)
    return name

# --- Async check protected paths ---
async def check_protected(file: str, protected_paths: list[str]):
    if any(file.startswith(p) for p in protected_paths):
        return f"Protected path modified: {file}"
    return None

@app.post("/provenance/check")
async def provenance_check(payload: ProvenanceRequest, authorization: str = Header(None)):
    # Verify API token
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    # Rule 1: Block direct pushes to main
    if payload.branch == "main" and payload.pr_number is None:
        log = log_decision(payload, "fail", "Direct pushes to main are forbidden.")
        return {"status": "fail", "reason": "Direct pushes to main are not allowed.", "log_id": log}

    # Rule 2: Mandatory diff content
    if len(payload.diff.strip()) == 0:
        log = log_decision(payload, "fail", "No diff provided.")
        return {"status": "fail", "reason": "Missing diff — provenance cannot be validated.", "log_id": log}

    # Rule 3: Parallel protected path checks
    protected_paths = ["cosmic_laws", "system_core"]
    tasks = [check_protected(f, protected_paths) for f in payload.files_changed]
    results = await asyncio.gather(*tasks)
    for r in results:
        if r:
            log = log_decision(payload, "fail", r)
            return {"status": "fail", "reason": r, "log_id": log}

    # Rule 4: Optional artifact hash verification
    if payload.metadata and "artifacts" in payload.metadata:
        for artifact in payload.metadata["artifacts"]:
            path = artifact["path"]
            expected_hash = artifact["hash"]
            if not Path(path).exists():
                log = log_decision(payload, "fail", f"Missing artifact: {path}")
                return {"status": "fail", "reason": f"Missing artifact: {path}", "log_id": log}
            actual_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                log = log_decision(payload, "fail", f"Hash mismatch for {path}")
                return {"status": "fail", "reason": f"Hash mismatch for {path}", "log_id": log}

    # All checks passed
    log = log_decision(payload, "pass", "All rules satisfied.")
    return {"status": "pass", "reason": "All provenance rules validated.", "log_id": log}
