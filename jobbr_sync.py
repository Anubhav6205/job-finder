"""
Jobbr sync — push a Finder-generated resume into the local Jobbr app as a
new application record so it shows up on the kanban automatically.

Jobbr stores everything under the KV key `jobbr:v1` (a JSON blob with
apps[], statuses[], seq, resumes[]) and resume PDFs under separate
`jobbr:file:<id>:0` keys as base64 data URLs.

Design:
- Best-effort. If jobbr is not reachable (server down, port closed), we
  log and continue — the Finder resume build still succeeded.
- Never raise — caller does not care about jobbr failures.
"""
from __future__ import annotations

import base64
import json
import random
import string
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests

JOBBR_URL = "http://127.0.0.1:4581"
KV_KEY = "jobbr:v1"
FILE_KEY_FMT = "jobbr:file:{rid}:0"
TIMEOUT = 8  # jobbr is local, should be instant

# Site -> Found-via label jobbr expects (matches existing records)
SOURCE_MAP = {
    "linkedin": "LinkedIn",
    "indeed":   "Indeed",
    "naukri":   "Naukri",
}


def _random_suffix(n: int = 5) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _kv_get(key: str) -> Optional[str]:
    try:
        r = requests.get(f"{JOBBR_URL}/api/kv/get", params={"key": key}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        body = r.json()
        return body.get("value") if body.get("found") else None
    except Exception:
        return None


def _kv_set(key: str, value) -> bool:
    try:
        r = requests.post(
            f"{JOBBR_URL}/api/kv/set",
            json={"key": key, "value": value},
            timeout=TIMEOUT,
        )
        return r.status_code == 200
    except Exception:
        return False


def _health() -> bool:
    try:
        r = requests.get(f"{JOBBR_URL}/api/health", timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _build_scratchpad(score, fit, fit_reason, skill_gaps, salary_source,
                      model: str, source_url: str) -> str:
    lines = []
    if score is not None:
        lines.append(f"Fit: {score}/10 ({fit or '-'})")
    if fit_reason:
        lines.append("")
        lines.append(fit_reason.strip())
    if skill_gaps:
        lines.append("")
        lines.append(f"Gaps: {skill_gaps}")
    if salary_source:
        lines.append("")
        lines.append(f"Salary basis: {salary_source}")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()} via {model}")
    if source_url:
        lines.append(f"Source: {source_url}")
    return "\n".join(lines)


def push_application(
    *,
    company: str,
    role: str,
    source_url: str,
    site: str,
    city: str,
    pdf_path: Path,
    salary_max_lpa: Optional[float] = None,
    score: Optional[int] = None,
    fit: Optional[str] = None,
    fit_reason: Optional[str] = None,
    skill_gaps: Optional[str] = None,
    salary_source: Optional[str] = None,
    notion_page_id: Optional[str] = None,
    model: str = "Claude Opus",
    follow_up_days: int = 10,
) -> tuple[bool, str]:
    """Push one application + its PDF resume into Jobbr. Returns (ok, message)."""
    if not _health():
        return False, "jobbr not reachable at " + JOBBR_URL

    # 1. Fetch current data blob
    raw = _kv_get(KV_KEY)
    if raw is None:
        return False, "jobbr:v1 key missing (jobbr not yet initialized)"
    try:
        data = json.loads(raw)
    except Exception as e:
        return False, f"could not parse jobbr:v1 JSON: {e}"

    apps = data.setdefault("apps", [])
    resumes = data.setdefault("resumes", [])
    seq = int(data.get("seq", 0))

    # 2. Idempotency: if notion_page_id already present, skip
    if notion_page_id:
        for a in apps:
            if a.get("notionPageId") == notion_page_id:
                return True, f"already synced (app id={a.get('id')})"

    # 3. Upload the PDF as a file record
    resume_id = None
    if pdf_path and pdf_path.exists():
        seq += 1
        resume_id = f"r{seq}-{_random_suffix()}"
        try:
            b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
        except Exception as e:
            return False, f"pdf read failed: {e}"
        data_url = f"data:application/pdf;base64,{b64}"
        if not _kv_set(FILE_KEY_FMT.format(rid=resume_id), data_url):
            return False, "file upload POST failed"
        resumes.append({
            "id": resume_id,
            "label": pdf_path.stem,
            "filename": pdf_path.name,
            "mime": "application/pdf",
            "size": pdf_path.stat().st_size,
            "store": "b64",
            "chunks": 1,
            "addedAt": date.today().isoformat(),
        })

    # 4. Build the new app record
    today = date.today()
    seq += 1
    app_id = f"a{seq}-{_random_suffix()}"
    app = {
        "id": app_id,
        "company": company or "",
        "role": role or "",
        "url": source_url or "",
        "location": city or "",
        "workMode": "",
        "source": SOURCE_MAP.get((site or "").lower(), site or ""),
        "appliedDate": today.isoformat(),
        "status": "s1",
        "statusId": "s1",
        "resumeId": resume_id or "",
        "archived": False,
        "notes": _build_scratchpad(score, fit, fit_reason, skill_gaps,
                                    salary_source, model, source_url),
        "nextWhat": "Follow up with recruiter",
        "nextDate": (today + timedelta(days=follow_up_days)).isoformat(),
        "contacts": [],
        "interviews": [],
        "history": [{"date": today.isoformat(), "note": "auto-created by Finder"}],
        "expectedCtc": float(salary_max_lpa) if salary_max_lpa is not None else "",
        "offerBase": "",
        "offerVariable": "",
        "seniority": "",
        "notionPageId": notion_page_id or "",
    }
    apps.append(app)
    data["seq"] = seq

    # 5. Save back
    if not _kv_set(KV_KEY, json.dumps(data, ensure_ascii=False)):
        return False, "KV set for jobbr:v1 failed"
    return True, f"pushed app id={app_id} resume id={resume_id or '(none)'}"
