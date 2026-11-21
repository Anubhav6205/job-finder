"""
Resume daemon.

Polls the Notion Job Finder DB for rows whose Status = "Build Resume".
For each: fetches the JD, hands it to the Claude Code CLI with the
resume Knowledge Base + LaTeX reference template already in-repo, lets
Claude build the .tex and compile it to PDF via Tectonic, then updates
Notion with the path (or an error message on failure).

Intended to be launched by Windows Task Scheduler every 5 minutes.
Processes all pending rows sequentially in a single invocation, then exits.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

REPO = Path(__file__).resolve().parent
ASSETS = REPO / "resume_assets"
JOBS_CSV = REPO / "data" / "jobs.csv"
JD_CACHE = REPO / "data" / "jd_cache"
TECTONIC = REPO / "tools" / "tectonic.exe"
KB_FILE = ASSETS / "Resume_Builder_Knowledge_Base_v4.docx"
TPL_FILE = ASSETS / "Resume_Reference_Template.txt"

OUT_DIR = Path(r"C:/Users/anubh/Downloads/Resumes")
JD_SAVE_DIR = Path(r"C:/Users/anubh/Downloads/Job Descriptions")
WORK_DIR = REPO / "resume_build"       # scratch for .tex + tectonic output
LOG_FILE = REPO / "resume_daemon.log"

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DB_ID = os.getenv("NOTION_DB_ID")
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}
def _resolve_claude_cmd() -> str:
    override = os.getenv("CLAUDE_CMD")
    if override:
        return override
    # Windows: npm wrappers live in %APPDATA%/npm/ as .cmd
    for cand in [
        Path(os.getenv("APPDATA", "")) / "npm" / "claude.cmd",
        Path("C:/Users/anubh/AppData/Roaming/npm/claude.cmd"),
    ]:
        if cand.exists():
            return str(cand)
    return "claude"  # last resort; may fail on Windows subprocess


CLAUDE_CMD = _resolve_claude_cmd()

# JD completeness policy
JD_MIN_CHARS = 200          # below this the JD is a stub: fail clearly, never build from it
JD_THIN_CHARS = 600         # 200-599: build, but flag as thin (source-limited posting)
JD_PROMPT_CAP = 20000       # observed max cached JD is ~16k; never silently cut real JDs
JD_WAIT_LIMIT_S = 2 * 3600  # how long a row may wait for its JD to reach GitHub
GIT_FETCH_THROTTLE_S = 60
NO_WINDOW = 0x08000000      # CREATE_NO_WINDOW: no console flash when run via pythonw

_GIT_LOCK = threading.Lock()
_last_fetch_at = 0.0
_last_fetch_ok = False
_jd_wait_since: dict[str, float] = {}      # page_id -> first time its JD was missing
_jd_wait_notified: set[str] = set()

# Claude usage-limit pause (epoch seconds). While now < _PAUSE_UNTIL no Claude calls are made.
_PAUSE_UNTIL = 0.0
_pause_logged_for = 0.0
USAGE_PAUSE_CAP = timedelta(hours=6)


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    try:
        # pythonw.exe has no real stdout; wrap in try to stay non-fatal under both.
        print(line, flush=True)
    except Exception:
        pass
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def sanitize(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "-", (s or "").strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:maxlen] or "Unknown"


# ---------- Notion ----------
def query_skipped_with_jobbr() -> list[dict]:
    """Rows the user has moved to Skipped AFTER we already pushed a jobbr card.
    Identified by Status=Skipped AND a non-empty Resume Path.
    We clear Resume Path on cleanup so we never re-process the same row."""
    r = requests.post(
        f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query",
        headers=NOTION_HEADERS,
        json={
            "filter": {
                "and": [
                    {"property": "Status", "select": {"equals": "Skipped"}},
                    {"property": "Resume Path", "rich_text": {"is_not_empty": True}},
                ]
            },
            "page_size": 20,
        },
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for p in r.json().get("results", []):
        props = p["properties"]
        out.append({
            "page_id": p["id"],
            "role": (props.get("Role", {}).get("title") or [{}])[0].get("plain_text", ""),
            "company": (props.get("Company", {}).get("rich_text") or [{}])[0].get("plain_text", ""),
        })
    return out


def cleanup_skipped() -> None:
    """For each Skipped row that previously had a jobbr card, delete ONLY that
    one jobbr app + its attached resume. Then clear the row's Resume Path so
    we don't re-process next tick. Nothing else is touched."""
    try:
        rows = query_skipped_with_jobbr()
    except Exception as e:
        log(f"cleanup skipped-query failed: {e}")
        return
    if not rows:
        return
    log(f"cleanup: {len(rows)} skipped row(s) with jobbr card to delete")
    try:
        import jobbr_sync
    except Exception as e:
        log(f"cleanup: cannot import jobbr_sync: {e}")
        return
    for item in rows:
        page_id = item["page_id"]
        ok, msg = jobbr_sync.delete_by_notion_page_id(page_id)
        # Clear Resume Path regardless (prevents retry loop); append to Log
        try:
            stamp = datetime.now().isoformat(timespec="seconds")
            requests.patch(
                f"https://api.notion.com/v1/pages/{page_id}",
                headers=NOTION_HEADERS,
                json={"properties": {
                    "Resume Path": {"rich_text": [{"text": {"content": ""}}]},
                    "Resume Log":  {"rich_text": [{"text": {"content": f"jobbr card deleted @ {stamp} — {msg}"[:1900]}}]},
                }},
                timeout=30,
            )
        except Exception as e:
            log(f"  warn: could not clear path for {page_id}: {e}")
        log(f"  cleanup {item['company']}/{item['role'][:40]}: {msg}")


def query_pending() -> list[dict]:
    r = requests.post(
        f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query",
        headers=NOTION_HEADERS,
        json={
            "filter": {"property": "Status", "select": {"equals": "Build Resume"}},
            "page_size": 20,
        },
        timeout=30,
    )
    r.raise_for_status()
    pages = r.json().get("results", [])
    out = []
    for p in pages:
        props = p["properties"]
        out.append({
            "page_id": p["id"],
            "role": (props.get("Role", {}).get("title") or [{}])[0].get("plain_text", ""),
            "company": (props.get("Company", {}).get("rich_text") or [{}])[0].get("plain_text", ""),
            "url": props.get("URL", {}).get("url", ""),
            "site": (props.get("Site", {}).get("select") or {}).get("name", ""),
            "city": (props.get("City", {}).get("select") or {}).get("name", ""),
            "score": props.get("Score", {}).get("number"),
            "fit": (props.get("Fit", {}).get("select") or {}).get("name", ""),
            "fit_reason": (props.get("Fit Reason", {}).get("rich_text") or [{}])[0].get("plain_text", ""),
            "salary_max": props.get("Salary Max (LPA)", {}).get("number"),
            "salary_source": (props.get("Salary Source", {}).get("rich_text") or [{}])[0].get("plain_text", ""),
        })
    return out


def set_status(page_id: str, status: str, path: str = "", log_msg: str = "",
               gaps: str = "") -> None:
    props = {"Status": {"select": {"name": status}}}
    if path:
        props["Resume Path"] = {"rich_text": [{"text": {"content": path[:1900]}}]}
    if log_msg:
        props["Resume Log"] = {"rich_text": [{"text": {"content": log_msg[:1900]}}]}
    if gaps:
        props["Skill Gaps"] = {"rich_text": [{"text": {"content": gaps[:1900]}}]}
    r = requests.patch(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=NOTION_HEADERS,
        json={"properties": props},
        timeout=30,
    )
    if r.status_code >= 300:
        log(f"  notion update warn {r.status_code}: {r.text[:200]}")


# ---------- JD fetch ----------
def _cache_key(job_url: str) -> str:
    import hashlib
    return hashlib.md5(job_url.encode()).hexdigest()


def jd_from_cache(job_url: str) -> str | None:
    """JDs scraped by job_finder.py are cached here keyed by URL hash."""
    f = JD_CACHE / f"{_cache_key(job_url)}.txt"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="replace")
    return text if len(text) >= 200 else None


def jd_rescrape(job_url: str) -> str | None:
    """Last-resort: hit JobSpy with a broad search on India Indeed to pick this URL back up.
    Only works for Indeed URLs; LinkedIn/Naukri would need site-specific search terms."""
    try:
        from jobspy import scrape_jobs
        # Try multiple broad search terms; jobspy returns URL-keyed results
        for term in ["AI Engineer", "Software Engineer AI", "GenAI Engineer"]:
            df = scrape_jobs(
                site_name=["indeed"], search_term=term, location="India",
                results_wanted=50, hours_old=168, country_indeed="India", verbose=0,
            )
            hit = df[df["job_url"] == job_url]
            if not hit.empty:
                desc = str(hit.iloc[0].get("description", ""))
                if len(desc) >= 200:
                    JD_CACHE.mkdir(parents=True, exist_ok=True)
                    (JD_CACHE / f"{_cache_key(job_url)}.txt").write_text(desc, encoding="utf-8")
                    return desc
    except Exception as e:
        log(f"  rescrape err: {e}")
    return None


def _git_fetch_origin() -> bool:
    """One serialized, throttled `git fetch`. Returns whether the last fetch succeeded.
    Only updates remote-tracking refs; never touches the working tree."""
    global _last_fetch_at, _last_fetch_ok
    with _GIT_LOCK:
        if time.time() - _last_fetch_at < GIT_FETCH_THROTTLE_S:
            return _last_fetch_ok
        try:
            r = subprocess.run(
                ["git", "-C", str(REPO), "fetch", "-q", "origin", "master"],
                capture_output=True, timeout=90, creationflags=NO_WINDOW,
            )
            _last_fetch_ok = r.returncode == 0
            if not _last_fetch_ok:
                log(f"  git fetch failed: {(r.stderr or b'')[-200:].decode('utf-8', 'replace')}")
        except Exception as e:
            _last_fetch_ok = False
            log(f"  git fetch error: {e}")
        _last_fetch_at = time.time()
        return _last_fetch_ok


def jd_from_origin(job_url: str) -> tuple[str | None, str]:
    """Read the JD the GitHub Actions scrape committed for this URL.
    Returns (text, state) with state in ok | absent | fetch_failed.
    The blob is read as raw bytes and written back unchanged, so the local copy is
    byte-identical to what was scraped."""
    key = _cache_key(job_url)
    fetched = _git_fetch_origin()
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "show", f"origin/master:data/jd_cache/{key}.txt"],
            capture_output=True, timeout=30, creationflags=NO_WINDOW,
        )
    except Exception as e:
        log(f"  git show error: {e}")
        return None, "fetch_failed"
    if r.returncode != 0:
        # Fetch OK and file still missing = not committed yet. Fetch failed = unknown, retry later.
        return None, ("absent" if fetched else "fetch_failed")
    JD_CACHE.mkdir(parents=True, exist_ok=True)
    (JD_CACHE / f"{key}.txt").write_bytes(r.stdout)
    return r.stdout.decode("utf-8", errors="replace"), "ok"


def resolve_jd(url: str, page_id: str) -> tuple[str | None, str, str]:
    """Find the full JD and judge its completeness.
    Returns (jd, source, state):
      ok       >= 600 chars
      thin     200-599 chars (genuinely short posting; build, but flag it)
      short    < 200 chars (stub; never build from it)
      wait     not available yet; row stays at Build Resume and retries next tick
      missing  still not available after JD_WAIT_LIMIT_S
    Sources are tried in order of fidelity: local cache, GitHub, then JobSpy re-scrape.
    There is deliberately no raw page scraping: it can return unrelated page text."""
    text, source = None, ""
    local = JD_CACHE / f"{_cache_key(url)}.txt"
    if local.exists():
        text, source = local.read_text(encoding="utf-8", errors="replace"), "cache"
    if text is None:
        text, origin_state = jd_from_origin(url)
        if text is not None:
            source = "origin"
        elif origin_state == "fetch_failed":
            return None, "", "wait"          # offline/unknown: retry, don't start the failure clock
    if text is None and page_id not in _jd_wait_since:
        text = jd_rescrape(url)              # slow; only on the first miss for this row
        if text is not None:
            source = "rescrape"
    if text is None:
        since = _jd_wait_since.setdefault(page_id, time.time())
        return None, "", ("missing" if time.time() - since > JD_WAIT_LIMIT_S else "wait")
    _jd_wait_since.pop(page_id, None)
    _jd_wait_notified.discard(page_id)
    n = len(text)
    return text, source, ("short" if n < JD_MIN_CHARS else "thin" if n < JD_THIN_CHARS else "ok")


# ---------- Claude + Tectonic ----------
def build_prompt(company: str, role: str, jd: str, tex_path: Path, pdf_path: Path,
                 gaps_path: Path) -> str:
    return f"""You are the Resume Builder. Today's date: {date.today().isoformat()}.

Build a tailored resume for the following job opening. Follow these steps STRICTLY and in order.
Operate autonomously: do not ask the user questions; make reasonable judgment calls grounded in
the Knowledge Base. Treat this as an ISOLATED build — do not reference, compare to, or copy
from any file in the work directory other than the ones named below.

1. Read the Knowledge Base completely: `{KB_FILE}`.
   It is the single source of truth for facts, writing rules, skills, build steps and file naming.
   Section 2 writing rules rank directly below truthfulness and above keyword coverage.

2. Read the LaTeX reference template completely: `{TPL_FILE}`.
   Preserve its preamble, packages, macros, colors, spacing and block structure EXACTLY.
   Change only content, not structure.

3. Note the company and role (already resolved from the source Notion row):
   - Company: {company}
   - Role:    {role}

4. Produce the tailored LaTeX at EXACTLY this path:
   `{tex_path}`
   (Use the Write tool. Do not write the file anywhere else.)

5. Compile the .tex to a PDF at EXACTLY this path using Tectonic:
   `{pdf_path}`
   Run this command with the Bash tool:
   `"{TECTONIC}" -o "{pdf_path.parent}" "{tex_path}"`
   (Tectonic produces a PDF with the same stem as the .tex in -o.
   If the resulting PDF name differs from the target, rename it to `{pdf_path.name}`.)

6. Verify the PDF exists and is non-empty. If compilation fails, read the Tectonic
   error output, fix the .tex, and retry (up to 2 retries). Do not invent packages.

7. Write a ONE-LINE skill-gaps summary to EXACTLY this path:
   `{gaps_path}`
   Format: plain text, single line, under 200 characters, no leading bullet.
   Content: the top skill/experience gaps the JD requires that the candidate's
   Evidence Bank (from the Knowledge Base) does NOT support. Comma-separated
   terms, specific not generic. Example of a good line:
   `SageMaker pipelines; Terraform IaC; 5+ yrs ML Ops; healthcare domain`
   If there are no meaningful gaps, write: `none — strong coverage`.

8. Final answer: one line confirming the PDF path.

JOB DESCRIPTION:
---
{jd[:JD_PROMPT_CAP]}
---
"""


def run_claude(prompt: str) -> tuple[bool, str]:
    """Invoke Claude CLI non-interactively. Prompt goes via stdin so Windows .cmd
    wrappers don't truncate multi-line text at the first newline."""
    cmd = [
        CLAUDE_CMD, "-p",
        "--model", "opus",
        "--fallback-model", "sonnet",
        "--allowed-tools", "Read,Write,Edit,Bash,Glob,Grep",
    ]
    try:
        r = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, "claude CLI timed out after 10 min"
    except FileNotFoundError:
        return False, f"claude CLI not found ({CLAUDE_CMD}). Set CLAUDE_CMD env var."
    tail = (r.stdout or "")[-1500:] + ("\n" + (r.stderr or "")[-500:] if r.stderr else "")
    return (r.returncode == 0), tail


def parse_usage_limit(text: str, now: datetime | None = None) -> datetime | None:
    """If Claude's failure output is the usage-limit message, return when to resume.
    Matches e.g. "You've hit your 5-hour limit ... resets 7pm (Asia/Kolkata)".
    Returns None for any other failure, so ordinary errors never trigger a pause.
    Pause is capped at USAGE_PAUSE_CAP and falls back to 30 minutes if no time is given."""
    if not re.search(r"hit your[^\n]{0,60}?limit", text or "", re.I):
        return None
    now = now or datetime.now().astimezone()
    tz = now.tzinfo
    zone = re.search(r"\(([A-Za-z_]+/[A-Za-z_]+)\)", text)
    if zone:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(zone.group(1))
        except Exception:
            pass  # tzdata missing: assume the machine's own zone
    m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text, re.I)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        local_now = now.astimezone(tz)
        reset = local_now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
        if reset <= local_now:
            reset += timedelta(days=1)
        reset += timedelta(minutes=2)
    else:
        reset = now + timedelta(minutes=30)
    return min(reset, now + USAGE_PAUSE_CAP)


# ---------- Main ----------
def process_one(item: dict) -> None:
    global _PAUSE_UNTIL
    role, company, url, page_id = item["role"], item["company"], item["url"], item["page_id"]
    if time.time() < _PAUSE_UNTIL:
        return  # Claude usage limit: leave the row at Build Resume for after the reset
    log(f"processing: {company} / {role}")

    jd, jd_source, jd_state = resolve_jd(url, page_id)
    if jd_state == "wait":
        log("  JD not available yet; will retry next tick")
        if page_id not in _jd_wait_notified:
            _jd_wait_notified.add(page_id)
            set_status(page_id, "Build Resume",
                       log_msg="waiting for the JD to reach GitHub; retrying each tick")
        return
    if jd_state == "missing":
        _jd_wait_since.pop(page_id, None)
        _jd_wait_notified.discard(page_id)
        set_status(page_id, "Resume Failed",
                   log_msg=f"no JD text found for url={url} (waited {JD_WAIT_LIMIT_S // 3600}h)")
        log("  FAIL: no JD")
        return
    if jd_state == "short":
        set_status(page_id, "Resume Failed",
                   log_msg=f"JD too short ({len(jd)} chars, minimum {JD_MIN_CHARS}); "
                           f"not building from a stub. source={jd_source} url={url}")
        log(f"  FAIL: JD too short ({len(jd)} chars)")
        return
    jd_note = f" | JD: {len(jd)} chars, source={jd_source}"
    if jd_state == "thin":
        jd_note += " (thin posting: resume tailored from limited input)"
    if len(jd) > JD_PROMPT_CAP:
        log(f"  warn: JD is {len(jd)} chars; {len(jd) - JD_PROMPT_CAP} beyond the prompt cap will be cut")

    today = date.today().isoformat()
    stem = f"Resume_{sanitize(company)}_{sanitize(role)}_{today}"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JD_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = WORK_DIR / f"{stem}.tex"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    gaps_path = WORK_DIR / f"{stem}_gaps.txt"
    jd_save_path = JD_SAVE_DIR / f"JD_{sanitize(company)}_{sanitize(role)}_{today}.txt"

    # Save the JD alongside the resume for later reference
    try:
        header = (
            f"Company: {company}\nRole: {role}\nURL: {url}\n"
            f"JD length: {len(jd)} chars (source: {jd_source})\n"
            f"Fetched: {datetime.now().isoformat(timespec='seconds')}\n"
            f"{'=' * 70}\n\n"
        )
        jd_save_path.write_text(header + jd, encoding="utf-8")
    except Exception as e:
        log(f"  warn: could not save JD to {jd_save_path}: {e}")

    prompt = build_prompt(company, role, jd, tex_path, pdf_path, gaps_path)
    t0 = time.time()
    ok, tail = run_claude(prompt)
    dur = int(time.time() - t0)

    if not ok:
        reset = parse_usage_limit(tail)
        if reset:
            _PAUSE_UNTIL = max(_PAUSE_UNTIL, reset.timestamp())
            msg = f"Claude usage limit hit; builds paused until {reset.strftime('%H:%M')}"
            set_status(page_id, "Build Resume", log_msg=msg)
            log(f"  PAUSE: {msg} (row stays at Build Resume)")
            return

    gaps = ""
    if gaps_path.exists():
        try:
            gaps = gaps_path.read_text(encoding="utf-8").strip().splitlines()[0][:200]
        except Exception:
            gaps = ""

    if ok and pdf_path.exists() and pdf_path.stat().st_size > 1000:
        jobbr_note = ""
        try:
            import jobbr_sync
            j_ok, j_msg = jobbr_sync.push_application(
                company=company, role=role, source_url=url,
                site=item.get("site", ""), city=item.get("city", ""),
                pdf_path=pdf_path,
                salary_max_lpa=item.get("salary_max"),
                score=item.get("score"), fit=item.get("fit"),
                fit_reason=item.get("fit_reason"),
                skill_gaps=gaps,
                salary_source=item.get("salary_source"),
                notion_page_id=page_id,
                model="Claude Opus 4.7",
                follow_up_days=10,
            )
            jobbr_note = f" | jobbr: {j_msg}" if j_ok else f" | jobbr FAIL: {j_msg}"
        except Exception as e:
            jobbr_note = f" | jobbr exception: {e}"
        set_status(page_id, "Resume Ready", path=str(pdf_path),
                   log_msg=f"built in {dur}s via Claude+Tectonic{jd_note}{jobbr_note}", gaps=gaps)
        log(f"  OK  -> {pdf_path} ({dur}s){jd_note}  gaps={gaps[:80]!r}{jobbr_note}")
    else:
        reason = "pdf missing or empty" if ok else "claude CLI non-zero"
        set_status(page_id, "Resume Failed",
                   log_msg=f"{reason} after {dur}s\n---\n{tail}", gaps=gaps)
        log(f"  FAIL: {reason} ({dur}s)")


def main() -> int:
    missing = [n for n, v in (("NOTION_TOKEN", NOTION_TOKEN),
                              ("NOTION_DB_ID", NOTION_DB_ID)) if not v]
    if missing:
        log(f"FATAL: missing env vars: {missing}")
        return 2
    if not TECTONIC.exists():
        log(f"FATAL: tectonic.exe not found at {TECTONIC}")
        return 2
    if not KB_FILE.exists() or not TPL_FILE.exists():
        log(f"FATAL: resume assets missing: {KB_FILE.exists()=}, {TPL_FILE.exists()=}")
        return 2

    # Cleanup first: drop jobbr cards for rows the user has moved to Skipped
    # AFTER we pushed them. Direct To-apply → Skipped (no prior build) is a
    # no-op because its Resume Path is empty.
    cleanup_skipped()

    global _pause_logged_for
    if time.time() < _PAUSE_UNTIL:
        if _pause_logged_for != _PAUSE_UNTIL:
            _pause_logged_for = _PAUSE_UNTIL
            log(f"builds paused (Claude usage limit) until "
                f"{datetime.fromtimestamp(_PAUSE_UNTIL).strftime('%H:%M')}")
        return 0

    pending = query_pending()
    if not pending:
        return 0

    try:
        parallelism = max(1, int(os.getenv("RESUME_PARALLELISM", "3")))
    except ValueError:
        parallelism = 3
    workers = min(parallelism, len(pending))
    log(f"{len(pending)} pending resume(s), building {workers} in parallel")

    def _safe_process(item: dict) -> None:
        try:
            process_one(item)
        except Exception as e:
            log(f"  unhandled exception for {item.get('company')}/{item.get('role')}: {e}")
            try:
                set_status(item["page_id"], "Resume Failed", log_msg=f"exception: {e}")
            except Exception:
                pass

    if workers == 1:
        for item in pending:
            _safe_process(item)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_safe_process, item) for item in pending]
            for _ in as_completed(futures):
                pass
    return 0


def _pid_is_alive(pid: int) -> bool:
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'alive' }}"],
            capture_output=True, text=True, timeout=5,
        )
        return "alive" in (r.stdout or "")
    except Exception:
        return False


def _single_instance_lock() -> bool:
    """Simple PID-file lock. If lock exists AND that PID is still running, refuse."""
    lock_path = REPO / "resume_daemon.lock"
    if lock_path.exists():
        try:
            content = lock_path.read_text(encoding="utf-8")
            prev_pid = int(content.split("pid=")[1].split()[0])
            if prev_pid != os.getpid() and _pid_is_alive(prev_pid):
                log(f"--- daemon already running (pid={prev_pid}); exiting ---")
                return False
        except Exception:
            pass  # stale/corrupt lock — overwrite
    try:
        lock_path.write_text(
            f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return True


def loop(interval_seconds: int) -> int:
    """Long-running mode: poll the Notion DB forever, sleeping `interval` seconds between ticks."""
    if not _single_instance_lock():
        log("--- another resume_daemon loop already holds the lock; exiting ---")
        return 0
    log(f"--- resume_daemon loop started (interval={interval_seconds}s, pid={os.getpid()}) ---")
    while True:
        try:
            main()
        except Exception as e:
            log(f"loop tick exception: {e}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--loop":
        try:
            n = int(args[1])
        except ValueError:
            n = 300
        sys.exit(loop(n))
    sys.exit(main())
