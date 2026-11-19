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
import time
from datetime import date, datetime
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


def jd_refetch(url: str) -> str | None:
    """Fallback if CSV is missing the row or desc is thin."""
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        txt = soup.get_text("\n", strip=True)
        return txt if len(txt) >= 400 else None
    except Exception:
        return None


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
{jd[:12000]}
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


# ---------- Main ----------
def process_one(item: dict) -> None:
    role, company, url, page_id = item["role"], item["company"], item["url"], item["page_id"]
    log(f"processing: {company} / {role}")

    jd = jd_from_cache(url) or jd_refetch(url) or jd_rescrape(url)
    if not jd:
        set_status(page_id, "Resume Failed", log_msg=f"no JD text found for url={url}")
        log("  FAIL: no JD")
        return

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
                   log_msg=f"built in {dur}s via Claude+Tectonic{jobbr_note}", gaps=gaps)
        log(f"  OK  -> {pdf_path} ({dur}s)  gaps={gaps[:80]!r}{jobbr_note}")
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
