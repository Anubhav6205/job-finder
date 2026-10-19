"""
Job finder for Sakshi Gupta — Credit Risk & Underwriting roles.
Same logic as job_finder.py, separate config, resume and Notion DB.
"""
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
import requests
from openai import OpenAI
from jobspy import scrape_jobs

# ============================== CONFIG ==============================
SEARCH_TERMS = [
    "Credit Risk Analyst",
    "Credit Underwriting",
    "Senior Credit Analyst",
    "Risk Analyst NBFC",
    "Financial Risk Analyst",
]
LOCATIONS = ["Bengaluru, India", "Mumbai, India", "Hyderabad, India", "Pune, India"]
SITES = ["linkedin", "indeed", "naukri"]
RESULTS_PER_QUERY = 25
HOURS_OLD = 96            # 4-day window

YOE_CAP = 8               # Sakshi has 5+ years; reject only if JD needs 9+

TITLE_BLOCKLIST = re.compile(
    r"\b(intern|internship|fresher|trainee|vp|president|cxo|chief)\b",
    re.I,
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DATA = Path("data_sakshi")
SEEN_FILE = DATA / "seen_urls.txt"
OUT_FILE  = DATA / "jobs.csv"
NOTION_TOKEN  = os.getenv("NOTION_TOKEN")
NOTION_DB_ID  = "8c7401d073224e54aee1fb6f21be9e2d"   # Sakshi Job Finder DB
RESUME_FILE   = Path("sakshi_resume.txt")
# ====================================================================

PROMPT = """You read job descriptions like a careful recruiter.
Extract the hiring constraints from the JD below.
Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
{"min_yoe": number|null, "max_yoe": number|null, "yoe_quote": string|null,
 "max_joining_days": number|null, "joining_quote": string|null}

Rules:
- min_yoe / max_yoe = total professional experience required for THIS role.
  "3-5 years" -> 3 and 5. "5+ years" -> 5 and null. "at least 4 years" -> 4 and null.
  If no experience requirement is stated, both null.
- max_joining_days = the longest notice period or joining time the JD will ACCEPT, in days.
  "Immediate joiners only" -> 0. "Notice up to 30 days" -> 30.
  A soft preference is NOT a hard limit: set max_joining_days to null but fill joining_quote.
  If nothing is said about notice, null.
- *_quote = the shortest exact phrase from the JD that supports the value, else null.
"""

RESUME_PROFILE = RESUME_FILE.read_text(encoding="utf-8") if RESUME_FILE.exists() else ""

SCORE_PROMPT = f"""You are a recruiter scoring fit between a candidate and a job.
Score 1-10 (10 = perfect fit) based on skills, experience level and role type.
Return ONLY a JSON object: {{"score": number, "fit_reason": string}}
The fit_reason must be one sentence explaining the score.

CANDIDATE PROFILE:
{RESUME_PROFILE}
"""

client = OpenAI()

NOTION_HEADERS = lambda: {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def fetch_jobs() -> pd.DataFrame:
    frames = []
    for term in SEARCH_TERMS:
        for loc in LOCATIONS:
            try:
                df = scrape_jobs(
                    site_name=SITES,
                    search_term=term,
                    location=loc,
                    results_wanted=RESULTS_PER_QUERY,
                    hours_old=HOURS_OLD,
                    country_indeed="India",
                    linkedin_fetch_description=True,
                    verbose=0,
                )
                print(f"[fetch] {term} @ {loc}: {len(df)}")
                frames.append(df)
            except Exception as e:
                print(f"[warn] {term} @ {loc}: {e}", file=sys.stderr)
            time.sleep(3)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="job_url")


def _llm_call(messages: list, max_tokens: int = 2000) -> str | None:
    delays = [2, 4, 8, 16]
    for attempt in range(len(delays) + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=max_tokens,
                reasoning_effort="low",
                response_format={"type": "json_object"},
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            if "insufficient_quota" in err:
                print("\n[FATAL] OpenAI quota exhausted — add credits at https://platform.openai.com/settings/organization/billing/", file=sys.stderr)
                sys.exit(1)
            status = getattr(e, "status_code", None)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if retryable and attempt < len(delays):
                wait = delays[attempt]
                print(f"[retry {attempt+1}/4] HTTP {status} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"[warn] LLM error: {err}", file=sys.stderr)
            return None


def extract(title: str, company: str, desc: str) -> dict | None:
    text = _llm_call([{
        "role": "user",
        "content": f"{PROMPT}\nTITLE: {title}\nCOMPANY: {company}\n\nJD:\n{desc[:15000]}",
    }])
    if text is None:
        return None
    try:
        return json.loads(re.sub(r"```(?:json)?", "", text).strip())
    except Exception:
        return None


def score_job(title: str, company: str, desc: str) -> dict:
    if not RESUME_PROFILE:
        return {"score": None, "fit_reason": "no resume profile"}
    text = _llm_call(
        messages=[{
            "role": "user",
            "content": f"{SCORE_PROMPT}\n\nJOB TITLE: {title}\nCOMPANY: {company}\n\nJD:\n{desc[:8000]}",
        }],
        max_tokens=1500,
    )
    if text is None:
        return {"score": None, "fit_reason": "scoring failed"}
    try:
        return json.loads(re.sub(r"```(?:json)?", "", text).strip())
    except Exception:
        return {"score": None, "fit_reason": "scoring failed"}


def fit_label(score) -> str | None:
    if score is None:
        return None
    if score >= 8:
        return "Strong"
    if score >= 5:
        return "Moderate"
    return "Weak"


def decide(x: dict | None) -> tuple[str, str]:
    if x is None:
        return "unclear", "could not read JD"
    mn = x.get("min_yoe")
    if mn is not None and mn > YOE_CAP:
        return "reject", f"needs {mn}+ yrs"
    if mn is None:
        return "unclear", "YOE not stated"
    return "match", ""


def text_prop(value) -> dict:
    return {"rich_text": [{"text": {"content": str(value or "")[:1900]}}]}


def push_notion(r: dict) -> None:
    if not (NOTION_TOKEN and NOTION_DB_ID):
        return
    props = {
        "Role":     {"title": [{"text": {"content": str(r["title"])[:200]}}]},
        "Company":  text_prop(r["company"]),
        "URL":      {"url": r["job_url"]},
        "Bucket":   {"select": {"name": r["bucket"]}},
        "Site":     {"select": {"name": r["site"]}},
        "Location": text_prop(r["location"]),
        "YOE":      text_prop(r["yoe_quote"] or r["reason"]),
        "Joining":  text_prop(r["joining_quote"]),
        "Found":    {"date": {"start": r["found"]}},
        "Status":   {"select": {"name": "To apply"}},
    }
    if r.get("score") is not None:
        props["Score"] = {"number": r["score"]}
    if r.get("fit") is not None:
        props["Fit"] = {"select": {"name": r["fit"]}}
    if r.get("fit_reason"):
        props["Fit Reason"] = text_prop(r["fit_reason"])

    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers=NOTION_HEADERS(),
        json={"parent": {"database_id": NOTION_DB_ID}, "properties": props},
        timeout=30,
    )
    if resp.status_code >= 300:
        print(f"[warn] Notion {resp.status_code}: {resp.text[:300]}", file=sys.stderr)


def clean(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def main() -> None:
    DATA.mkdir(exist_ok=True)
    seen = set(SEEN_FILE.read_text().split()) if SEEN_FILE.exists() else set()

    df = fetch_jobs()
    if df.empty:
        print("No jobs fetched.")
        return

    df = df[~df["job_url"].isin(seen)]
    df = df[~df["title"].fillna("").str.contains(TITLE_BLOCKLIST)]
    key = df["company"].fillna("").str.lower().str.strip() + "|" + df["title"].fillna("").str.lower().str.strip()
    df = df.loc[~key.duplicated()]
    print(f"[filter] {len(df)} new jobs to read")

    rows, today = [], date.today().isoformat()
    for _, j in df.iterrows():
        desc = j.get("description")
        desc = desc if isinstance(desc, str) else ""
        if len(desc) < 200:
            x, (bucket, reason) = None, ("unclear", "no JD text fetched")
            scored = {"score": None, "fit_reason": None}
        else:
            x = extract(j["title"], j["company"], desc)
            bucket, reason = decide(x)
            scored = score_job(j["title"], j["company"], desc) if bucket != "reject" else {"score": None, "fit_reason": None}
        x = x or {}
        row = {
            "found":            today,
            "bucket":           bucket,
            "reason":           reason,
            "score":            scored.get("score"),
            "fit":              fit_label(scored.get("score")),
            "fit_reason":       scored.get("fit_reason"),
            "title":            clean(j.get("title")),
            "company":          clean(j.get("company")),
            "location":         clean(j.get("location")),
            "site":             clean(j.get("site")),
            "job_url":          j["job_url"],
            "min_yoe":          x.get("min_yoe"),
            "max_yoe":          x.get("max_yoe"),
            "max_joining_days": x.get("max_joining_days"),
            "yoe_quote":        x.get("yoe_quote"),
            "joining_quote":    x.get("joining_quote"),
        }
        rows.append(row)
        if bucket != "reject":
            push_notion(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, mode="a", header=not OUT_FILE.exists(), index=False)
    with SEEN_FILE.open("a") as f:
        f.writelines(u + "\n" for u in out["job_url"])

    counts = out["bucket"].value_counts().to_dict()
    print(f"[done] match={counts.get('match', 0)} unclear={counts.get('unclear', 0)} reject={counts.get('reject', 0)}")
    if RESUME_PROFILE:
        scored_rows = [r for r in rows if r["score"] is not None]
        if scored_rows:
            top = sorted(scored_rows, key=lambda r: r["score"], reverse=True)[:5]
            print("\nTop 5 by score:")
            for r in top:
                print(f"  [{r['score']:>2}/10] {r['fit']:<8} {r['title']} @ {r['company']}")


if __name__ == "__main__":
    main()
