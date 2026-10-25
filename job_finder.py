"""
Daily job finder.
Scrape fresh jobs -> GPT reads each JD -> plain-code rules filter -> score vs resume -> Notion + CSV.
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
    "AI Engineer",
    "GenAI Engineer",
    "Applied AI Engineer",
    "LLM Engineer",
    "Software Engineer Generative AI",
]
LOCATIONS = ["Hyderabad, India", "Bengaluru, India", "Pune, India"]
SITES = ["linkedin", "indeed", "naukri"]
RESULTS_PER_QUERY = 25
HOURS_OLD = 48            # 48-hour window so a missed run doesn't lose jobs

YOE_CAP = 4               # reject if the JD's minimum YOE is above this (stretch)

TITLE_BLOCKLIST = re.compile(
    r"\b(intern|internship|principal|staff|director|head|manager|vp|architect|trainee|fresher)\b",
    re.I,
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DATA = Path("data")
SEEN_FILE = DATA / "seen_urls.txt"
OUT_FILE = DATA / "jobs.csv"
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DB_ID = os.getenv("NOTION_DB_ID")
RESUME_FILE = Path("resume.txt")
# ====================================================================

PROMPT = """You read job descriptions like a careful recruiter.
Extract the hiring constraints from the JD below.
Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
{"min_yoe": number|null, "max_yoe": number|null, "yoe_quote": string|null,
 "max_joining_days": number|null, "joining_quote": string|null,
 "city": string|null}

Rules:
- min_yoe / max_yoe = total professional experience required for THIS role.
  "3-5 years" -> 3 and 5. "5+ years" -> 5 and null. "at least 4 years" -> 4 and null.
  If the JD gives an overall number and a narrower one ("5+ years overall, 2+ in GenAI"),
  use the overall number. Ignore numbers about the company or its products.
  If no experience requirement is stated, both null.
- max_joining_days = the longest notice period or joining time the JD will ACCEPT, in days.
  "Immediate joiners only" -> 0. "Immediate to 15 days" -> 15. "Notice up to 30 days" -> 30.
  "Must join within 1 month" -> 30. "Notice up to 90 days" -> 90.
  A soft preference ("immediate joiners preferred", "early joiners a plus") is NOT a hard limit:
  set max_joining_days to null but still fill joining_quote.
  If nothing is said about notice or joining, null.
- city = the work location city explicitly mentioned in the JD.
  Look for city names, office addresses or location lines.
  Return exactly one of: "Hyderabad", "Bangalore", "Pune", "Mumbai", "Remote", "Pan-India".
  If multiple cities listed, return the primary one (first mentioned or most prominent).
  If the JD says remote/WFH/hybrid with no city, return "Remote".
  If no location is mentioned at all, return null.
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

client = OpenAI()  # reads OPENAI_API_KEY from the environment

NOTION_HEADERS = lambda: {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def setup_notion_schema() -> None:
    """Ensure Score (number) and Fit (select) properties exist in the Notion DB."""
    if not (NOTION_TOKEN and NOTION_DB_ID):
        return
    resp = requests.get(
        f"https://api.notion.com/v1/databases/{NOTION_DB_ID}",
        headers=NOTION_HEADERS(),
        timeout=10,
    )
    if resp.status_code != 200:
        return
    existing = resp.json().get("properties", {})
    updates = {}
    if "Score" not in existing:
        updates["Score"] = {"number": {"format": "number"}}
    if "Fit" not in existing:
        updates["Fit"] = {"select": {"options": [
            {"name": "Strong",   "color": "green"},
            {"name": "Moderate", "color": "yellow"},
            {"name": "Weak",     "color": "red"},
        ]}}
    if "Fit Reason" not in existing:
        updates["Fit Reason"] = {"rich_text": {}}
    if "City" not in existing:
        updates["City"] = {"select": {"options": [
            {"name": "Hyderabad",  "color": "blue"},
            {"name": "Bangalore",  "color": "purple"},
            {"name": "Pune",       "color": "orange"},
            {"name": "Mumbai",     "color": "pink"},
            {"name": "Remote",     "color": "green"},
            {"name": "Pan-India",  "color": "yellow"},
            {"name": "Unknown",    "color": "gray"},
        ]}}
    if updates:
        r = requests.patch(
            f"https://api.notion.com/v1/databases/{NOTION_DB_ID}",
            headers=NOTION_HEADERS(),
            json={"properties": updates},
            timeout=10,
        )
        if r.status_code < 300:
            print(f"[notion] added properties: {list(updates.keys())}")


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
            time.sleep(3)  # be gentle, avoids rate limits
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="job_url")


def _llm_call(messages: list, max_tokens: int = 2000) -> str | None:
    """Shared retry wrapper for all LLM calls."""
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


def normalize_city(city: str | None) -> str:
    if not city:
        return "Unknown"
    c = city.strip().lower()
    if any(x in c for x in ["hyderabad", "telangana", "secunderabad"]):
        return "Hyderabad"
    if any(x in c for x in ["bangalore", "bengaluru", "bengalore"]):
        return "Bangalore"
    if "pune" in c:
        return "Pune"
    if any(x in c for x in ["mumbai", "bombay"]):
        return "Mumbai"
    if any(x in c for x in ["remote", "work from home", "wfh"]):
        return "Remote"
    if any(x in c for x in ["pan-india", "pan india", "anywhere"]):
        return "Pan-India"
    return city.strip().title()   # return as-is if unrecognised


def yoe_display(x: dict) -> str:
    mn, mx = x.get("min_yoe"), x.get("max_yoe")
    if mn is None:
        return "Could not identify"
    if mx is not None:
        return f"{mn}–{mx} years required"
    return f"{mn}+ years required"


def fit_label(score) -> str | None:
    if score is None:
        return None
    if score >= 8:
        return "Strong"
    if score >= 5:
        return "Moderate"
    return "Weak"


def decide(x: dict | None) -> tuple[str, str]:
    """Plain rules, so every decision is predictable and easy to debug."""
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
        "Role":    {"title": [{"text": {"content": str(r["title"])[:200]}}]},
        "Company": text_prop(r["company"]),
        "URL":     {"url": r["job_url"]},
        "Bucket":  {"select": {"name": r["bucket"]}},
        "Site":    {"select": {"name": r["site"]}},
        "Location": text_prop(r["location"]),
        "YOE":     text_prop(r["yoe_display"]),
        "Joining": text_prop(r["joining_quote"]),
        "Found":   {"date": {"start": r["found"]}},
        "Status":  {"select": {"name": "To apply"}},
    }
    if r.get("score") is not None:
        props["Score"] = {"number": r["score"]}
    if r.get("fit") is not None:
        props["Fit"] = {"select": {"name": r["fit"]}}
    if r.get("fit_reason"):
        props["Fit Reason"] = text_prop(r["fit_reason"])
    if r.get("city"):
        props["City"] = {"select": {"name": r["city"]}}

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
    setup_notion_schema()
    seen = set(SEEN_FILE.read_text().split()) if SEEN_FILE.exists() else set()

    df = fetch_jobs()
    if df.empty:
        print("No jobs fetched.")
        return

    df = df[~df["job_url"].isin(seen)]
    df = df[~df["title"].fillna("").str.contains(TITLE_BLOCKLIST)]
    # same job posted on LinkedIn and Naukri -> keep one
    key = df["company"].fillna("").str.lower().str.strip() + "|" + df["title"].fillna("").str.lower().str.strip()
    df = df.loc[~key.duplicated()]
    print(f"[filter] {len(df)} new jobs to read")

    rows, today = [], date.today().isoformat()
    for _, j in df.iterrows():
        desc = j.get("description")
        desc = desc if isinstance(desc, str) else ""
        if len(desc) < 200:
            x, (bucket, reason) = {}, ("unclear", "no JD text fetched")
            scored = {"score": None, "fit_reason": None}
        else:
            x = extract(j["title"], j["company"], desc)
            bucket, reason = decide(x)
            if bucket != "reject":
                scored = score_job(j["title"], j["company"], desc)
            else:
                scored = {"score": None, "fit_reason": None}
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
            "city":             normalize_city(x.get("city")),
            "site":             clean(j.get("site")),
            "job_url":          j["job_url"],
            "min_yoe":          x.get("min_yoe"),
            "max_yoe":          x.get("max_yoe"),
            "yoe_display":      yoe_display(x),
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
