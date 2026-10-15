"""
Daily job finder.
Scrape fresh jobs -> Claude reads each JD -> plain-code rules filter -> Notion + CSV.
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
HOURS_OLD = 24            # only jobs posted in the last day

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
# ====================================================================

PROMPT = """You read job descriptions like a careful recruiter.
Extract the hiring constraints from the JD below.
Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
{"min_yoe": number|null, "max_yoe": number|null, "yoe_quote": string|null,
 "max_joining_days": number|null, "joining_quote": string|null}

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
- *_quote = the shortest exact phrase from the JD that supports the value, else null.
"""

client = OpenAI()  # reads OPENAI_API_KEY from the environment


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


def extract(title: str, company: str, desc: str) -> dict | None:
    delays = [2, 4, 8, 16]
    for attempt in range(len(delays) + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=2000,
                reasoning_effort="low",
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": f"{PROMPT}\nTITLE: {title}\nCOMPANY: {company}\n\nJD:\n{desc[:15000]}",
                }],
            )
            text = resp.choices[0].message.content or ""
            text = re.sub(r"```(?:json)?", "", text).strip()
            return json.loads(text)
        except Exception as e:
            err = str(e)
            if "insufficient_quota" in err:
                print("\n[FATAL] OpenAI quota exhausted — add credits at https://platform.openai.com/settings/organization/billing/", file=sys.stderr)
                sys.exit(1)
            status = getattr(e, "status_code", None)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if retryable and attempt < len(delays):
                wait = delays[attempt]
                print(f"[retry {attempt+1}/4] {title} @ {company}: HTTP {status} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"[warn] LLM failed on {title} @ {company}: {err}", file=sys.stderr)
            return None


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
        "Role": {"title": [{"text": {"content": str(r["title"])[:200]}}]},
        "Company": text_prop(r["company"]),
        "URL": {"url": r["job_url"]},
        "Bucket": {"select": {"name": r["bucket"]}},
        "Site": {"select": {"name": r["site"]}},
        "Location": text_prop(r["location"]),
        "YOE": text_prop(r["yoe_quote"] or r["reason"]),
        "Joining": text_prop(r["joining_quote"]),
        "Found": {"date": {"start": r["found"]}},
        "Status": {"select": {"name": "To apply"}},
    }
    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
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
    # same job posted on LinkedIn and Naukri -> keep one
    key = df["company"].fillna("").str.lower().str.strip() + "|" + df["title"].fillna("").str.lower().str.strip()
    df = df.loc[~key.duplicated()]
    print(f"[filter] {len(df)} new jobs to read")

    rows, today = [], date.today().isoformat()
    for _, j in df.iterrows():
        desc = j.get("description")
        desc = desc if isinstance(desc, str) else ""
        if len(desc) < 200:
            x, (bucket, reason) = None, ("unclear", "no JD text fetched")
        else:
            x = extract(j["title"], j["company"], desc)
            bucket, reason = decide(x)
        x = x or {}
        row = {
            "found": today,
            "bucket": bucket,
            "reason": reason,
            "title": clean(j.get("title")),
            "company": clean(j.get("company")),
            "location": clean(j.get("location")),
            "site": clean(j.get("site")),
            "job_url": j["job_url"],
            "min_yoe": x.get("min_yoe"),
            "max_yoe": x.get("max_yoe"),
            "max_joining_days": x.get("max_joining_days"),
            "yoe_quote": x.get("yoe_quote"),
            "joining_quote": x.get("joining_quote"),
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


if __name__ == "__main__":
    main()
