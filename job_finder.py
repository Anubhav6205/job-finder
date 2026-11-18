"""
Daily job finder.
Scrape fresh jobs -> GPT reads each JD -> plain-code rules filter -> score vs resume -> Notion + CSV.
"""
import hashlib
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

YOE_CAP = 3               # reject if the JD's minimum YOE is above this (stretch)

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
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
RESUME_FILE = Path("resume.txt")
# ====================================================================

PROMPT = """You are a careful recruiter reading a job posting.
Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
{"min_yoe": number|null, "max_yoe": number|null, "yoe_quote": string|null,
 "max_joining_days": number|null, "joining_quote": string|null,
 "city": string|null}

You receive LOCATION_FIELD (the job-board's structured location string, most reliable),
TITLE, COMPANY and the JD body. Use all of them.

Rules for CITY:
- Pick the single canonical Indian city where the candidate would report to work.
- Return EXACTLY ONE of: "Hyderabad", "Bangalore", "Pune", "Mumbai", "Delhi NCR",
  "Chennai", "Gurgaon", "Noida", "Kolkata", "Ahmedabad", "Remote", "Pan-India",
  or null if genuinely unclear.
- Use LOCATION_FIELD as the ground truth. Expand state codes:
  KA/Karnataka -> Bangalore,  TS/TG/Telangana -> Hyderabad,
  MH/Maharashtra -> infer Mumbai vs Pune from JD (if JD mentions Pune, use Pune, else Mumbai),
  TN/Tamil Nadu -> Chennai,  WB/West Bengal -> Kolkata,
  DL/Delhi -> Delhi NCR,  HR/Haryana -> Gurgaon,  UP/Uttar Pradesh -> Noida.
- Expand suburbs/sub-areas to parent city:
  Pimpri-Chinchwad, Hinjewadi, Kharadi -> Pune.
  Whitefield, Electronic City, Bellandur, Koramangala, Bengaluru East -> Bangalore.
  Secunderabad, Gachibowli, HITEC City -> Hyderabad.
  Thane, Navi Mumbai, Andheri, Powai -> Mumbai.
- If LOCATION_FIELD lists multiple cities (separated by commas or slashes),
  pick the FIRST city mentioned.
- If LOCATION_FIELD is only "India" or empty, look in the JD for office/location lines.
  If JD says remote/WFH with no city, return "Remote".

Rules for YOE:
- min_yoe / max_yoe = total professional experience required for THIS role.
  "3-5 years" -> 3 and 5. "5+ years" -> 5 and null. "at least 4 years" -> 4 and null.
  If the JD gives an overall number and a narrower one ("5+ years overall, 2+ in GenAI"),
  use the overall number. Ignore numbers about the company or its products.
  If no experience requirement is stated, both null.
- yoe_quote = the shortest exact phrase from the JD that supports the value, else null.

Rules for JOINING:
- max_joining_days = the longest notice period or joining time the JD will ACCEPT, in days.
  "Immediate joiners only" -> 0. "Immediate to 15 days" -> 15. "Notice up to 30 days" -> 30.
  "Must join within 1 month" -> 30. "Notice up to 90 days" -> 90.
  A soft preference ("immediate joiners preferred", "early joiners a plus") is NOT a hard limit:
  set max_joining_days to null but still fill joining_quote.
  If nothing is said about notice or joining, null.
- joining_quote = the shortest exact phrase from the JD that supports the value, else null.
"""

RESUME_PROFILE = RESUME_FILE.read_text(encoding="utf-8") if RESUME_FILE.exists() else ""

SCORE_PROMPT = f"""You are a senior tech recruiter evaluating fit between a candidate
and a SPECIFIC job. Do not score on surface keyword overlap. Analyse deeply
across these five dimensions, then produce the score and bullets.

1. ROLE LEVEL — does the title/seniority band match the candidate?
   Senior candidate for junior role (or vice versa) = poor fit.
2. CORE SKILLS — do the JD's REQUIRED skills map to the candidate's primary
   strengths, not just any mention on the resume?
   ("Python" anywhere on resume is NOT a match; "2yrs production LangChain
   RAG" IS a match.)
3. DOMAIN & STACK — is the industry and actual stack aligned?
   (BFSI vs SaaS; LangChain/LlamaIndex vs HuggingFace fine-tuning;
   web scraping vs data engineering.)
4. YOE BAND — does the required experience range contain the candidate's YOE?
5. RED FLAGS — specific requirements the candidate CANNOT satisfy
   (certifications, team-lead headcount, on-call, non-tech skills, sales,
   specific vertical experience).

Score 1-10 (10 = perfect fit).

Return ONLY a JSON object: {{"score": number, "fit_reason": string}}

fit_reason must be 3-5 short bullet points, one per line, each starting
with "• " (bullet + space). Each bullet states ONE precise reason with
CONCRETE evidence quoted or paraphrased from the JD/resume. No fluff,
no restating the job title, no generic claims.

Bad bullet:  "• Candidate has relevant AI skills."
Good bullet: "• Strong: 2yrs production RAG with LangChain + FAISS matches JD"
Good bullet: "• Gap: JD needs 5+ yrs MLOps, candidate has ~1.5 yrs"
Good bullet: "• Red flag: role expects team lead (10 reports), candidate is IC"

CANDIDATE PROFILE:
{RESUME_PROFILE}
"""

SALARY_PROMPT = """You are a compensation analyst for Indian tech roles.
You receive TWO sets of Google search snippets:
  - CTC_SNIPPETS: likely contains annual CTC data (Glassdoor/AmbitionBox/Levels.fyi).
  - INHAND_SNIPPETS: likely contains monthly in-hand / take-home data.
Use BOTH to produce the best estimate for this COMPANY + ROLE in India.

Return ONLY a JSON object:
{"min_lpa": number|null, "max_lpa": number|null,
 "monthly_inhand_inr": number|null, "inhand_basis": string,
 "source": string}

Rules for min_lpa / max_lpa (annual CTC range):
- LPA = INR lakhs per annum (1 lakh = 100,000). Normalize everything to LPA.
  "₹15L" / "15 LPA" / "15 lakhs" -> 15.
  "₹15,00,000" / "₹1,500,000" / "1.5M INR" -> 15.
  "$30k" -> ~25 (use ~83 INR/USD).
- Prefer TOTAL CTC (base + variable + stock). If only base is listed, use base.
- If a range is given ("12-25 LPA"), min=12, max=25.
- If only a single number is listed ("avg ~18 LPA"), set min=max=18.
- Prefer the SPECIFIC COMPANY+ROLE match, else role-level industry avg.

Rules for monthly_inhand_inr (precise rupee amount per month, after tax):
- First look in INHAND_SNIPPETS for an EXPLICIT monthly in-hand / take-home
  figure for this COMPANY+ROLE. Examples:
    "in-hand ₹85,000 per month" -> 85000
    "take home around 1.2L monthly" -> 120000
    "net monthly salary of ₹65k" -> 65000
  Use the midpoint if a range is given.
- If no explicit in-hand is found but CTC is known, ESTIMATE using the
  candidate's standard India new-tax-regime bracket on a monthly basis:
     CTC <= 6 LPA     -> in-hand ~88% of (CTC_mid / 12)
     6  < CTC <= 10   -> ~82%
     10 < CTC <= 15   -> ~75%
     15 < CTC <= 25   -> ~70%
     25 < CTC <= 40   -> ~65%
     40 < CTC <= 60   -> ~60%
     CTC > 60         -> ~55%  (large variable/stock drags cash down)
  Use CTC_mid = (min_lpa + max_lpa) / 2 if both present, else whichever exists.
  Round to the nearest 1,000. Example: CTC 18 LPA -> 18*100000/12 = 150000 * 0.70 = 105000.
- inhand_basis = one short phrase:
    "explicit: AmbitionBox in-hand ₹85k"      (if found directly)
    "estimated from 12-18 LPA CTC @ 75%"      (if derived)
    "no data"                                 (if neither CTC nor in-hand found)

source = short phrase naming WHERE the CTC number comes from, max 90 chars.
  Examples: "AmbitionBox - Infosys AI Engineer (189 reports)",
            "Glassdoor - Microsoft AI Engineer, Bangalore",
            "Industry avg - no company data found",
            "no salary data in snippets".

If snippets mention nothing: all four values null; source="no data"; inhand_basis="no data".
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
    if "Salary Min (LPA)" not in existing:
        updates["Salary Min (LPA)"] = {"number": {"format": "number"}}
    if "Salary Max (LPA)" not in existing:
        updates["Salary Max (LPA)"] = {"number": {"format": "number"}}
    if "Monthly In-hand (INR)" not in existing:
        updates["Monthly In-hand (INR)"] = {"number": {"format": "rupee"}}
    if "In-hand Basis" not in existing:
        updates["In-hand Basis"] = {"rich_text": {}}
    if "Salary Source" not in existing:
        updates["Salary Source"] = {"rich_text": {}}
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


def extract(title: str, company: str, desc: str, location_field: str = "") -> dict | None:
    text = _llm_call([{
        "role": "user",
        "content": (
            f"{PROMPT}\n"
            f"LOCATION_FIELD: {location_field!r}\n"
            f"TITLE: {title}\n"
            f"COMPANY: {company}\n\n"
            f"JD:\n{desc[:15000]}"
        ),
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
    """LLM returns a canonical city; this is a thin fallback for null/empty."""
    if not city:
        return "Unknown"
    return str(city).strip() or "Unknown"


def _serper(query: str) -> str:
    """Return a joined snippet block from Serper, or empty string on failure."""
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5, "gl": "in"},
            timeout=10,
        )
        if resp.status_code != 200:
            return ""
        organic = resp.json().get("organic", [])[:5]
        return "\n".join(
            f"- {o.get('title','')}: {o.get('snippet','')}" for o in organic
        )[:1500]
    except Exception:
        return ""


def fetch_salary(company: str, role: str) -> dict:
    """Two Serper queries (CTC + in-hand) -> LLM returns CTC range + monthly in-hand INR."""
    empty = {"min_lpa": None, "max_lpa": None, "monthly_inhand_inr": None,
             "inhand_basis": None, "source": None}
    if not SERPER_API_KEY:
        return {**empty, "source": "no api key"}

    ctc_snips = _serper(f"{company} {role} salary india ambitionbox glassdoor")
    inhand_snips = _serper(f"{company} {role} in hand salary monthly take home india ambitionbox")

    if not (ctc_snips.strip() or inhand_snips.strip()):
        return {**empty, "source": "no search results"}

    text = _llm_call(
        messages=[{
            "role": "user",
            "content": (
                f"{SALARY_PROMPT}\nCOMPANY: {company}\nROLE: {role}\n\n"
                f"CTC_SNIPPETS:\n{ctc_snips or '(none)'}\n\n"
                f"INHAND_SNIPPETS:\n{inhand_snips or '(none)'}"
            ),
        }],
        max_tokens=800,
    )
    if text is None:
        return {**empty, "source": "llm failed"}
    try:
        return json.loads(re.sub(r"```(?:json)?", "", text).strip())
    except Exception:
        return {**empty, "source": "parse failed"}


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
        "Location": text_prop(r["city"]),
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
    if r.get("salary_min") is not None:
        props["Salary Min (LPA)"] = {"number": r["salary_min"]}
    if r.get("salary_max") is not None:
        props["Salary Max (LPA)"] = {"number": r["salary_max"]}
    if r.get("monthly_inhand_inr") is not None:
        props["Monthly In-hand (INR)"] = {"number": r["monthly_inhand_inr"]}
    if r.get("inhand_basis"):
        props["In-hand Basis"] = text_prop(r["inhand_basis"])
    if r.get("salary_source"):
        props["Salary Source"] = text_prop(r["salary_source"])

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

    jd_cache = DATA / "jd_cache"
    jd_cache.mkdir(parents=True, exist_ok=True)
    rows, today = [], date.today().isoformat()
    for _, j in df.iterrows():
        desc = j.get("description")
        desc = desc if isinstance(desc, str) else ""
        # Persist the raw JD body so the resume daemon can pick it up later
        # without re-scraping (Indeed blocks direct HTTP fetches).
        if desc:
            try:
                key = hashlib.md5(j["job_url"].encode()).hexdigest()
                (jd_cache / f"{key}.txt").write_text(desc, encoding="utf-8")
            except Exception:
                pass
        if len(desc) < 200:
            x, (bucket, reason) = {}, ("unclear", "no JD text fetched")
            scored = {"score": None, "fit_reason": None}
        else:
            x = extract(j["title"], j["company"], desc, j.get("location") or "")
            bucket, reason = decide(x)
            if bucket != "reject":
                scored = score_job(j["title"], j["company"], desc)
            else:
                scored = {"score": None, "fit_reason": None}
        x = x or {}
        salary = (
            fetch_salary(j.get("company") or "", j.get("title") or "")
            if bucket != "reject"
            else {"min_lpa": None, "max_lpa": None, "monthly_inhand_inr": None,
                  "inhand_basis": None, "source": None}
        )
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
            "salary_min":       salary.get("min_lpa"),
            "salary_max":       salary.get("max_lpa"),
            "monthly_inhand_inr": salary.get("monthly_inhand_inr"),
            "inhand_basis":     salary.get("inhand_basis"),
            "salary_source":    salary.get("source"),
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
