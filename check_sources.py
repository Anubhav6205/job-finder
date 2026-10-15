"""Diagnostic: check accessibility of every data source used by job_finder.py."""
import time
from jobspy import scrape_jobs

SITES = ["linkedin", "indeed", "naukri", "glassdoor", "google"]
COMMON = dict(
    search_term="AI Engineer",
    location="Hyderabad, India",
    results_wanted=5,
    hours_old=72,
    country_indeed="India",
    linkedin_fetch_description=True,
    verbose=0,
)

results = []

for i, site in enumerate(SITES):
    if i > 0:
        print(f"  (waiting 5 s...)\n")
        time.sleep(5)

    print(f"[{site}] testing...", end=" ", flush=True)
    kwargs = dict(site_name=[site], **COMMON)
    if site == "google":
        kwargs["google_search_term"] = "AI Engineer jobs in Hyderabad since 3 days"

    t0 = time.time()
    try:
        df = scrape_jobs(**kwargs)
        elapsed = round(time.time() - t0, 1)
        total = len(df)
        with_desc = int(df["description"].fillna("").str.len().ge(200).sum()) if total else 0
        if total == 0:
            status = "EMPTY"
        else:
            status = "OK"
        results.append(dict(site=site, status=status, jobs=total,
                            with_desc=with_desc, elapsed=elapsed, error=""))
        print(f"{status}  jobs={total}  with_desc={with_desc}  time={elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        err = str(e)[:200]
        results.append(dict(site=site, status="ERROR", jobs=0,
                            with_desc=0, elapsed=elapsed, error=err))
        print(f"ERROR  time={elapsed}s")
        print(f"  -> {err}")

print("\n" + "=" * 60)
print(f"{'Site':<12} {'Status':<8} {'Jobs':>5} {'With desc':>10} {'Time':>7}  Error")
print("-" * 60)
for r in results:
    print(f"{r['site']:<12} {r['status']:<8} {r['jobs']:>5} {r['with_desc']:>10} {r['elapsed']:>6}s  {r['error'][:60]}")
