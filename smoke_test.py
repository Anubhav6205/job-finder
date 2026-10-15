"""Smoke test: 1 term, 1 location, 5 results, output to data_test/."""
import sys
from pathlib import Path

import job_finder as jf

# Override config
jf.SEARCH_TERMS = ["AI Engineer"]
jf.LOCATIONS = ["Hyderabad, India"]
jf.RESULTS_PER_QUERY = 5
jf.DATA = Path("data_test")
jf.SEEN_FILE = jf.DATA / "seen_urls.txt"
jf.OUT_FILE = jf.DATA / "jobs.csv"
# Notion enabled — uses token and db_id from .env

jf.main()
