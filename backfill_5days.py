"""
One-time backfill: trash all existing Notion Job Finder pages,
then re-run job_finder for the last 5 days with updated logic.
Run once manually. Do not add to automation.
"""
import requests
import time
import sys
from dotenv import load_dotenv
load_dotenv(override=True)

import job_finder as jf

NOTION_TOKEN = jf.NOTION_TOKEN
NOTION_DB_ID = jf.NOTION_DB_ID
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def trash_all_pages():
    print("Trashing existing Notion pages...")
    trashed = 0
    cursor = None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        resp = requests.post(
            f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query",
            headers=HEADERS,
            json=body,
            timeout=30,
        )
        data = resp.json()
        pages = data.get("results", [])
        if not pages:
            break
        for page in pages:
            requests.patch(
                f"https://api.notion.com/v1/pages/{page['id']}",
                headers=HEADERS,
                json={"archived": True},
                timeout=10,
            )
            trashed += 1
            time.sleep(0.1)  # respect rate limits
        print(f"  Trashed {trashed} pages so far...")
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    print(f"Done. {trashed} pages trashed.\n")


def main():
    # Step 1: trash existing pages
    trash_all_pages()

    # Step 2: override config for 5-day window, skip seen_urls
    jf.HOURS_OLD = 120       # 5 days
    jf.SEEN_FILE.parent.mkdir(exist_ok=True)
    # temporarily back up and clear seen_urls
    seen_backup = None
    if jf.SEEN_FILE.exists():
        seen_backup = jf.SEEN_FILE.read_text()
        jf.SEEN_FILE.write_text("")
    print("Running 5-day job fetch with updated logic...\n")
    try:
        jf.main()
    finally:
        # restore seen_urls to the new one (jf.main appended to it)
        # no restore needed — the new seen list IS correct
        pass

    print("\nBackfill complete. From tomorrow, automation runs as usual with HOURS_OLD=48.")


if __name__ == "__main__":
    main()
