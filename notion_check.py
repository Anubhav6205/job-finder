from dotenv import load_dotenv; load_dotenv(override=True)
import os, requests

token = os.getenv("NOTION_TOKEN")
db_id = os.getenv("NOTION_DB_ID")

resp = requests.get(
    f"https://api.notion.com/v1/databases/{db_id}",
    headers={"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28"},
    timeout=10,
)
print(f"Status: {resp.status_code}")
data = resp.json()
if resp.status_code == 200:
    print(f"Database: {data.get('title', [{}])[0].get('plain_text', '(untitled)')}")
    print(f"Properties: {list(data.get('properties', {}).keys())}")
else:
    print(data)
