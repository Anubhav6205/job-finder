"""One-shot: send a real JD to gpt-5.4-mini and print token usage."""
from dotenv import load_dotenv; load_dotenv(override=True)
from openai import OpenAI
import json, sys

client = OpenAI()

PROMPT = """You read job descriptions like a careful recruiter.
Extract the hiring constraints from the JD below.
Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
{"min_yoe": number|null, "max_yoe": number|null, "yoe_quote": string|null,
 "max_joining_days": number|null, "joining_quote": string|null}
"""

JD = """
We are looking for an AI Engineer to join our team in Hyderabad.
Requirements: 2-4 years of experience in machine learning or AI development.
Strong knowledge of Python, LLMs, and prompt engineering required.
Immediate joiners preferred; notice period up to 30 days acceptable.
"""

try:
    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=2000,
        reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": f"{PROMPT}\nTITLE: AI Engineer\nCOMPANY: TestCo\n\nJD:\n{JD}"}]
    )
    print("Response:", resp.choices[0].message.content)
    u = resp.usage
    print(f"Tokens  — prompt: {u.prompt_tokens}  completion: {u.completion_tokens}  total: {u.total_tokens}")
except Exception as e:
    print("FAIL:", str(e)[:300])
