"""One-off probe: which free-tier Gemini models accept candidate_count>1?

answer_relevancy (RAGAS ResponseRelevancy) with strictness>1 needs a model that
allows multiple candidates in a single call. flash-lite rejects it. This tries a
single candidate_count=3 call per model and reports accept/reject + daily quota note.
Run: python evals/probe_candidates.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / ".env")
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"])

# (model id, free-tier req/day from prior notes — verify, may drift)
MODELS = [
    ("gemini-3.1-flash-lite", "500/day (current judge — expected to REJECT)"),
    ("gemini-2.0-flash", "unknown"),
    ("gemini-2.5-flash", "~25/day"),
    ("gemini-2.5-flash-lite", "unknown"),
    ("gemini-flash-latest", "alias"),
]

for model, note in MODELS:
    try:
        r = client.models.generate_content(
            model=model,
            contents="Reply with one short question.",
            config=types.GenerateContentConfig(candidate_count=3, max_output_tokens=20),
        )
        n = len(r.candidates or [])
        print(f"  {model:28s} ACCEPTS candidate_count=3  (got {n} candidates)  [{note}]")
    except Exception as e:
        msg = str(e).split("\n")[0][:90]
        print(f"  {model:28s} REJECTS / error: {msg}  [{note}]")
