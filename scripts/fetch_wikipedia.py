"""Fetch Wikipedia articles for the directors featured in the series.

Resolves each name to its English Wikipedia page (handling disambiguation),
and saves the plain-text article to data/wikipedia/<slug>.json. Idempotent:
skips names already saved. Prints a resolution report so wrong matches are
easy to eyeball.

    uv run scripts/fetch_wikipedia.py
"""
import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import wikipedia

# Wikipedia's API returns an empty/HTML body (which the lib then fails to
# JSON-decode) when hit too fast. Throttle to stay on its good side. Both
# page resolution AND .content trigger separate API calls.
wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=1))

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "wikipedia"

# Every filmmaker who appears as a participant across the 30 videos (deduped).
# A few need an explicit page title to dodge disambiguation / wrong matches.
DIRECTORS = [
    "Spike Lee", "Reinaldo Marcus Green", "Oliver Stone",
    ("Sean Baker", "Sean Baker (filmmaker)"), "Brady Corbet",
    "Guillermo del Toro", "Jane Campion", "Ari Aster", "Yorgos Lanthimos",
    "Denis Villeneuve", "Luca Guadagnino", "Emerald Fennell", "Olivia Wilde",
    "Rian Johnson", "Joseph Kosinski", "Jon M. Chu", "Shawn Levy",
    "Ridley Scott", "Bradley Cooper", "Zoë Kravitz", "Matt Reeves",
    "James Cameron", "Robert Rodriguez", "Kristen Stewart", "Jesse Eisenberg",
    "Ben Affleck", ("Michael B. Jordan", "Michael B. Jordan"), "Ryan Coogler",
    "Adam McKay", ("Joe Wright", "Joe Wright"), "Pedro Almodóvar",
    "Halina Reijn", "Josh Safdie", "Chloé Zhao", "Greta Gerwig",
    "Judd Apatow", "Jason Bateman", "Gina Prince-Bythewood", "Regina King",
    "Melina Matsoukas", "Ava DuVernay", ("Michael Mann", "Michael Mann"),
    "David Fincher", "Tyler Perry", "Chinonye Chukwu", "Sarah Polley",
    "Francis Ford Coppola", "Taylor Swift", "Martin McDonagh",
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fetch_article(name: str, page_hint: str | None) -> dict:
    # Both resolution and .content hit the API and can be rate-limited
    # (surfaces as a JSONDecodeError, a ValueError subclass). Retry the
    # whole thing so a throttled .content read also gets another shot.
    for attempt in range(5):
        try:
            page = _resolve_once(name, page_hint)
            content = page.content  # triggers its own API call
            return {"name": name, "title": page.title,
                    "url": page.url, "content": content}
        except ValueError:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _resolve_once(name: str, page_hint: str | None) -> wikipedia.WikipediaPage:
    target = page_hint or name
    try:
        return wikipedia.page(target, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        # prefer an option that looks like a filmmaker
        for opt in e.options:
            if any(w in opt.lower() for w in ("director", "filmmaker", "film")):
                return wikipedia.page(opt, auto_suggest=False)
        return wikipedia.page(e.options[0], auto_suggest=False)
    except wikipedia.PageError:
        hits = wikipedia.search(name)
        if not hits:
            raise
        return wikipedia.page(hits[0], auto_suggest=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for entry in DIRECTORS:
        name, hint = entry if isinstance(entry, tuple) else (entry, None)
        out = OUT_DIR / f"{slug(name)}.json"
        if out.exists():
            print(f"  skip (cached)  {name}")
            continue
        try:
            art = fetch_article(name, hint)
        except Exception as e:
            print(f"  FAIL           {name}: {type(e).__name__}: {e}")
            failures.append(name)
            continue
        out.write_text(json.dumps(art, ensure_ascii=False, indent=2))
        flag = "" if slug(art["title"]) == slug(name) else "  <-- check match"
        print(f"  ok  {name:24s} -> {art['title']} ({len(art['content'])} chars){flag}")

    print(f"\nSaved {len(list(OUT_DIR.glob('*.json')))} articles.")
    if failures:
        print(f"Failed: {failures}", file=sys.stderr)


if __name__ == "__main__":
    main()
