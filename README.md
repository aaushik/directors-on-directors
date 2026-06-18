# Directors on Directors — RAG Chatbot

A retrieval-augmented chatbot that answers questions about how famous film
directors think about their craft. It's built over the transcripts of Variety's
[*Directors on Directors*](https://www.youtube.com/playlist?list=PLqh9Js9wnQGEPpZG4__zQz0e9c6JYpYbi)
interview series (~30 conversations), augmented with Wikipedia articles for all
49 directors featured. Ask it things like *"How does Christopher Nolan think
about practical effects?"* or *"Which directors prefer shooting on film?"* and it
answers from the source material, with timestamped citations back to the videos.

> Built as a hands-on project to learn LangChain and RAG evaluation with RAGAS.

## How it works

```
YouTube transcripts ─┐
                     ├─► chunk ─► Gemini embeddings ─► ChromaDB ─► MMR retrieval ─► Gemini LLM ─► grounded answer + citations
Wikipedia articles ──┘
```

- **Ingestion** — transcripts via `youtube-transcript-api`; Wikipedia prose via
  the `wikipedia` package.
- **Chunking** — time-windowed for transcripts (60s windows, 10s overlap, so
  every chunk maps back to a clickable timestamp); prose-chunked for Wikipedia.
- **Vector store** — ChromaDB with `gemini-embedding-001`. Full index is **3,166
  chunks** (1,429 transcript + 1,737 Wikipedia, all 49 directors).
- **Retrieval + generation** — MMR retrieval (k=12, fetch_k=40, λ=0.5, all
  tunable) feeding `gemini-3.1-flash-lite` with a grounded-answer prompt that
  emits timestamped source citations.
- **Evaluation** — a RAGAS harness (14-question eval set) scoring faithfulness,
  answer relevancy, context precision, and context recall.

## Results

Wikipedia augmentation was measured by re-running the eval set as the index grew
from 12 → 33 → 49 directors:

| Metric            | 12 dirs | 33 dirs | 49 dirs (full) |
|-------------------|:-------:|:-------:|:--------------:|
| Faithfulness      | 0.91    | 0.96    | **0.98**       |
| Answer relevancy  | 0.63    | 0.64    | **0.81**       |
| Context precision | 0.24    | 0.27    | **0.48**       |
| Context recall    | 0.50    | 0.57    | **0.75**       |

Biographical questions saw the biggest lift (precision 0.04 → 0.46, recall
0.50 → 1.00). Two findings came out of inspecting retrieval directly rather than
trusting the judge:

1. **Low context-precision scores are largely a weak-judge artifact**, not a
   retrieval failure — the on-target chunk is retrieved at rank #1 even on
   questions the `flash-lite` judge scored 0.00.
2. **No single global `k` is optimal** — single-director questions want a small
   `k`; "which directors prefer X" breadth questions want a large `k`. The real
   fix is query-adaptive `k` or a reranker, not a config tweak.

## Stack

LangChain · ChromaDB · Gemini (embeddings + LLM, via AI Studio free tier) ·
youtube-transcript-api · RAGAS · Python 3.12 · [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
# 1. Install deps (uv reads pyproject.toml + uv.lock)
uv sync

# 2. Configure your key
cp .env.example .env        # then add your Google AI Studio key

# 3. Fetch source data
uv run scripts/fetch_transcripts.py
uv run scripts/fetch_wikipedia.py

# 4. Build the vector index (idempotent + resumable; --max N to budget the daily embed cap)
uv run scripts/build_index.py

# 5. Ask away
uv run scripts/ask.py "How does Christopher Nolan think about practical effects?"
uv run scripts/ask.py            # or run with no args for an interactive prompt
```

### Run the evals

```bash
uv run evals/run_evals.py            # full eval set; results land in evals/results/
uv run evals/run_evals.py --limit 3  # first 3 questions only
uv run evals/run_evals.py --dry      # build everything, no API calls
```

## Configuration

Set in `.env`. The vector store, transcripts, and Wikipedia data live locally and
are gitignored.

| Variable                   | Purpose                                            | Default                   |
|----------------------------|----------------------------------------------------|---------------------------|
| `GOOGLE_API_KEY`           | Google AI Studio key (embeddings + LLM)            | _required_                |
| `DOD_CHAT_MODEL`           | Generation + retrieval LLM                         | `gemini-3.1-flash-lite`   |
| `DOD_JUDGE_MODEL`          | RAGAS eval judge (set a different model for its own quota bucket) | `gemini-3.1-flash-lite` |
| `DOD_RELEVANCY_MODEL`      | answer_relevancy judge (needs `candidate_count>1`) | `gemini-2.5-flash-lite`   |

> All models run on the AI Studio **free tier** with per-model daily/minute
> quotas — index building and evals are written to be resumable so you can stay
> under the embed cap.

## Project layout

```
src/dod/          # library: chunking (chunk.py) + RAG chain (rag.py)
scripts/          # CLI entry points: fetch_*, build_index, ask
evals/            # RAGAS harness, eval set, and saved results/
data/             # fetched transcripts + wikipedia (gitignored)
chroma_db/        # vector store (gitignored)
```

## Roadmap

The RAG + evaluation milestone is complete. Possible next steps, in order of
effort: (a) query-adaptive `k` to lift recall on breadth/comparison questions;
(b) a cross-encoder reranker over a larger `fetch_k`; (c) a stronger eval judge
so precision scores are trustworthy; (d) a front-end to make it a clickable demo.
