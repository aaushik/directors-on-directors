# Directors on Directors — RAG Chatbot

## What this is
A RAG chatbot over YouTube's "Directors on Directors" series (~30 videos), augmented with Wikipedia articles about the directors and their films. Answers questions about different directors' approaches to filmmaking.

## Stack
- LangChain
- ChromaDB (vector store)
- Gemini embeddings + Gemini LLM (via AI Studio free tier)
- youtube-transcript-api (for transcripts)
- RAGAS (for evals)

## Goals
- Learn LangChain hands-on
- Learn how to do evals with RAGAS
- Build something actually useful/fun

## Status
- ✅ 30 transcripts fetched (`data/transcripts/`)
- ✅ Time-windowed chunking (`src/dod/chunk.py`, 60s / 10s overlap)
- ✅ Transcripts fully indexed: 1429 chunks in Chroma (`gemini-embedding-001`)
- ✅ RAG query chain (`src/dod/rag.py`) — MMR retrieval (k=12), gemini-3.5-flash,
     grounded answers with timestamped citations. CLI: `scripts/ask.py`
- ✅ Wikipedia fetched: all 49 directors (`scripts/fetch_wikipedia.py` →
     `data/wikipedia/`), prose-chunked via `chunk_article`
- 🟡 Wikipedia indexing PARTIAL: 481 of 1737 chunks embedded (12 of 49 directors).
     Hit Gemini free-tier **daily** embed cap (1000 req/day). 1256 chunks remain.
- ✅ RAGAS eval harness (`evals/run_evals.py` + `evals/eval_set.jsonl`, 14 Qs).
     Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
     NOT YET RUN — waiting on the index to finish (see below). `--dry` validated.
- ⬜ No git commits yet

## Models (free tier, per-model quotas)
- Generation + retrieval LLM: `gemini-3.1-flash-lite` (500/day, 15/min) — set via
  `DOD_CHAT_MODEL`. (Do NOT use `gemini-3.5-flash`: only 20 req/day free.)
- Eval judge: `DOD_JUDGE_MODEL`, defaults to `gemini-3.1-flash-lite`. Set a
  different model (or OpenAI `gpt-4o-mini` + `OPENAI_API_KEY`) for its own quota
  bucket and to avoid the generator grading itself.
- Embeddings: `gemini-embedding-001` (1000/day).

## RESUME: finish the Wikipedia index (quota-blocked)
Free tier = 1000 embed requests/day. To finish the remaining 1256 wiki chunks,
re-run the build on the next two days (after the daily reset, ~midnight PT):

    .venv/bin/python scripts/build_index.py   # tomorrow: ~1000 chunks
    .venv/bin/python scripts/build_index.py   # day after: last ~256 → done

It's idempotent/resumable (skips already-embedded ids). Must run locally (writes
the local `chroma_db/`). Do NOT query during build days — each query spends one
embed from the same daily pool. After completion, queries are ~1 embed each.

Once the index is complete (day 3), run the evals to get a baseline:

    .venv/bin/python evals/run_evals.py --limit 3   # cheap sanity check first
    .venv/bin/python evals/run_evals.py             # full 14-question run

The `bio` questions in the eval set target directors whose Wikipedia ISN'T
embedded yet (Greta Gerwig, Ridley Scott, Spike Lee, ...) — their context_recall
should jump once the index is finished, quantifying what the augmentation buys.
