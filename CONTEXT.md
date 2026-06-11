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

## Status (as of 2026-06-11)
- ✅ 30 transcripts fetched (`data/transcripts/`)
- ✅ Time-windowed chunking (`src/dod/chunk.py`, 60s / 10s overlap)
- ✅ Transcripts fully indexed: 1429 chunks in Chroma (`gemini-embedding-001`)
- ✅ RAG query chain (`src/dod/rag.py`) — MMR retrieval (k=12, fetch_k=40,
     lambda_mult=0.5, all tunable), `gemini-3.1-flash-lite`, grounded answers
     with timestamped citations. CLI: `scripts/ask.py`
- ✅ Wikipedia fetched: all 49 directors (`scripts/fetch_wikipedia.py` →
     `data/wikipedia/`), prose-chunked via `chunk_article`
- 🟡 Wikipedia indexing PARTIAL: **1131 of 1737 chunks embedded (33 of 49
     directors**, alphabetical through Oliver Stone). 606 chunks / 16 directors
     remain (Olivia Wilde → Zoë Kravitz, incl. Ridley Scott, Spike Lee, Sean
     Baker, Pedro Almodóvar). `build_index.py` now takes `--max N` to budget the
     daily embed cap; it is idempotent/resumable.
- ✅ RAGAS eval harness (`evals/run_evals.py` + `evals/eval_set.jsonl`, 14 Qs).
     answer_relevancy needs candidate_count>1 → routed to a separate
     `gemini-2.5-flash-lite` judge (`DOD_RELEVANCY_MODEL`); `flash-lite` rejects
     it (see `evals/probe_candidates.py`). Results saved to `evals/results/`.
- ✅ BASELINE eval (06-10, 12 dirs) and POST-INDEX eval (06-11, 33 dirs) both run.
     See "Findings" below. `evals/tune_retrieval.py` is a retrieval-only k/lambda
     sweep (no answer generation).
- ✅ First git commit done (06-11).

## Findings (06-11) — what the 12→33-director re-index bought
Overall (06-10 → 06-11): faithfulness 0.91→0.96, answer_relevancy 0.63→0.64,
context_precision 0.24→0.27, context_recall 0.50→0.57. Bio recall 0.50→0.67
(**1.00 for indexed-only directors** — excluding the not-yet-indexed Ridley
Scott + Spike Lee). So Wikipedia augmentation works: recall + faithfulness up.

Two things diagnosed for ~free (direct retrieval inspection, no judge):
1. **The low context_precision is a JUDGE ARTIFACT, not a retrieval failure.**
   For every indexed director the on-target Wikipedia chunk is retrieved at
   rank #1, yet `flash-lite` scored some (Fincher, DuVernay) at 0.00. The weak
   judge mislabels; retrieval ranking is fine. Don't trust precision *absolutes*.
2. **No single global `k` is optimal.** Bio (single-director) Qs want small k
   (k=6 → 6/6 on-target, cleaner); compare Qs want large k (k=12 covers ~11
   distinct directors, k=6 only ~5 → would tank compare recall). A tuned
   constant can't win both → real fix is query-adaptive k or a reranker, NOT a
   config tweak. **Retrieval config left unchanged.**

NB: RAGAS `context_precision_with_reference` makes one judge call PER retrieved
chunk → ~k calls/question. On free-tier rate limits (15/min) a multi-config
sweep times out. Keep sweeps tiny, or use a paid/higher-limit judge.

## Models (free tier, per-model quotas)
- Generation + retrieval LLM: `gemini-3.1-flash-lite` (500/day, 15/min) — set via
  `DOD_CHAT_MODEL`. (Do NOT use `gemini-3.5-flash`: only 20 req/day free.)
- Eval judge: `DOD_JUDGE_MODEL`, defaults to `gemini-3.1-flash-lite`. Set a
  different model (or OpenAI `gpt-4o-mini` + `OPENAI_API_KEY`) for its own quota
  bucket and to avoid the generator grading itself. (Known weak: it mislabels
  context_precision — see Findings.)
- answer_relevancy judge: `DOD_RELEVANCY_MODEL`, defaults to
  `gemini-2.5-flash-lite` (separate bucket; needs candidate_count>1, which
  `gemini-3.1-flash-lite` rejects). Strictness via `DOD_RELEVANCY_STRICTNESS`.
- Embeddings: `gemini-embedding-001` (1000/day).

## RESUME (plan as of 2026-06-11 EOD): FINISH INDEXING, then final eval
606 wiki chunks (16 directors) remain. The only genuine eval zeros left are
not-yet-indexed directors (Ridley Scott, Spike Lee, + the Anora/Sean Baker
"mixed" Q) — finishing the index is the one real lever left.

Tomorrow (after the ~midnight-PT daily reset), in order:

    cd ~/projects/directors-on-directors
    .venv/bin/python scripts/build_index.py         # 1) finish — 606 < 1000/day cap, fits in one run
    .venv/bin/python evals/run_evals.py             # 2) final 14-Q eval on the COMPLETE index

Quota (per-model daily buckets, reset together): build = ~606 embeds (under the
1000/day cap, so no --max needed). Eval = ~75 embeds + ~210 `gemini-3.1-flash-lite`
calls (500/day) — context_precision alone is ~k calls/question. Comfortable
together. Compare the new `evals/results/*.json` against the 06-11 post-index run.

Then (NOT a config tweak — see Findings): the precision ceiling is architectural.
Future work options, in order of effort: (a) query-adaptive k (small k for
single-director Qs, large k for compare); (b) a cross-encoder reranker over a
larger fetch_k; (c) a stronger eval judge so precision scores are trustworthy.

All steps run LOCALLY (write the local `chroma_db/`). Idempotent/resumable.

## FUTURE DIRECTION (raised 2026-06-11, not yet decided): portfolio front-end
Turn this into a portfolio piece — a question front-end that returns answers.
`RAG.answer()` already returns answer text + structured `sources` (timestamped
YouTube URLs), so the UI is mostly a wrapper; per-question cost is just 1 query
embed + 1 LLM call. Two open decisions:
- Stack: Streamlit (fastest, pure Python, free deploy) vs FastAPI + web UI
  (more eng signal) vs integrate into the existing Astro/Vercel personal site
  (needs the Python RAG hosted as a separate API).
- Hosting/demo: live with own key + rate-limit/cache (clickable; free-tier
  quota risk if viral) vs local + recorded GIF (zero risk, not interactive) vs
  visitors-bring-own-key. Clarify audience + learn-vs-ship before picking.
