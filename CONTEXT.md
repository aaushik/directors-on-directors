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
- ✅ Wikipedia indexing COMPLETE (06-12): **1737/1737 chunks, all 49 directors**.
     Full index = 3166 chunks (1429 transcript + 1737 wiki). `build_index.py`
     takes `--max N` to budget the daily embed cap; idempotent/resumable.
- ✅ RAGAS eval harness (`evals/run_evals.py` + `evals/eval_set.jsonl`, 14 Qs).
     answer_relevancy needs candidate_count>1 → routed to a separate
     `gemini-2.5-flash-lite` judge (`DOD_RELEVANCY_MODEL`); `flash-lite` rejects
     it (see `evals/probe_candidates.py`). Results saved to `evals/results/`.
- ✅ BASELINE eval (06-10, 12 dirs) and POST-INDEX eval (06-11, 33 dirs) both run.
     See "Findings" below. `evals/tune_retrieval.py` is a retrieval-only k/lambda
     sweep (no answer generation).
- ✅ First git commit done (06-11).

## Findings — what the Wikipedia augmentation bought (12 → 33 → 49 dirs)
Overall across the three index states (06-10 / 06-11 / 06-12-complete):
faithfulness 0.91 → 0.96 → 0.98; answer_relevancy 0.63 → 0.64 → 0.81;
context_precision 0.24 → 0.27 → 0.48; context_recall 0.50 → 0.57 → 0.75.
Bio is the headline: precision 0.04 → 0.13 → 0.46 (~10×), recall 0.50 → 0.67 →
1.00. The complete index delivered big gains everywhere it could.
EXCEPTION: `compare` recall stuck at 0.33 across all three — breadth questions
("which directors prefer X") where the reference names several directors but
MMR k=12 surfaces only a fraction. NOT an indexing problem (see #2 below);
needs query-adaptive k or a reranker. (n is tiny + flash-lite judge noisy →
trust the trend, not absolutes.)

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

## RESUME (as of 2026-06-12): index + eval arc DONE. Next: portfolio front-end.
Indexing complete (49/49 dirs), final eval run (`results/20260612T140512Z.json`).
The RAG + evals milestone is finished. Next direction is the portfolio front-end
(see below). Optional later retrieval work (NOT a config tweak — see Findings),
in order of effort: (a) query-adaptive k (small k for single-director Qs, large
k for compare — would lift the stuck compare recall); (b) a cross-encoder
reranker over a larger fetch_k; (c) a stronger eval judge so precision scores
are trustworthy.

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
