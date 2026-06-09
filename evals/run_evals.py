"""Evaluate the RAG with RAGAS.

For each question in evals/eval_set.jsonl we run the RAG to get an answer plus
its retrieved contexts, then score the batch with four RAGAS metrics:

  - faithfulness            : are the answer's claims grounded in the contexts?
  - answer_relevancy        : does the answer actually address the question?
  - context_precision       : are the retrieved chunks relevant / well-ranked?
  - context_recall          : did retrieval fetch what the reference answer needs?

Quota notes (free tier):
  - generation + judge run on gemini-3.1-flash-lite (500 req/day, 15 req/min).
    Override the judge with DOD_JUDGE_MODEL (a *different* model gets its own
    daily bucket and avoids the model grading its own output).
  - embeddings (retrieval + answer_relevancy) use gemini-embedding-001 (the
    1000/day bucket) — a ~15-question run spends well under 100.
  - max_workers is kept low to respect the 15 req/min ceiling; the run is slow
    by design.

    uv run evals/run_evals.py            # full eval set
    uv run evals/run_evals.py --limit 3  # quick subset
    uv run evals/run_evals.py --dry      # build everything, make no API calls
"""
import argparse
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

# ragas 0.4.3 hard-imports a Vertex AI path that langchain-community 0.4.x
# removed. We don't use Vertex AI, so satisfy the import with a stub before
# ragas loads. Must precede the ragas import below.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from langchain_google_genai import (  # noqa: E402
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from dod.rag import RAG  # noqa: E402

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
JUDGE_MODEL = os.getenv("DOD_JUDGE_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = "models/gemini-embedding-001"


def load_eval_set(limit: int | None) -> list[dict]:
    rows = [json.loads(ln) for ln in EVAL_SET.read_text().splitlines() if ln.strip()]
    return rows[:limit] if limit else rows


def build_dataset(rag: RAG, rows: list[dict]) -> EvaluationDataset:
    samples = []
    for i, r in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {r['question'][:60]}...")
        ans = rag.answer(r["question"])
        samples.append({
            "user_input": r["question"],
            "retrieved_contexts": ans.contexts,
            "response": ans.text,
            "reference": r["ground_truth"],
        })
    return EvaluationDataset.from_list(samples)


def summarise(df, rows: list[dict]) -> dict:
    metric_cols = [c for c in df.select_dtypes("number").columns]
    overall = {c: round(float(df[c].mean()), 3) for c in metric_cols}
    df = df.copy()
    df["kind"] = [r.get("kind", "?") for r in rows]
    by_kind = {
        kind: {c: round(float(g[c].mean()), 3) for c in metric_cols}
        for kind, g in df.groupby("kind")
    }
    return {"overall": overall, "by_kind": by_kind, "metric_cols": metric_cols}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    ap.add_argument("--dry", action="store_true", help="build everything, no API calls")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    rows = load_eval_set(args.limit)
    print(f"Eval set: {len(rows)} questions | judge={JUDGE_MODEL}")

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=EMBED_MODEL))

    if args.dry:
        print("[dry] constructed RAG, judge, embeddings, metrics — no API calls made.")
        print("[dry] metrics:", [type(m).__name__ for m in metrics])
        RAG()  # verify the store loads
        print("[dry] Chroma store loaded OK. Ready for a real run.")
        return

    rag = RAG()
    print("Generating answers (1 embed + 1 LLM call each)...")
    dataset = build_dataset(rag, rows)

    # Low concurrency to stay under the 15 req/min free-tier ceiling.
    run_config = RunConfig(max_workers=2, timeout=180)
    print(f"Scoring with RAGAS (call-heavy; throttled to ~{run_config.max_workers} workers)...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge,
                      embeddings=embeddings, run_config=run_config)

    df = result.to_pandas()
    summary = summarise(df, rows)

    print("\n=== Overall ===")
    for k, v in summary["overall"].items():
        print(f"  {k:34s} {v:.3f}")
    print("\n=== By question kind ===")
    for kind, scores in summary["by_kind"].items():
        print(f"  {kind}:")
        for k, v in scores.items():
            print(f"    {k:32s} {v:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{ts}.json"
    out.write_text(json.dumps({
        "timestamp": ts,
        "judge_model": JUDGE_MODEL,
        "n": len(rows),
        "summary": summary,
        "per_question": df.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
