"""Retrieval-only sweep: compare context_precision/recall across MMR settings.

context_precision_with_reference and context_recall depend only on
(question, retrieved_contexts, reference) — NOT the generated answer. So we can
sweep k / lambda_mult cheaply: retrieve per config, score just those two metrics.
No answer generation, no faithfulness/answer_relevancy → minimal quota.

Run: python evals/tune_retrieval.py
"""
import json
import os
import sys
import types
from pathlib import Path

# Same Vertex AI stub shim as run_evals.py — must precede ragas import.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from dod.rag import RAG  # noqa: E402

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
JUDGE_MODEL = os.getenv("DOD_JUDGE_MODEL", "gemini-3.1-flash-lite")

# (k, fetch_k, lambda_mult) — first row is the current baseline config.
# Trimmed to baseline vs one tighter config: context_precision makes one judge
# call PER retrieved chunk, so each extra config / larger k burns the
# rate-limited judge quota fast. k=6 + lambda toward relevance (0.8) is the
# precision bet — fewer, more on-target chunks.
CONFIGS = [
    (12, 40, 0.5),   # baseline
    (6, 40, 0.8),
]


def main() -> None:
    load_dotenv(ROOT / ".env")
    rows = [json.loads(l) for l in EVAL_SET.read_text().splitlines() if l.strip()]
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0))
    run_config = RunConfig(max_workers=2, timeout=300)
    metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]

    print(f"{len(rows)} questions | judge={JUDGE_MODEL} | retrieval-only sweep\n")
    results = []
    for k, fetch_k, lam in CONFIGS:
        rag = RAG(k=k, fetch_k=fetch_k, lambda_mult=lam)
        samples = []
        # retrieve contexts only (no LLM answer generation)
        for r in rows:
            docs = rag.retriever.invoke(r["question"])
            samples.append({
                "user_input": r["question"],
                "retrieved_contexts": [d.page_content for d in docs],
                "response": "",  # unused by context metrics; satisfies schema
                "reference": r["ground_truth"],
            })
        ds = EvaluationDataset.from_list(samples)
        res = evaluate(dataset=ds, metrics=metrics, llm=judge, run_config=run_config)
        df = res.to_pandas()
        prec = float(df["llm_context_precision_with_reference"].mean())
        rec = float(df["context_recall"].mean())
        results.append((k, fetch_k, lam, prec, rec))
        print(f"  k={k:<2} fetch_k={fetch_k} lambda={lam}  ->  "
              f"precision={prec:.3f}  recall={rec:.3f}")

    print("\n=== Summary (baseline first) ===")
    print(f"{'k':>3} {'fetch_k':>7} {'lambda':>6} {'precision':>10} {'recall':>7}")
    for k, fetch_k, lam, prec, rec in results:
        print(f"{k:>3} {fetch_k:>7} {lam:>6} {prec:>10.3f} {rec:>7.3f}")


if __name__ == "__main__":
    main()
