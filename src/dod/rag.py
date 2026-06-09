"""Retrieval-augmented answering over the Directors on Directors index.

Loads the persisted Chroma store, retrieves the most relevant transcript
chunks for a question, and asks Gemini to answer grounded in them — citing
each director/video with a timestamped YouTube link.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

ROOT = Path(__file__).parent.parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "directors_on_directors"
EMBED_MODEL = "models/gemini-embedding-001"
# Free-tier note: newer flagship models are the most rate-limited (3.5-flash is
# 20 req/day free). flash-lite gives 500/day, plenty for querying + evals.
CHAT_MODEL = os.getenv("DOD_CHAT_MODEL", "gemini-3.1-flash-lite")

SYSTEM_PROMPT = """You are a knowledgeable film buff answering questions about \
how directors approach filmmaking, drawn from the "Directors on Directors" \
interview series plus Wikipedia articles on the directors.

Answer using ONLY the excerpts provided in the context. Each is labelled with a \
[source N] tag indicating its origin:
- "(interview, at M:SS)" — the director's own words from the video series. Best \
for their opinions, philosophy, and how they describe their process.
- "Wikipedia — ..." — biographical/factual background. Best for career facts, \
filmographies, and context.

Rules:
- Ground every claim in the excerpts. If the context doesn't cover the question, \
say so plainly — do not invent answers from general knowledge.
- Attribute opinions to the specific director who expressed them; prefer the \
director's own interview words over Wikipedia when describing their views.
- Cite the excerpts you used inline like [source 2].
- Be conversational but concise.
"""


@dataclass
class Source:
    n: int
    title: str
    url: str
    source: str           # "youtube" | "wikipedia"
    start: float = 0.0    # only meaningful for youtube


@dataclass
class Answer:
    text: str
    sources: list[Source]
    contexts: list[str]   # raw retrieved chunk texts (for eval / inspection)


def _label(m: dict) -> str:
    """Human-readable provenance for a chunk, by source type."""
    if m.get("source") == "wikipedia":
        return f"Wikipedia — {m.get('title', '?')}"
    mm, ss = divmod(int(m.get("start", 0)), 60)
    return f"{m.get('title', '?')} (interview, at {mm:d}:{ss:02d})"


def _format_context(docs: list[Document]) -> tuple[str, list[Source]]:
    blocks: list[str] = []
    sources: list[Source] = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        blocks.append(f"[source {i}] {_label(m)}\n{d.page_content}")
        sources.append(Source(
            n=i, title=m.get("title", "?"), url=m.get("url", ""),
            source=m.get("source", "youtube"), start=m.get("start", 0.0),
        ))
    return "\n\n".join(blocks), sources


class RAG:
    def __init__(self, k: int = 12, fetch_k: int = 40):
        load_dotenv(ROOT / ".env")
        self.embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embedder,
            persist_directory=str(CHROMA_DIR),
        )
        # MMR diversifies across videos/directors instead of returning
        # near-duplicate chunks from whichever one video matches hardest —
        # important for "which directors are similar in X" comparisons.
        self.retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)

    def answer(self, question: str) -> Answer:
        docs = self.retriever.invoke(question)
        context, sources = _format_context(docs)
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        resp = self.llm.invoke(messages)
        return Answer(
            text=_as_text(resp.content),
            sources=sources,
            contexts=[d.page_content for d in docs],
        )


def _as_text(content) -> str:
    """Gemini may return content as a string or a list of content parts."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and "text" in p:
            parts.append(p["text"])
    return "".join(parts).strip()
