"""Embed chunked transcripts into Chroma using Gemini embeddings.

Idempotent + resumable + throttled. Free tier is 100 embed req/min — we pace
one request at a time with a small sleep and checkpoint to Chroma periodically.
"""
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dod.chunk import chunk_article, chunk_segments  # noqa: E402

TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
WIKIPEDIA_DIR = ROOT / "data" / "wikipedia"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "directors_on_directors"
EMBED_MODEL = "models/gemini-embedding-001"

RPM_LIMIT = 90  # under free-tier 100/min
SLEEP_PER_REQ = 60.0 / RPM_LIMIT
CHECKPOINT_EVERY = 25


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_transcript_documents() -> tuple[list[Document], list[str]]:
    docs: list[Document] = []
    ids: list[str] = []
    for f in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        for c in chunk_segments(data["segments"]):
            chunk_id = f"{data['video_id']}_{int(c.start)}"
            url = f"{data['url']}&t={int(c.start)}s"
            docs.append(Document(
                page_content=c.text,
                metadata={
                    "video_id": data["video_id"],
                    "title": data["title"],
                    "url": url,
                    "start": c.start,
                    "end": c.end,
                    "source": "youtube",
                },
            ))
            ids.append(chunk_id)
    return docs, ids


def build_wikipedia_documents() -> tuple[list[Document], list[str]]:
    docs: list[Document] = []
    ids: list[str] = []
    for f in sorted(WIKIPEDIA_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        slug = _slug(data["name"])
        for i, text in enumerate(chunk_article(data["content"])):
            docs.append(Document(
                page_content=text,
                metadata={
                    "title": data["title"],
                    "url": data["url"],
                    "director": data["name"],
                    "source": "wikipedia",
                },
            ))
            ids.append(f"wiki_{slug}_{i}")
    return docs, ids


def build_documents() -> tuple[list[Document], list[str]]:
    t_docs, t_ids = build_transcript_documents()
    w_docs, w_ids = build_wikipedia_documents()
    print(f"  transcripts: {len(t_docs)} chunks | wikipedia: {len(w_docs)} chunks")
    return t_docs + w_docs, t_ids + w_ids


def checkpoint(collection, docs, ids, embeddings) -> None:
    if not docs:
        return
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedder,
        persist_directory=str(CHROMA_DIR),
    )
    collection = store._collection

    docs, ids = build_documents()
    print(f"Built {len(docs)} candidate chunks")
    existing = set(collection.get(ids=ids)["ids"])
    pending = [(d, i) for d, i in zip(docs, ids) if i not in existing]
    print(f"Already indexed: {len(existing)}. To embed: {len(pending)}")
    if not pending:
        return

    eta_min = len(pending) * SLEEP_PER_REQ / 60
    print(f"ETA at {RPM_LIMIT} req/min: ~{eta_min:.1f} min\n")

    buf_docs: list[Document] = []
    buf_ids: list[str] = []
    buf_embs: list[list[float]] = []
    done = 0
    t_start = time.monotonic()

    for d, i in pending:
        t0 = time.monotonic()
        try:
            emb = embedder.embed_query(d.page_content)
        except Exception as e:
            print(f"  [retry after 65s] {i}: {type(e).__name__}")
            checkpoint(collection, buf_docs, buf_ids, buf_embs)
            buf_docs.clear(); buf_ids.clear(); buf_embs.clear()
            time.sleep(65)
            emb = embedder.embed_query(d.page_content)
        buf_docs.append(d); buf_ids.append(i); buf_embs.append(emb)

        if len(buf_docs) >= CHECKPOINT_EVERY:
            checkpoint(collection, buf_docs, buf_ids, buf_embs)
            done += len(buf_docs)
            elapsed = time.monotonic() - t_start
            rate = done / elapsed * 60
            print(f"  {done}/{len(pending)} ({elapsed:.0f}s, {rate:.0f} rpm)")
            buf_docs.clear(); buf_ids.clear(); buf_embs.clear()

        dt = time.monotonic() - t0
        if dt < SLEEP_PER_REQ:
            time.sleep(SLEEP_PER_REQ - dt)

    checkpoint(collection, buf_docs, buf_ids, buf_embs)
    done += len(buf_docs)
    print(f"\nDone. Collection now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
