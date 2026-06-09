"""Chunking for the two corpora.

Transcripts are chunked by time window (`chunk_segments`); Wikipedia prose is
chunked by character length on natural boundaries (`chunk_article`). Both aim
for a similar retrieval granularity with a small overlap so an idea spanning a
boundary still surfaces.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start: float
    end: float


def chunk_segments(
    segments: list[dict],
    window_seconds: float = 60.0,
    overlap_seconds: float = 10.0,
) -> list[Chunk]:
    if not segments:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(segments):
        window_start = segments[i]["start"]
        texts: list[str] = []
        j = i
        while j < len(segments):
            seg_end = segments[j]["start"] + segments[j]["duration"]
            if seg_end - window_start > window_seconds and texts:
                break
            texts.append(segments[j]["text"])
            j += 1
        window_end = segments[j - 1]["start"] + segments[j - 1]["duration"]
        chunks.append(Chunk(text=" ".join(texts).strip(), start=window_start, end=window_end))

        if j >= len(segments):
            break
        # advance i so next window starts ~overlap_seconds before the previous end
        target = window_end - overlap_seconds
        next_i = j
        while next_i > i + 1 and segments[next_i - 1]["start"] >= target:
            next_i -= 1
        i = next_i

    return chunks


def chunk_article(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split Wikipedia prose into overlapping character chunks.

    Splits on paragraph/sentence boundaries first (via the recursive splitter)
    so chunks stay readable. Drops Wikipedia's trailing reference sections,
    which are noise for retrieval.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cuts = [text.find(f"== {h} ==") for h in
            ("References", "External links", "See also", "Notes", "Footnotes")]
    cuts = [i for i in cuts if i != -1]
    if cuts:
        text = text[:min(cuts)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]
