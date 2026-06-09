"""Ask the Directors on Directors chatbot a question.

One-shot:     uv run scripts/ask.py "How does Spike Lee use color?"
Interactive:  uv run scripts/ask.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dod.rag import RAG  # noqa: E402


def show(ans) -> None:
    print("\n" + ans.text + "\n")
    print("Sources:")
    for s in ans.sources:
        if s.source == "wikipedia":
            tag = "wiki "
        else:
            mm, ss = divmod(int(s.start), 60)
            tag = f"{mm:d}:{ss:02d}"
        print(f"  [{s.n}] {tag:>6}  {s.title}\n       {s.url}")


def main() -> None:
    rag = RAG()
    if len(sys.argv) > 1:
        show(rag.answer(" ".join(sys.argv[1:])))
        return
    print("Directors on Directors — ask a question (Ctrl-C to quit)\n")
    try:
        while True:
            q = input("> ").strip()
            if q:
                show(rag.answer(q))
                print()
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
