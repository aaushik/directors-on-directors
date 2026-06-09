"""Fetch video list from the Variety DoD playlist + transcripts for each video.

Idempotent: skips videos already saved in data/transcripts/.
"""
import json
from pathlib import Path

from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

PLAYLIST_URL = "https://youtube.com/playlist?list=PLLx_Nt-I7ViqK8k-H33Jr12lumPc8xl2-"
OUT_DIR = Path(__file__).parent.parent / "data" / "transcripts"


def list_playlist(url: str) -> list[dict]:
    opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return [{"id": e["id"], "title": e.get("title", "")} for e in info["entries"] if e]


def fetch_transcript(video_id: str) -> list[dict] | None:
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    except (TranscriptsDisabled, NoTranscriptFound):
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = list_playlist(PLAYLIST_URL)
    print(f"Found {len(videos)} videos in playlist")

    new, skipped, failed = 0, 0, 0
    for v in videos:
        out_path = OUT_DIR / f"{v['id']}.json"
        if out_path.exists():
            skipped += 1
            continue
        segments = fetch_transcript(v["id"])
        if segments is None:
            print(f"  [no transcript] {v['title']}")
            failed += 1
            continue
        out_path.write_text(json.dumps({
            "video_id": v["id"],
            "title": v["title"],
            "url": f"https://youtube.com/watch?v={v['id']}",
            "segments": segments,
        }, indent=2))
        print(f"  [saved] {v['title']} ({len(segments)} segments)")
        new += 1

    print(f"\nDone. new={new} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
