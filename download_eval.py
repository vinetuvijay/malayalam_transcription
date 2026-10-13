#!/usr/bin/env python3
"""Save Malayalam FLEURS clips with their transcripts for the evaluation page.

Each saved row has the recording and the Malayalam text for that same clip.
FLEURS is published under CC BY 4.0.
"""

import argparse
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset


def save_split(out: Path, split: str, limit: int) -> int:
    rows = load_dataset("google/fleurs", "ml_in", split=split, streaming=True)
    rows = rows.cast_column("audio", Audio(decode=False))
    manifest = out / "references.tsv"
    written = 0
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("file\ttranscription\traw_transcription\n")
        for index, row in enumerate(rows):
            if index >= limit:
                break
            name = f"{index:04d}.wav"
            _write_audio(out / name, row["audio"]["bytes"])
            transcription = _one_line(row["transcription"])
            raw = _one_line(row.get("raw_transcription") or transcription)
            handle.write(f"{name}\t{transcription}\t{raw}\n")
            print(f"{name}: {transcription}")
            written += 1
    return written


def _one_line(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def _write_audio(path: Path, payload: bytes) -> None:
    samples, sample_rate = sf.read(io.BytesIO(payload))
    sf.write(path, samples, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Number of clips to save")
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--out", type=Path, default=Path("data/fleurs-ml"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    args.out.mkdir(parents=True, exist_ok=True)
    written = save_split(args.out, args.split, args.limit)
    print(f"Wrote {written} clips and references.tsv to {args.out}")


if __name__ == "__main__":
    main()
