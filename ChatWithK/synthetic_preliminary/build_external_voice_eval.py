"""Build public, unlabeled voice-continuation inputs; keep source text off Git.

Inputs: official Positive Psychology Frames test CSV, Real World Worry phase-1
CSV, and OSF think-aloud sentence-level XLSX files. All sampling is deterministic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path


SEED = 20261121
N_UNPAIRED = 835


def source_prefix(text: str) -> str:
    words = list(re.finditer(r"\S+", text))
    if len(words) < 4:
        raise ValueError("source stream too short")
    count = min(len(words) - 1, max(1, math.ceil(0.4 * len(words))))
    return text[: words[count - 1].end()] + " "


def item(identifier: str, dataset: str, role: str, text: str, group: str,
         prompt: str) -> dict:
    return {"id": identifier, "dataset": dataset, "source_role": role,
            "source_text": text, "source_prefix": source_prefix(text),
            "card_id": group, "prompt": prompt}


def build(ppf: Path, worry: Path, transcript_dir: Path) -> list[dict]:
    voice_prompt = "Continue the private inner voice from the words already begun. Output only that voice."
    spoken_prompt = "Continue this person's spoken stream of thought from the words already begun. Output only that voice."
    rows = list(csv.DictReader(ppf.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 835:
        raise ValueError(f"expected 835 Positive Psychology Frames test pairs, found {len(rows)}")
    output = []
    for i, row in enumerate(rows):
        for role, field in (("original", "original_text"), ("reframe", "reframed_text")):
            text = row[field].strip()
            output.append(item(f"ppf-{i:04d}-{role}", "Positive Psychology Frames",
                               role, text, f"ppf-{i:04d}", voice_prompt))
    worry_rows = list(csv.DictReader(worry.open(encoding="utf-8-sig", newline="")))
    candidates = [(i, r["text_long"].strip()) for i, r in enumerate(worry_rows)
                  if r.get("text_long") and len(r["text_long"].split()) >= 30]
    if len(candidates) < N_UNPAIRED:
        raise ValueError("too few long worry records")
    selected = sorted(random.Random(SEED).sample(candidates, N_UNPAIRED))
    for i, text in selected:
        output.append(item(f"worry-{i:04d}", "Real World Worry", "long",
                           text, f"worry-{i:04d}", voice_prompt))
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and openpyxl are needed for OSF XLSX files") from exc
    files = sorted(transcript_dir.glob("sub-*_transcripts.xlsx"))
    if len(files) != 112:
        raise ValueError(f"expected 112 accessible think-aloud participant files, found {len(files)}")
    windows = []
    for path in files:
        frame = pd.read_excel(path)
        if "Transcribed Sentence" not in frame:
            raise ValueError(f"missing transcript column: {path.name}")
        sentences = [str(v).strip() for v in frame["Transcribed Sentence"].dropna()
                     if str(v).strip()]
        # Non-overlapping four-sentence windows avoid near-duplicate inputs.
        for start in range(0, len(sentences) - 3, 4):
            text = " ".join(sentences[start:start + 4])
            if len(text.split()) >= 12:
                windows.append((path.stem, start, text))
    if len(windows) < N_UNPAIRED:
        raise ValueError(f"too few think-aloud windows: {len(windows)}")
    selected_windows = sorted(random.Random(SEED).sample(windows, N_UNPAIRED))
    for participant, start, text in selected_windows:
        identifier = f"think-{participant}-{start:03d}"
        output.append(item(identifier, "Think Aloud", "spoken", text,
                           participant, spoken_prompt))
    ids = [r["id"] for r in output]
    if len(ids) != len(set(ids)) or len(output) != 3340:
        raise ValueError("unexpected or duplicate external input")
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ppf", type=Path, required=True)
    p.add_argument("--worry", type=Path, required=True)
    p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = build(args.ppf, args.worry, args.transcripts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    summary = {"items": len(rows), "sampling_seed": SEED,
               "by_dataset": {name: sum(r["dataset"] == name for r in rows)
                              for name in sorted({r["dataset"] for r in rows})},
               "source_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
