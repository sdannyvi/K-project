"""
Remove short transcript files from K-texts/ after scraping.

Counts whitespace-separated words in each *.txt (excluding dotfiles like
.completed_urls.txt). Deletes files below a minimum word count and prunes
matching lines from .completed_urls.txt when present.

Run from project root: python clean_k_texts.py [--dry-run]. Adjust threshold in
k_config.MIN_K_TEXT_WORDS or --min-words.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import k_config

logger = logging.getLogger(__name__)


def word_count(text: str) -> int:
    """
    Count whitespace-separated words in a string.

    Args:
        text: Full file contents.

    Returns:
        Non-negative word count.
    """
    s = text.strip()
    if not s:
        return 0
    return len(s.split())


def prune_completed_urls(completed_path: Path, deleted_stems: set[str]) -> None:
    """
    Remove URL lines whose /content/<slug>/ matches a deleted filename stem.

    Args:
        completed_path: Path to .completed_urls.txt.
        deleted_stems: Set of filename stems (no .txt) that were removed.
    """
    if not deleted_stems or not completed_path.is_file():
        return
    raw = completed_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    kept: list[str] = []
    dropped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        m = re.search(r"/content/([^/?#]+)/?", stripped, re.I)
        slug = m.group(1) if m else ""
        if slug in deleted_stems:
            dropped += 1
            continue
        kept.append(line)
    completed_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    if dropped:
        logger.info("Removed %s line(s) from %s", dropped, completed_path.name)


def main() -> None:
    """CLI: list or delete K-texts files with fewer than min_words words."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Delete K-texts/*.txt with fewer than N words (title+metadata+body)."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=k_config.K_TEXTS_DIR,
        help="Directory containing transcript .txt files (default: K-texts/).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=k_config.MIN_K_TEXT_WORDS,
        help=f"Minimum word count to keep (default: {k_config.MIN_K_TEXT_WORDS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be deleted without deleting.",
    )
    args = parser.parse_args()

    text_dir: Path = args.dir
    min_w = args.min_words
    if not text_dir.is_dir():
        raise SystemExit(f"Not a directory: {text_dir}")

    to_remove: list[tuple[Path, int]] = []
    for path in sorted(text_dir.glob("*.txt")):
        if path.name.startswith("."):
            continue
        n = word_count(path.read_text(encoding="utf-8"))
        if n < min_w:
            to_remove.append((path, n))

    if not to_remove:
        logger.info("No files under %s with < %s words.", text_dir, min_w)
        return

    logger.info(
        "Found %s file(s) with < %s words.",
        len(to_remove),
        min_w,
    )
    for p, n in to_remove:
        logger.info("  %s words  %s", n, p.name)

    if args.dry_run:
        logger.info("Dry-run: no files deleted.")
        return

    deleted_stems: set[str] = set()
    for p, _ in to_remove:
        stem = p.stem
        p.unlink()
        deleted_stems.add(stem)
        logger.info("Deleted %s", p.name)

    prune_completed_urls(k_config.SCRAPE_COMPLETED_URLS_FILE, deleted_stems)
    logger.info("Done. Removed %s file(s).", len(to_remove))


if __name__ == "__main__":
    main()
