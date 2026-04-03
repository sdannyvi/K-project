"""
K-augmented LLM demo: build or load the RAG index and ask a default question.

Runs the vertical slice: local embeddings over K-texts, retrieval, then one
chat completion via the configured Azure/OpenAI deployment. Pass --question
to override the default smoke-test prompt.

Update when CLI flags or default prompt text should change.
"""

from __future__ import annotations

import argparse
import logging
import sys

import k_config
from rag import INDEX_CHUNKS_NAME, answer_with_rag, build_and_save_index

logger = logging.getLogger(__name__)

DEFAULT_QUESTION: str = (
    "Removing all conditioning will lead to completely automatic actions in the world? "
    "Is that what is meant? To return to being almost an animal on the savannah.\n\n"
    "Suppose that I have reached such a state. Krishnamurti claims that then intelligence "
    "(according to how he defines it) flows through you. What does he mean?"
)


def main() -> None:
    """Build index if needed, run RAG + chat once, print the reply."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="K-augmented LLM RAG demo")
    parser.add_argument(
        "--question",
        "-q",
        default=DEFAULT_QUESTION,
        help="User question for the model (default: spiritual path prompt).",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the embedding index from K-texts/ even if an index exists.",
    )
    args = parser.parse_args()

    k_dir = k_config.K_TEXTS_DIR
    idx_dir = k_config.K_INDEX_DIR
    txts = list(k_dir.glob("*.txt"))
    if not txts:
        logger.error(
            "No transcript .txt files in %s. Run: python scrape_jk.py (requires Playwright).",
            k_dir,
        )
        sys.exit(1)

    index_chunks = idx_dir / INDEX_CHUNKS_NAME
    if args.rebuild_index or not index_chunks.is_file():
        n = build_and_save_index(k_dir, idx_dir)
        logger.info("Indexed %s chunks.", n)
    else:
        logger.info("Using existing index under %s", idx_dir)

    answer = answer_with_rag(args.question)
    print(answer)


if __name__ == "__main__":
    main()
