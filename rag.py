"""
RAG over K-texts: chunking, embedding index, retrieval, and GPT-4o answering.

Builds a local numpy embedding matrix from plain-text transcripts under K-texts/,
persists metadata alongside, and at query time retrieves top-k chunks and sends
them with the user question to the configured chat deployment.

Update when chunking strategy, embedding model, or prompt contract changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

import k_config
from llm_client import (
    get_deployment,
    get_embedding_deployment,
    get_max_tokens,
    initialize_client,
)

logger = logging.getLogger(__name__)

INDEX_CHUNKS_NAME: str = "chunks.jsonl"
INDEX_EMBEDDINGS_NAME: str = "embeddings.npy"


@dataclass(frozen=True)
class ChunkRecord:
    """One searchable text chunk with provenance for citations."""

    chunk_id: str
    source_file: str
    title: str
    metadata: str
    text: str
    chunk_index: int


def parse_k_text_file(path: Path) -> tuple[str, str, str]:
    """
    Parse title, metadata, and body from a K-texts/*.txt file.

    Layout: title, blank, metadata, blank, body (see scrape_jk.format_k_text_file).

    Args:
        path: Path to a transcript .txt file.

    Returns:
        (title, metadata, body) strings.
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    if len(lines) < 5:
        return "", "", raw.strip()
    title = lines[0].strip()
    metadata = lines[2].strip() if len(lines) > 2 else ""
    body = "\n".join(lines[4:]).strip()
    return title, metadata, body


def chunk_body(
    body: str,
    target_tokens: int,
    overlap_tokens: int,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Split body text into overlapping token windows.

    Args:
        body: Transcript body only.
        target_tokens: Approximate max tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.
        encoding_name: tiktoken encoding name.

    Returns:
        List of decoded chunk strings (non-empty).
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(body or "")
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(1, target_tokens - overlap_tokens)
    i = 0
    while i < len(tokens):
        piece = tokens[i : i + target_tokens]
        decoded = enc.decode(piece).strip()
        if decoded:
            chunks.append(decoded)
        i += step
    return chunks


def build_chunk_records(k_texts_dir: Path) -> list[ChunkRecord]:
    """
    Load all .txt files under k_texts_dir and produce ChunkRecords.

    Args:
        k_texts_dir: Directory containing transcript files.

    Returns:
        Flat list of ChunkRecord across all files.
    """
    records: list[ChunkRecord] = []
    paths = sorted(k_texts_dir.glob("*.txt"))
    if not paths:
        logger.warning("No .txt files under %s", k_texts_dir)
    for path in paths:
        title, metadata, body = parse_k_text_file(path)
        parts = chunk_body(body, k_config.CHUNK_TARGET_TOKENS, k_config.CHUNK_OVERLAP_TOKENS)
        stem = path.stem
        if not parts:
            # Still index title/metadata for empty-body edge cases
            display = f"Title: {title}\n{metadata}\n\n{body}".strip()
            if display:
                parts = [display]
        for idx, text in enumerate(parts):
            enriched = (
                f"Title: {title}\nContext: {metadata}\n\n{text}"
                if title or metadata
                else text
            )
            cid = f"{stem}_{idx}"
            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    source_file=path.name,
                    title=title,
                    metadata=metadata,
                    text=enriched,
                    chunk_index=idx,
                )
            )
    return records


def _embed_batches(
    client: Any,
    model: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """
    Call the OpenAI-compatible embeddings API and return a float32 matrix.

    Args:
        client: OpenAI client.
        model: Embedding deployment name.
        texts: All chunk strings in order.
        batch_size: Max texts per API request.

    Returns:
        Array of shape (len(texts), dim) with L2-normalized rows for cosine sim.
    """
    all_rows: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # API returns data in index order
        data = sorted(resp.data, key=lambda d: d.index)
        for d in data:
            all_rows.append(list(d.embedding))
    mat = np.array(all_rows, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def build_and_save_index(k_texts_dir: Path, index_dir: Path) -> int:
    """
    Chunk K-texts, embed, and write chunks.jsonl + embeddings.npy.

    Args:
        k_texts_dir: Source directory of .txt transcripts.
        index_dir: Directory to create/write index artifacts.

    Returns:
        Number of chunks indexed.
    """
    records = build_chunk_records(k_texts_dir)
    if not records:
        raise RuntimeError(f"No chunks built from {k_texts_dir}. Run scrape_jk.py first.")
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [r.text for r in records]
    client = initialize_client()
    emb_model = get_embedding_deployment()
    logger.info("Embedding %s chunks with deployment %s", len(texts), emb_model)
    matrix = _embed_batches(client, emb_model, texts, k_config.EMBEDDING_BATCH_SIZE)
    chunks_path = index_dir / INDEX_CHUNKS_NAME
    emb_path = index_dir / INDEX_EMBEDDINGS_NAME
    with chunks_path.open("w", encoding="utf-8") as f:
        for r in records:
            row = {
                "chunk_id": r.chunk_id,
                "source_file": r.source_file,
                "title": r.title,
                "metadata": r.metadata,
                "text": r.text,
                "chunk_index": r.chunk_index,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.save(emb_path, matrix)
    logger.info("Wrote %s and %s", chunks_path, emb_path)
    return len(records)


def load_index(index_dir: Path) -> tuple[list[ChunkRecord], np.ndarray]:
    """
    Load chunk metadata and embedding matrix from disk.

    Args:
        index_dir: Directory containing chunks.jsonl and embeddings.npy.

    Returns:
        (records, embeddings_matrix) with normalized rows.
    """
    chunks_path = index_dir / INDEX_CHUNKS_NAME
    emb_path = index_dir / INDEX_EMBEDDINGS_NAME
    if not chunks_path.is_file() or not emb_path.is_file():
        raise FileNotFoundError(f"Missing index under {index_dir}; run build_and_save_index first.")
    records: list[ChunkRecord] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        records.append(
            ChunkRecord(
                chunk_id=o["chunk_id"],
                source_file=o["source_file"],
                title=o["title"],
                metadata=o["metadata"],
                text=o["text"],
                chunk_index=int(o["chunk_index"]),
            )
        )
    matrix = np.load(emb_path)
    return records, matrix.astype(np.float32)


def embed_query(client: Any, model: str, question: str) -> np.ndarray:
    """
    Embed a single user question as a normalized row vector.

    Args:
        client: OpenAI client.
        model: Embedding deployment name.
        question: User query string.

    Returns:
        Shape (1, dim) float32 normalized vector.
    """
    resp = client.embeddings.create(model=model, input=[question])
    vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return vec / n


def retrieve_top_k(
    question: str,
    records: list[ChunkRecord],
    matrix: np.ndarray,
    client: Any,
    k: int,
) -> list[tuple[ChunkRecord, float]]:
    """
    Return the top-k chunks by cosine similarity to the question.

    Args:
        question: User query.
        records: Chunk metadata aligned with matrix rows.
        matrix: Normalized embedding matrix (n, dim).
        client: OpenAI client for query embedding.
        k: Number of chunks to return.

    Returns:
        List of (ChunkRecord, score) sorted by descending score.
    """
    model = get_embedding_deployment()
    q = embed_query(client, model, question)
    scores = (matrix @ q.T).reshape(-1)
    top_idx = np.argsort(-scores)[:k]
    out: list[tuple[ChunkRecord, float]] = []
    for i in top_idx:
        out.append((records[int(i)], float(scores[int(i)])))
    return out


def build_rag_user_message(question: str, retrieved: list[tuple[ChunkRecord, float]]) -> str:
    """
    Build the user message containing numbered sources and the question.

    Args:
        question: User's natural language question.
        retrieved: Retrieved chunks with scores (scores may be omitted in text).

    Returns:
        A single user-role message string.
    """
    blocks: list[str] = []
    for n, (rec, _score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[{n}] Source file: {rec.source_file}\n"
            f"Title: {rec.title}\n"
            f"Details: {rec.metadata}\n\n{rec.text}"
        )
    sources = "\n\n---\n\n".join(blocks)
    return (
        "Use ONLY the following excerpts from J. Krishnamurti talks to ground your answer. "
        "Cite passage numbers like [1], [2] where relevant. "
        "Do not claim to be Krishnamurti; respond as a careful assistant helping inquiry.\n\n"
        f"{sources}\n\n---\n\nQuestion:\n{question}"
    )


RAG_SYSTEM_PROMPT: str = (
    "You help the user explore questions in light of J. Krishnamurti's teachings. "
    "Rely on the provided excerpts; if something is not in the excerpts, say so. "
    "Prefer clear, non-authoritarian language and inquiry over fixed prescriptions."
)


def answer_with_rag(
    question: str,
    index_dir: Path | None = None,
    top_k: int | None = None,
    preloaded: tuple[list[ChunkRecord], np.ndarray] | None = None,
) -> str:
    """
    Retrieve context from the local index and return the chat model reply.

    Args:
        question: User question.
        index_dir: Directory with chunks.jsonl and embeddings.npy (default: k_config.K_INDEX_DIR).
        top_k: Number of chunks to retrieve (default: k_config.RAG_TOP_K).
        preloaded: Optional (records, embedding_matrix) from load_index to avoid reloading disk.

    Returns:
        Assistant message content string.
    """
    idx_dir = index_dir or k_config.K_INDEX_DIR
    k = top_k if top_k is not None else k_config.RAG_TOP_K
    if preloaded is not None:
        records, matrix = preloaded
    else:
        records, matrix = load_index(idx_dir)
    client = initialize_client()
    retrieved = retrieve_top_k(question, records, matrix, client, k)
    user_msg = build_rag_user_message(question, retrieved)
    chat_model = get_deployment()
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=k_config.RAG_TEMPERATURE,
        max_tokens=min(get_max_tokens(), 4096),
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Empty completion from chat model")
    return content
