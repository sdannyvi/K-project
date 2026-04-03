"""
Streamlit web UI for K-augmented RAG: chat grounded in K-texts transcripts.

Loads API keys from .env locally or from Streamlit secrets in the cloud (see README).
Caches the vector index for faster follow-up questions.

Update when UI flow or RAG wiring changes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

import k_config
from rag import INDEX_CHUNKS_NAME, answer_with_rag, build_and_save_index, load_index

logger = logging.getLogger(__name__)

# Keys mirrored in .env / Streamlit Cloud Secrets (same names).
_SECRET_KEYS: tuple[str, ...] = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "OPENAI_EMBEDDING_DEPLOYMENT",
    "OPENAI_TEMPERATURE",
    "OPENAI_MAX_TOKENS",
    "OPENAI_TIMEOUT_SECONDS",
)


def sync_streamlit_secrets_to_environ() -> None:
    """
    Copy Streamlit Cloud ``st.secrets`` into ``os.environ`` for llm_client.

    Local dev uses ``.env`` via dotenv in ``initialize_client``; cloud has no .env file.
    """
    try:
        for key in _SECRET_KEYS:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except (FileNotFoundError, TypeError, KeyError):
        pass


@st.cache_resource
def load_rag_index_cached() -> tuple[list, object]:
    """
    Load and cache chunks + embedding matrix from .k_index/.

    Returns:
        Same tuple as ``load_index`` (ChunkRecord list + numpy matrix).
    """
    return load_index(k_config.K_INDEX_DIR)


def main() -> None:
    """Run Streamlit chat: build index if missing, then RAG-backed replies."""
    st.set_page_config(page_title="Chat with K", page_icon="📖", layout="centered")
    sync_streamlit_secrets_to_environ()

    st.title("Chat with K")
    st.caption("Grounded in J. Krishnamurti transcripts (local RAG + your LLM deployment).")

    k_dir = k_config.K_TEXTS_DIR
    idx_dir = k_config.K_INDEX_DIR
    txts = [p for p in k_dir.glob("*.txt") if not p.name.startswith(".")]
    idx_path = idx_dir / INDEX_CHUNKS_NAME

    if not txts:
        st.error(
            f"No `.txt` transcripts in `{k_dir}`. "
            "Deploy the repo with `K-texts/` committed (see README), or add files locally."
        )
        st.stop()

    if not idx_path.is_file():
        st.warning("Embedding index not found. Build it once (uses your embedding API).")
        if st.button("Build index from K-texts"):
            with st.spinner("Embedding chunks (may take several minutes)…"):
                n = build_and_save_index(k_dir, idx_dir)
            st.success(f"Indexed **{n}** chunks.")
            load_rag_index_cached.clear()
            st.rerun()
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating…"):
                try:
                    preloaded = load_rag_index_cached()
                    answer = answer_with_rag(prompt, preloaded=preloaded)
                except Exception as exc:
                    logger.exception("RAG chat failed")
                    answer = f"**Error:** `{exc}`"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
