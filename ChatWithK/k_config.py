"""
Configuration for the Krishnamurti corpus scraper and RAG pipeline.

Belongs to the K-augmented LLM experiment: paths, JK site URLs, chunking and
retrieval defaults, and embedding deployment name. Update when site URL patterns,
batch sizes, or index location change.
"""

from pathlib import Path

# Project root (directory containing this file).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Directory for one plain-text file per downloaded transcript.
K_TEXTS_DIR: Path = PROJECT_ROOT / "K-texts"

# All canonical transcript URLs from the last CDX discovery (one per line).
DISCOVERED_URLS_FILE: Path = PROJECT_ROOT / "discovered_urls.txt"

# URLs successfully written under K-texts/ (append-only; used for --resume).
SCRAPE_COMPLETED_URLS_FILE: Path = K_TEXTS_DIR / ".completed_urls.txt"

# Drop K-texts/*.txt with fewer than this many whitespace-separated words (clean step).
MIN_K_TEXT_WORDS: int = 30

# Persisted vector index (embeddings + chunk metadata) for RAG.
K_INDEX_DIR: Path = PROJECT_ROOT / ".k_index"

JK_SITE_BASE: str = "https://jkrishnamurti.org"

# Search listing: transcripts (type/media_type from JK search UI).
JK_SEARCH_PATH: str = "/jksearch"
JK_SEARCH_QUERY: str = "keyword=&type=16616&media_type=16616"

# Wayback Machine: snapshot for fetching archived HTML (transcript pages).
WAYBACK_SNAPSHOT: str = "20190819104847"

# CDX API prefix to list unique /content/... URLs (collapse duplicates).
WAYBACK_CDX_CONTENT_PREFIX: str = "http://www.jkrishnamurti.org/content/"

# Request pacing (seconds between transcript fetches).
SCRAPE_DELAY_SECONDS: float = 1.5

# Upper bound on search result pages to crawl (10 items per page on site).
SCRAPE_MAX_PAGES: int = 500

# Chunking for embeddings (approximate tokens via tiktoken).
CHUNK_TARGET_TOKENS: int = 600
CHUNK_OVERLAP_TOKENS: int = 80

# Embedding batch size for OpenAI-compatible API.
EMBEDDING_BATCH_SIZE: int = 64

# Retrieved chunks per query.
RAG_TOP_K: int = 8

# RAG generation temperature (override OPENAI_TEMPERATURE for main demo).
RAG_TEMPERATURE: float = 0.35

# Env vars tried in order for embedding deployment name (RAG). Set one in .env.
EMBEDDING_DEPLOYMENT_ENV_KEYS: tuple[str, ...] = (
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "OPENAI_EMBEDDING_DEPLOYMENT",
)
