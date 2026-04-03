# K-Project

RAG over J. Krishnamurti transcripts: **CLI** (`main.py`) or **Streamlit** (`app.py`).

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Uses `.env` in the project root (same variables as `llm_client.py`). Or:

```bash
python main.py
```

## Streamlit Cloud (deploy)

1. **GitHub**: create the repo (e.g. `sadnnyi/K-project`), push this project.
2. **Data**: `.gitignore` excludes `K-texts/` and `.k_index/`. For a working cloud app you must either:
   - **Commit** `K-texts/` and `.k_index/` (temporarily remove those two lines from `.gitignore`, then `git add -f K-texts .k_index` or add normally), or  
   - Ship **only** `K-texts/` and use **“Build index”** in the app (first load is slow; needs embedding API quota).
3. **Streamlit Community Cloud**: New app → pick repo → **Main file**: `app.py` → Deploy.
4. **Secrets** (dashboard → App settings → Secrets): add the same keys as `.env.example`, in **TOML** form:

```toml
AZURE_OPENAI_ENDPOINT = "https://api.openai.com/v1"
AZURE_OPENAI_API_KEY = "your-key"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
```

Optional: `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`, `OPENAI_TIMEOUT_SECONDS`.

Streamlit injects these into `st.secrets`; `app.py` copies them into `os.environ` so `llm_client` works without a `.env` file on the server.

**Yes — set keys in Streamlit app secrets**, not only locally. Do not commit real keys to git.

## Layout

| Path | Role |
|------|------|
| `K-texts/*.txt` | Transcript corpus |
| `.k_index/` | Built embeddings (optional to commit) |
| `app.py` | Streamlit chat UI |
| `main.py` | One-shot CLI question |
| `rag.py` | Chunking, index, retrieval |
