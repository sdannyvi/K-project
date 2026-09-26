# K-Project / Intelligence of Non-Action

RAG over J. Krishnamurti transcripts: **CLI** (`main.py`) or **Streamlit** (`app.py`).

The repository now also contains the full research blueprint for the PNAS-scale
study **The Intelligence of Non-Action: Can a Language Model Cease Without a
Verbal Controller?** The manuscript is in [`paper/main.tex`](paper/main.tex), the
benchmark and pilots are under [`benchmark/`](benchmark/), the working open-weight
POC is under [`neural_poc/`](neural_poc/), and the SLURM handoff is
[`cluster/PNAS_CLUSTER_EXECUTION_PLAN.md`](cluster/PNAS_CLUSTER_EXECUTION_PLAN.md).

The PNAS program studies selective, recognition-coupled termination in model
generation dynamics. It does not claim that a language model is conscious or
awakened. Location-2/PNSE experience and Krishnamurti's distinction are preserved
as hypothesis-generating provenance, with independent validation.

The Streamlit app also includes an experimental, LLM-compatible adaptation of
the public [Finders assessment](https://app.thefinders.org/assessment). Human
self-report items can be marked not applicable; its output is a research
comparison, not a diagnosis or proof of machine consciousness.

See [`docs/finders_org_context.md`](docs/finders_org_context.md) for the
organization, research, evidence, privacy, and project-integration context.

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

1. **GitHub**: push this project (e.g. `sdannyvi/K-project`).
2. **Data**: `K-texts/` is **committed** so Streamlit has transcripts. `.k_index/` stays gitignored; on first cloud run use **“Build index”** in the app (slow; uses embedding API), or commit a prebuilt `.k_index/` if you remove that line from `.gitignore`.
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
| `k_eval.py` | Evaluation conditions and LLM Finders assessment |
| `benchmark/` | Construct development, negative results, prompts, and pilot records |
| `neural_poc/` | Open-weight activation, gate, CAA, and native LoRA pilot code |
| `paper/` | LaTeX manuscript, bibliography, and editorial review trail |
| `cluster/` | 4×A5000 SLURM execution plan, manifest, and restart-safe templates |
| `docs/` | Phenomenology, literature positioning, sources, and licensing notes |
