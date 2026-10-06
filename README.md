# DocuMind – RAG Document Q&A

Ask questions about your PDFs and get answers **grounded in the documents, with page-level citations**.
Runs fully locally: FAISS for retrieval, a BGE embedding model, and Llama 3 through Ollama.

## How it works

```
Upload PDF ──► split into pages ──► 800-char chunks (150 overlap) ──► BGE-small embeddings ──► FAISS index (saved to disk)

Question ──► embed ──► top-k most similar chunks ──► numbered context + question ──► Llama 3 (Ollama)
                                                                                        │
                                       answer with [1][2] citations + source file/page + timings
```

- **Retrieval-augmented generation:** only the retrieved passages go to the LLM, so long documents work and answers stay grounded.
- **Citations:** each answer lists its source passages (file, page, similarity score, snippet).
- **"I don't know" behaviour:** the prompt tells the model to say so when the answer isn't in the passages.
- **Multiple documents:** upload several PDFs, remove one, or clear all. Re-uploading a file replaces its old chunks.
- **Persistent index:** the FAISS index is saved to `vector_store/` and reloaded on restart.
- **Timing:** every answer reports retrieval, generation and total time.

## Tech stack

FastAPI · Streamlit · LangChain · FAISS · BAAI/bge-small-en-v1.5 (sentence-transformers) · Llama 3 via Ollama · PyPDF

## Project structure

```
rag_core.py           RAG engine: indexing, retrieval, prompting, citations, timings
app.py                FastAPI backend (upload / documents / ask)
streamlit_app.py      Chat UI with sources and timings
launcher.py           Starts backend + UI and opens the browser
ingest.py             Index every PDF in docs/ from the command line
benchmark.py          Measure retrieval accuracy (hit@k) and latency on your own questions
tests/test_rag.py     End-to-end API tests (fake LLM, no Ollama needed)
```

## Setup

```bash
python -m venv rag-env
rag-env\Scripts\activate          # Windows  (macOS/Linux: source rag-env/bin/activate)
pip install -r requirements.txt

# install Ollama from https://ollama.com, then:
ollama pull llama3
```

## Run

```bash
python launcher.py
```

Or start the two parts yourself:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
streamlit run streamlit_app.py
```

Open http://localhost:8501, upload PDFs in the sidebar, and ask questions.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend status and number of indexed documents |
| POST | `/upload` | Upload and index one PDF (form field `file`) |
| GET | `/documents` | List indexed documents and chunk counts |
| DELETE | `/documents/{filename}` | Remove one document |
| DELETE | `/documents` | Remove everything |
| GET | `/ask?q=...&k=4` | Answer with sources and timings |

## Evaluate it

1. Index your PDFs, copy `eval_questions.example.json` to `eval_questions.json`, and write 15–20 questions with the file and page where each answer is.
2. Run `python benchmark.py` (or `python benchmark.py --retrieval-only` to skip the LLM).

It prints **hit@k** (how often the correct page is retrieved) and median / p90 response times.

## Tests

```bash
python -m pytest -q
```

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `DOCUMIND_LLM_MODEL` | `llama3` |
| `DOCUMIND_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `DOCUMIND_DOCS_DIR` / `DOCUMIND_INDEX_DIR` | `docs` / `vector_store` |

## Limitations

- Scanned (image-only) PDFs have no extractable text and are rejected; they need OCR first.
- Whole-document summaries only see the top-k passages; raise k for broader questions.
- Answers are generated locally, so speed depends on your CPU/GPU.
