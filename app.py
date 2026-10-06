"""DocuMind FastAPI backend: upload PDFs, ask questions, get cited answers."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from rag_core import DEFAULT_TOP_K, DOCS_DIR, NoDocumentsError, RAGEngine

app = FastAPI(title="DocuMind – RAG Document Q&A")
engine = RAGEngine()


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(engine.documents())}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name  # strip any client-supplied folders
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    path = DOCS_DIR / name
    path.write_bytes(await file.read())
    try:
        return {"status": "indexed", **engine.add_pdf(path)}
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))


@app.get("/documents")
def documents():
    return {"documents": engine.documents()}


@app.delete("/documents")
def clear_documents():
    engine.reset()
    return {"status": "cleared"}


@app.delete("/documents/{filename}")
def remove_document(filename: str):
    removed = engine.remove_source(filename)
    if not removed:
        raise HTTPException(404, f"{filename} is not indexed.")
    (DOCS_DIR / Path(filename).name).unlink(missing_ok=True)
    return {"status": "removed", "file": filename, "chunks": removed}


@app.get("/ask")
def ask(q: str = Query(..., min_length=1), k: int = Query(DEFAULT_TOP_K, ge=1, le=10)):
    if not q.strip():
        raise HTTPException(400, "Empty question.")
    try:
        return engine.answer(q.strip(), k=k)
    except NoDocumentsError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # most often: Ollama is not running
        msg = str(e)
        if "Connection" in msg or "11434" in msg:
            raise HTTPException(503, "Can't reach Ollama. Start it and run `ollama pull llama3`.")
        raise HTTPException(500, msg)
