"""DocuMind RAG core: index PDFs into FAISS and answer questions with page citations.

Pipeline
    PDF -> pages (PyPDFLoader) -> 800-char chunks, 150 overlap -> BGE embeddings
        -> FAISS index (saved to disk)
    question -> embed -> top-k similar chunks -> grounded prompt -> Llama 3 (Ollama)
        -> answer + cited sources + timings
"""

from __future__ import annotations

import os
import shutil
import time
import warnings
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The community Ollama / HuggingFace wrappers still work but emit deprecation notices.
warnings.filterwarnings("ignore", message=".*was deprecated.*")

DOCS_DIR = Path(os.getenv("DOCUMIND_DOCS_DIR", "docs"))
INDEX_DIR = Path(os.getenv("DOCUMIND_INDEX_DIR", "vector_store"))
EMBED_MODEL = os.getenv("DOCUMIND_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("DOCUMIND_LLM_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
SNIPPET_CHARS = 300

NOT_FOUND = "I couldn't find this in the uploaded documents."

PROMPT_TEMPLATE = """You are DocuMind, a document question-answering assistant.
Answer the question using ONLY the numbered context passages below.
Cite the passages you used with their numbers in square brackets, e.g. [1] or [2][3].
If the answer is not in the passages, reply exactly: "{not_found}"

Context:
{context}

Question: {question}

Answer:"""


class NoDocumentsError(RuntimeError):
    """Raised when a question is asked before any document is indexed."""


class RAGEngine:
    """Owns the embeddings, the FAISS index and the LLM."""

    def __init__(self, llm=None, embeddings=None):
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self._llm = llm
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self.db = self._load_index()

    # ------------------------------------------------------------------ LLM
    @property
    def llm(self):
        if self._llm is None:
            from langchain_community.llms import Ollama

            self._llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
        return self._llm

    # ---------------------------------------------------------------- index
    def _load_index(self):
        if (INDEX_DIR / "index.faiss").exists():
            # The index is written by this app only, so loading its pickle is safe.
            return FAISS.load_local(
                str(INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True
            )
        return None

    def _save(self):
        if self.db is not None:
            self.db.save_local(str(INDEX_DIR))

    def _ids_for_source(self, filename: str) -> list[str]:
        if self.db is None:
            return []
        return [
            doc_id
            for doc_id, doc in self.db.docstore._dict.items()
            if doc.metadata.get("source") == filename
        ]

    def add_pdf(self, path: str | Path) -> dict:
        """Index one PDF. Re-adding a file with the same name replaces its chunks."""
        path = Path(path)
        pages = PyPDFLoader(str(path)).load()
        for p in pages:
            p.metadata["source"] = path.name  # store the file name, not the full path
        chunks = [c for c in self.splitter.split_documents(pages) if c.page_content.strip()]
        if not chunks:
            raise ValueError(
                f"No extractable text in {path.name} (scanned PDFs need OCR first)."
            )

        replaced = self.remove_source(path.name, save=False)
        if self.db is None:
            self.db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.db.add_documents(chunks)
        self._save()
        return {
            "file": path.name,
            "pages": len(pages),
            "chunks": len(chunks),
            "replaced_existing": replaced > 0,
        }

    def remove_source(self, filename: str, save: bool = True) -> int:
        ids = self._ids_for_source(filename)
        if not ids:
            return 0
        if len(ids) == len(self.db.docstore._dict):
            self.db = None  # FAISS cannot hold an empty index; drop it entirely
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
        else:
            self.db.delete(ids)
            if save:
                self._save()
        return len(ids)

    def documents(self) -> list[dict]:
        if self.db is None:
            return []
        counts: dict[str, int] = {}
        for doc in self.db.docstore._dict.values():
            src = doc.metadata.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"file": f, "chunks": n} for f, n in sorted(counts.items())]

    def reset(self):
        """Remove every indexed document and the uploaded PDFs."""
        self.db = None
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        for f in DOCS_DIR.glob("*.pdf"):
            f.unlink()

    # ------------------------------------------------------------------ ask
    def retrieve(self, question: str, k: int = DEFAULT_TOP_K):
        if self.db is None:
            raise NoDocumentsError("No documents indexed yet. Upload a PDF first.")
        hits = self.db.similarity_search_with_score(question, k=k)
        # Vectors are normalised, so squared L2 distance d maps to cosine = 1 - d/2.
        return [(doc, round(1 - float(dist) / 2, 3)) for doc, dist in hits]

    def answer(self, question: str, k: int = DEFAULT_TOP_K) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        t1 = time.perf_counter()

        context = "\n\n".join(
            f"[{i}] ({doc.metadata.get('source')}, page {doc.metadata.get('page', 0) + 1})\n"
            f"{doc.page_content}"
            for i, (doc, _) in enumerate(hits, start=1)
        )
        prompt = PROMPT_TEMPLATE.format(
            context=context, question=question, not_found=NOT_FOUND
        )
        answer = str(self.llm.invoke(prompt)).strip()
        t2 = time.perf_counter()

        sources = [
            {
                "id": i,
                "file": doc.metadata.get("source"),
                "page": doc.metadata.get("page", 0) + 1,
                "score": score,
                "snippet": doc.page_content[:SNIPPET_CHARS],
            }
            for i, (doc, score) in enumerate(hits, start=1)
        ]
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "timings_ms": {
                "retrieval": round((t1 - t0) * 1000),
                "generation": round((t2 - t1) * 1000),
                "total": round((t2 - t0) * 1000),
            },
        }
