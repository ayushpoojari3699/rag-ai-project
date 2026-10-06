"""End-to-end tests for the DocuMind API with a fake LLM (no Ollama needed).

Run:  python -m pytest -q
The first run downloads the small BGE embedding model (~130 MB).
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of text per page (no extra libraries)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 11 Tf 40 760 Td ({safe}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_no = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_no} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


HANDBOOK = make_pdf([
    "Welcome to Acme. This handbook explains company policies for all employees.",
    "Leave policy: employees receive 24 days of paid annual leave each calendar year.",
    "Resignation notice period: employees must serve a notice period of 60 days.",
])
TRAVEL = make_pdf([
    "Travel policy: economy class flights are mandatory for trips under six hours.",
    "Hotel expenses are reimbursed up to 5000 rupees per night with original receipts.",
])
BLANK = make_pdf([""])


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return "Based on the documents [1]."


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMIND_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("DOCUMIND_INDEX_DIR", str(tmp_path / "vector_store"))
    import rag_core, app as app_module
    importlib.reload(rag_core)
    importlib.reload(app_module)
    app_module.engine._llm = FakeLLM()
    from fastapi.testclient import TestClient
    c = TestClient(app_module.app)
    c.engine = app_module.engine
    return c


def upload(client, name, data):
    return client.post("/upload", files={"file": (name, data, "application/pdf")})


def test_ask_before_upload_is_rejected(client):
    r = client.get("/ask", params={"q": "anything"})
    assert r.status_code == 400


def test_non_pdf_rejected(client):
    r = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_blank_pdf_rejected(client):
    assert upload(client, "blank.pdf", BLANK).status_code == 422


def test_multiple_documents_are_kept(client):
    assert upload(client, "handbook.pdf", HANDBOOK).json()["pages"] == 3
    assert upload(client, "travel.pdf", TRAVEL).status_code == 200
    files = {d["file"] for d in client.get("/documents").json()["documents"]}
    assert files == {"handbook.pdf", "travel.pdf"}


def test_retrieval_finds_the_right_page_and_cites_it(client):
    upload(client, "handbook.pdf", HANDBOOK)
    upload(client, "travel.pdf", TRAVEL)
    r = client.get("/ask", params={"q": "What is the resignation notice period?", "k": 2})
    assert r.status_code == 200
    data = r.json()
    top = data["sources"][0]
    assert (top["file"], top["page"]) == ("handbook.pdf", 3)
    assert "60 days" in top["snippet"]
    assert len(data["sources"]) == 2
    assert set(data["timings_ms"]) == {"retrieval", "generation", "total"}
    # the LLM only sees the retrieved passages, numbered for citation
    prompt = client.engine.llm.prompts[-1]
    assert "[1] (handbook.pdf, page 3)" in prompt
    assert "economy class" not in prompt or "[2]" in prompt

    r = client.get("/ask", params={"q": "hotel reimbursement per night receipts", "k": 1})
    assert (r.json()["sources"][0]["file"], r.json()["sources"][0]["page"]) == ("travel.pdf", 2)


def test_reupload_replaces_instead_of_duplicating(client):
    first = upload(client, "handbook.pdf", HANDBOOK).json()
    again = upload(client, "handbook.pdf", HANDBOOK).json()
    assert again["replaced_existing"] is True
    docs = client.get("/documents").json()["documents"]
    assert docs == [{"file": "handbook.pdf", "chunks": first["chunks"]}]


def test_remove_one_document(client):
    upload(client, "handbook.pdf", HANDBOOK)
    upload(client, "travel.pdf", TRAVEL)
    assert client.delete("/documents/travel.pdf").status_code == 200
    r = client.get("/ask", params={"q": "hotel reimbursement", "k": 3})
    assert {s["file"] for s in r.json()["sources"]} == {"handbook.pdf"}


def test_clear_all(client):
    upload(client, "handbook.pdf", HANDBOOK)
    assert client.delete("/documents").status_code == 200
    assert client.get("/documents").json()["documents"] == []
    assert client.get("/ask", params={"q": "leave"}).status_code == 400


def test_index_persists_across_restarts(client):
    upload(client, "handbook.pdf", HANDBOOK)
    import rag_core
    restarted = rag_core.RAGEngine(llm=FakeLLM())
    assert restarted.documents()[0]["file"] == "handbook.pdf"
    doc, _ = restarted.retrieve("annual leave days", k=1)[0]
    assert doc.metadata["page"] + 1 == 2
