from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

import os
import shutil
import traceback

app = FastAPI(title="DocuMind Pro – No Indexing Version")

# ===============================
# CONFIG
# ===============================

DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

llm = Ollama(model="llama3")

# ===============================
# HEALTH CHECK (IMPORTANT)
# ===============================

@app.get("/health")
def health():
    return {"status": "ok"}

# ===============================
# UPLOAD PDF
# ===============================

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:

        # remove old PDFs
        for f in os.listdir(DOCS_DIR):
            os.remove(os.path.join(DOCS_DIR, f))

        path = os.path.join(DOCS_DIR, file.filename)

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "uploaded",
            "file": file.filename
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# ===============================
# LOAD DOCUMENT TEXT
# ===============================

def load_document():
    try:

        for f in os.listdir(DOCS_DIR):

            if f.lower().endswith(".pdf"):

                loader = PyPDFLoader(os.path.join(DOCS_DIR, f))
                docs = loader.load()

                text = "\n\n".join([d.page_content for d in docs])

                return text[:8000]  # limit for speed

        return None

    except Exception as e:
        traceback.print_exc()
        return None

# ===============================
# ASK QUESTION
# ===============================

@app.get("/ask")
def ask(q: str):

    try:

        if not q.strip():
            return {"error": "Empty question"}

        text = load_document()

        if not text:
            return {"error": "No document uploaded"}

        prompt = f"""
You are an intelligent document assistant.

The user uploaded a document.

User request:
{q}

Document content:
{text}

Instructions:
- If user asks for summary → summarize the document
- If user asks question → answer using the document
- If user asks improvement or suggestions → provide clear bullet points
- Keep answers clear and structured
"""

        answer = llm.invoke(prompt)

        return {
            "question": q,
            "answer": answer
        }

    except Exception as e:

        traceback.print_exc()
        return {"error": str(e)}