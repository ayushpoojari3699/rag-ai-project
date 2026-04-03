#  DocuMind Pro – AI Document Q&A System

##  Overview

**DocuMind Pro** is an AI-powered document intelligence system that enables users to upload PDF documents and interact with them through a conversational interface.

It leverages **Retrieval-Augmented Generation (RAG)** with a local Large Language Model to provide accurate, context-aware answers based on document content.

---

##  Features

*  Upload PDF documents
*  Ask questions in a chat interface
*  AI-generated contextual responses
*  Real-time interaction
*  FastAPI backend for inference
*  Streamlit frontend for UI
*  Local LLM inference using **LLaMA3 via Ollama**

---

## System Architecture

```id="arch02"
User
 ↓
Streamlit Chat Interface
 ↓
FastAPI Backend
 ↓
Document Processing
 ↓
LangChain
 ↓
LLaMA3 (Ollama)
 ↓
Generated Response
```

---

##  Tech Stack

###  Programming

* Python

###  Backend

* FastAPI

###  Frontend

* Streamlit

###  AI / LLM

* LangChain
* LLaMA3
* Ollama

###  Document Processing

* PDF Parsing

---

##  Project Structure

```id="struct02"
documind/
│
├── app.py              # FastAPI backend
├── streamlit_app.py    # Streamlit frontend
├── launcher.py         # Run backend + frontend together
└── README.md
```

---

##  Installation

### 1️ Clone the Repository

```id="clone02"
git clone https://github.com/yourusername/documind.git
cd documind
```

---

### 2️ Install Dependencies

```id="install02"
pip install -r requirements.txt
```

---

### 3️ Install Ollama

Download from:
 https://ollama.ai

---

### 4️ Pull LLaMA3 Model

```id="ollama02"
ollama pull llama3
```

---

##  Run the Application

### Start FastAPI Backend

```id="runapi02"
uvicorn app:app --host 127.0.0.1 --port 8000
```

---

### Run Streamlit UI

```id="runui02"
streamlit run streamlit_app.py
```

---

### Open in Browser

```id="url02"
http://localhost:8501
```

---

##  How It Works

1. User uploads a PDF document
2. Backend processes and extracts text
3. User submits a query
4. Query is passed through **LangChain**
5. Relevant context is retrieved
6. **LLaMA3 (via Ollama)** generates an answer
7. Response is displayed in chat UI

---

##  Use Cases

*  Research document analysis
*  Knowledge base assistants
* Study material interaction
* Corporate document search

---

##  Future Improvements

*  Vector database integration (FAISS / Chroma)
*  Document chunking & embeddings
* Source citations in responses
*  Multi-document support
* Streaming responses

---

##  Notes

* Ensure Ollama is running before starting the app
* Model must be downloaded (`llama3`)
* Large documents may require optimization

---

##  Author

**Ayush Poojari**
GitHub: https://github.com/ayushpoojari3699


