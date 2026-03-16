DocuMind Pro – AI Document Q&A System

DocuMind Pro is an AI-powered document intelligence system that allows users to upload PDF documents and interact with them through a chat-based interface. The system uses Retrieval-Augmented Generation (RAG) with a large language model to generate answers based on document content.

Users can ask questions about uploaded documents and receive contextual responses in real time.


 Features:

Upload PDF documents

Ask questions about document content

Chat-based interface

AI-generated contextual answers

FastAPI backend for API handling

Streamlit interface for interactive UI

Local LLM inference using LLaMA3 via Ollama



 System Architecture:
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
 Tech Stack


Programming:

Python


Backend:

FastAPI


Frontend:

Streamlit


AI / LLM:

LangChain

LLaMA3

Ollama


Document Processing:

PDF parsing


Project Structure:
documind/
│
├── app.py              # FastAPI backend
├── streamlit_app.py    # Streamlit frontend
├── launcher.py         # Runs backend + frontend
└── README.md

Installation:

1. Clone repository
git clone https://github.com/yourusername/documind.git
cd documind
2. Install dependencies
pip install -r requirements.txt
3. Install Ollama


Download Ollama:
https://ollama.ai


Pull the model:
ollama pull llama3


Run the Application:

Start the backend:

uvicorn app:app --host 127.0.0.1 --port 8000

Run Streamlit:

streamlit run streamlit_app.py

Open in browser:

http://localhost:8501


How It Works:

User uploads a PDF document

Backend processes the document

User asks a question

Query is sent to the LLM through LangChain

LLaMA3 generates a contextual response

Answer is displayed in the chat interface



Future Improvements:

Vector database integration (FAISS / Chroma)

Document chunking and embeddings

Source citations

Multi-document support

Streaming responses



Use Cases:

Research document analysis

Knowledge base assistants

Study material interaction

Corporate document search