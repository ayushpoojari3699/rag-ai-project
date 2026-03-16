from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

docs = []
for file in os.listdir("docs"):
    path = os.path.join("docs", file)
    if file.endswith(".pdf"):
        docs.extend(PyPDFLoader(path).load())
    elif file.endswith(".txt"):
        docs.extend(TextLoader(path).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vector_store")

print("✅ Documents indexed into vector_store")