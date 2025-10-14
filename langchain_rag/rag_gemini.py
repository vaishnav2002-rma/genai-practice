import os
import math 
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import

from google import genai 
from google.genai import types
import numpy as np

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

client = genai.Client(api_key=api_key)

PDF_PATH = Path("C:\\Users\\Dell\\sample_pdf\\A Detailed Guide to Mastering Time Management.pdf")
INDEX_DIR = Path("faiss_index")
GEMINI_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"  

def get_gemini_embedding(text: str):
    """Return embedding vector from Gemini."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
    )
    return response.embeddings[0].values

def build_or_load_vectorstore(pdf_path: str) -> FAISS:
    """Build FAISS index from PDF or load existing one."""
    faiss = dependable_faiss_import()
    index_file = INDEX_DIR / "index.faiss"
    docs_file = INDEX_DIR / "docs.txt"

    if INDEX_DIR.exists() and index_file.exists() and docs_file.exists():
        print("🔹 Loading existing FAISS index...")
        index = faiss.read_index(str(index_file))
        with open(docs_file, "r", encoding="utf-8") as f:
            doc_texts = f.read().split("\n\n---\n\n")
        return index, doc_texts

    print("📘 Loading and indexing PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("🔹 Generating Gemini embeddings (this may take a minute)...")
    embeddings = [get_gemini_embedding(doc.page_content) for doc in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_file))
    with open(docs_file, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join([d.page_content for d in chunks]))

    print(f"✅ Created FAISS index with {len(chunks)} chunks.")
    return index, [d.page_content for d in chunks]


def search_similar(question: str, index, docs, k: int = 4):
    """Find top-k similar document chunks."""
    q_emb = np.array(get_gemini_embedding(question), dtype="float32").reshape(1, -1)
    distances, indices = index.search(q_emb, k)
    return [docs[i] for i in indices[0]]


def answer_question(question: str, index, docs, top_k: int = 4):
    """Retrieve relevant context and query Gemini."""
    results = search_similar(question, index, docs, k=top_k)
    context = "\n\n".join(results)

    prompt = (
        "You are an expert assistant. Answer the question only using the provided context.\n"
        "If the context does not contain the answer, say 'I don’t know.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(parts=[types.Part(text=prompt)])]  # ✅ Fixed
    )

    return getattr(response, "text", str(response)).strip()

app = FastAPI(title="Gemini PDF RAG API", version="1.0")

index, docs = None, None 

class QuestionRequest(BaseModel):
    question: str 

@app.on_event("startup")
def load_index_on_startup():
    global index, docs 
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    index, docs = build_or_load_vectorstore(PDF_PATH)
    print("FAISS index ready for use.")

@app.get("/")
def root():
    return {"message": "Welcome to the Gemini PDF RAG API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    global index, docs 
    if index is None or docs is None:
        raise HTTPException(status_code=503, detail="FAISS index not ready yet.")
    try:
        answer = answer_question(payload.question, index, docs)
        return {"question": payload.question, "answer":answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/rebuild-index")
def rebuild_index():
    global index, docs 
    try:
        index, docs = build_or_load_vectorstore(PDF_PATH)
        return {"message": "INDEX rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))