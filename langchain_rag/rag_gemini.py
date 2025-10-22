import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain.memory import PostgresChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage

from google import genai 
from google.genai import types
import numpy as np
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

client = genai.Client(api_key=api_key)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database models
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, nullable=True)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    message_type = Column(String, nullable=False)  # 'human' or 'ai'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, nullable=True)  # JSON string for additional data

# Create tables
Base.metadata.create_all(bind=engine)

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
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0] if i < len(docs)]


def get_or_create_session(session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """Get existing session or create a new one."""
    db = SessionLocal()
    try:
        if session_id:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if session:
                # Update last activity
                session.last_activity = datetime.utcnow()
                db.commit()
                return session_id
        
        # Create new session
        new_session_id = str(uuid.uuid4())
        new_session = ChatSession(
            session_id=new_session_id,
            user_id=user_id
        )
        db.add(new_session)
        db.commit()
        return new_session_id
    finally:
        db.close()

def save_chat_message(session_id: str, message_type: str, content: str, metadata: Optional[str] = None):
    """Save a chat message to the database."""
    db = SessionLocal()
    try:
        message = ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            metadata=metadata
        )
        db.add(message)
        db.commit()
    finally:
        db.close()

def get_chat_history(session_id: str, limit: int = 50) -> List[dict]:
    """Retrieve chat history for a session."""
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()
        
        return [
            {
                "id": msg.id,
                "type": msg.message_type,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in reversed(messages)  # Reverse to get chronological order
        ]
    finally:
        db.close()

def get_conversation_context(session_id: str, max_messages: int = 10) -> str:
    """Get recent conversation context for better responses."""
    history = get_chat_history(session_id, max_messages)
    context_parts = []
    
    for msg in history[-max_messages:]:  # Get last N messages
        if msg["type"] == "human":
            context_parts.append(f"Human: {msg['content']}")
        elif msg["type"] == "ai":
            context_parts.append(f"Assistant: {msg['content']}")
    
    return "\n".join(context_parts) if context_parts else ""

def answer_question(question: str, index, docs, session_id: str, top_k: int = 4):
    """Retrieve relevant context and query Gemini with conversation memory."""
    results = search_similar(question, index, docs, k=top_k)
    context = "\n\n".join(results)
    
    # Get conversation history for context
    conversation_context = get_conversation_context(session_id)
    
    # Build enhanced prompt with conversation context
    prompt_parts = [
        "You are an expert assistant with access to conversation history. Answer the question using both the provided context and conversation history.",
        "If the context does not contain the answer, say 'I don't know.'",
        "Consider the conversation flow and provide relevant, contextual responses.",
        "",
        "Context from documents:",
        context,
        ""
    ]
    
    if conversation_context:
        prompt_parts.extend([
            "Recent conversation history:",
            conversation_context,
            ""
        ])
    
    prompt_parts.extend([
        f"Current question: {question}",
        "",
        "Answer:"
    ])
    
    prompt = "\n".join(prompt_parts)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(parts=[types.Part(text=prompt)])]
    )

    return getattr(response, "text", str(response)).strip()

app = FastAPI(title="Gemini PDF RAG API", version="1.0")

index, docs = None, None 

RATE_LIMIT = 500
TIME_WINDOOW = 60
request_timestamps = []

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    global request_timestamps
    now = time.time()

    request_timestamps = [t for t in request_timestamps if now - t < TIME_WINDOOW]

    if len(request_timestamps) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429, 
            content = {"detail": "Rate limit exceeded. Please try again later."},
        )
    
    request_timestamps.append(now)
    response = await call_next(request)
    return response 

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    timestamp: datetime

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]
    created_at: datetime
    last_activity: datetime

class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime 

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

@app.post("/ask", response_model=ChatResponse)
def ask_question(payload: QuestionRequest):
    global index, docs 
    if index is None or docs is None:
        raise HTTPException(status_code=503, detail="FAISS index not ready yet.")
    
    try:
        # Get or create session
        session_id = get_or_create_session(payload.session_id, payload.user_id)
        
        # Save user question
        save_chat_message(session_id, "human", payload.question)
        
        # Get answer with memory context
        answer = answer_question(payload.question, index, docs, session_id)
        
        # Save AI response
        save_chat_message(session_id, "ai", answer)
        
        return ChatResponse(
            question=payload.question,
            answer=answer,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest):
    """Create a new chat session."""
    try:
        session_id = get_or_create_session(user_id=payload.user_id)
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            return SessionResponse(
                session_id=session_id,
                created_at=session.created_at
            )
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
def get_session_history(session_id: str, limit: int = 50):
    """Get chat history for a specific session."""
    try:
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            messages = get_chat_history(session_id, limit)
            
            return ChatHistoryResponse(
                session_id=session_id,
                messages=messages,
                created_at=session.created_at,
                last_activity=session.last_activity
            )
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
def list_sessions(user_id: Optional[str] = None, limit: int = 20):
    """List chat sessions, optionally filtered by user_id."""
    try:
        db = SessionLocal()
        try:
            query = db.query(ChatSession)
            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            
            sessions = query.order_by(ChatSession.last_activity.desc()).limit(limit).all()
            
            return [
                {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "created_at": session.created_at,
                    "last_activity": session.last_activity
                }
                for session in sessions
            ]
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a chat session and all its messages."""
    try:
        db = SessionLocal()
        try:
            # Delete messages first (foreign key constraint)
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            
            # Delete session
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            db.delete(session)
            db.commit()
            
            return {"message": "Session deleted successfully"}
        finally:
            db.close()
    except HTTPException:
        raise
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