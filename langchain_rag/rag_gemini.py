import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from google import genai 
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

client = genai.Client(api_key=api_key)

PDF_PATH = "C:\\Users\\Dell\\sample_pdf\\A Detailed Guide to Mastering Time Management.pdf"
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
    import numpy as np
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_community.vectorstores.faiss import dependable_faiss_import

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
    import numpy as np

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

    print("\n🔍 Querying Gemini model...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(parts=[types.Part(text=prompt)])]  # ✅ Fixed
    )

    # Extract text safely
    answer = getattr(response, "text", str(response))
    print("\n🧩 Answer:\n", answer.strip())
    print("\n" + "-" * 80 + "\n")


def main():
    if not Path(PDF_PATH).exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    index, docs = build_or_load_vectorstore(PDF_PATH)

    print("\nReady! Ask questions about your PDF (type 'exit' to quit)\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if question:
            answer_question(question, index, docs)


if __name__ == "__main__":
    main()
