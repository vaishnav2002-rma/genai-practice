from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   
from langchain_ollama import OllamaEmbeddings, OllamaLLM                
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load and split the document
loader = PyPDFLoader("C:\\Users\\Dell\\sample_pdf\\A Detailed Guide to Mastering Time Management.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Create embeddings and vectorstore
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Set up retriever and LLM
retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="llama3.1")

# Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Run query
query = "Summarize the key points from the document."
answer = qa.run(query)
print(answer)
