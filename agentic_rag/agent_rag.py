# ==========================
# IMPORTS
# ==========================
import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from IPython.display import Image, display

# LangChain & LangGraph
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import convert_to_messages

# Google Gemini SDK
from google import genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini 2.5 Flash model
model = genai.GenerativeModel("gemini-2.5-flash")

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

def gemini_invoke(prompt: str) -> str:
    """Call Gemini model and return text output."""
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else ""

def generate_query_or_respond(state: MessagesState):
    """Generate a query or a direct response."""
    messages = state["messages"]
    user_message = messages[0].content
    response = gemini_invoke(f"You are a helpful AI assistant. Respond to: {user_message}")
    return {"messages": [{"role": "assistant", "content": response}]}

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Document:\n{context}\n\n"
    "Question: {question}\n"
    "If the document contains keywords or semantic meaning related to the question, reply 'yes'. Otherwise, reply 'no'."
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Grade retrieved documents for relevance."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = gemini_invoke(prompt).lower()

    if "yes" in result:
        return "generate_answer"
    else:
        return "rewrite_question"

REWRITE_PROMPT = (
    "Look at the input and reason about the underlying semantic meaning.\n"
    "Here is the question:\n"
    "{question}\n\n"
    "Formulate an improved, clearer version of the question."
)

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = gemini_invoke(prompt)
    return {"messages": [{"role": "user", "content": rewritten}]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know, say 'I don't know'. "
    "Keep it concise within three sentences.\n\n"
    "Question: {question}\n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """Generate an answer based on retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    answer = gemini_invoke(prompt)
    return {"messages": [{"role": "assistant", "content": answer}]}

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
