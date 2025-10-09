from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from dotenv import load_dotenv 
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

#GENERATE EMBEDDINGS

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_1 = embeddings.embed_query("Hello, world!")
print(len(vector_1))

vector_2 = embeddings.embed_documents(
    [
        "Today is Monday",
        "Today is Tuesday",
        "Today is April Fools day",
    ]
)

print(vector_2)

