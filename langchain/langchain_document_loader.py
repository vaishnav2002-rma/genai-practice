import os 
from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

file_path = ("C:\\Users\\Dell\\Documents\\Importance_of_Reading_Books.pdf")

loader = PyPDFLoader(file_path)

pages = []

for page in loader.lazy_load():
    pages.append(page)

print(pages[0].page_content)