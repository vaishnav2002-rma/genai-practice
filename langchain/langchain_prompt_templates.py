import os 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt_template.invoke({"topic": "criket players"})