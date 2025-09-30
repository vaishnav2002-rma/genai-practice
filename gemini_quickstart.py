from google import genai 
import os 
from dotenv import load_dotenv 

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") 

os.environ["GEMINI_API_KEY"] = api_key

client = genai.Client()

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = "Write a code that print all the multiples of 5" 
)

print(response.python)