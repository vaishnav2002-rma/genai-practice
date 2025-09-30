from google import genai 
import os 
from dotenv import load_dotenv 

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key 

prompt = """
QUERY: list all movies released after 2015 that won more than 3 Academy Awards.  
CONTEXT:

Table title: Movies and Academy Awards Won  
The Lord of the Rings: The Return of the King, 11, 2003  
Slumdog Millionaire, 8, 2008  
The Shape of Water, 4, 2017  
Parasite, 4, 2019  
Mad Max: Fury Road, 6, 2015  
La La Land, 6, 2016  
Everything Everywhere All At Once, 7, 2022  
Oppenheimer, 7, 2023 
"""

client = genai.Client()
result = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = prompt     
)

print(result.text)