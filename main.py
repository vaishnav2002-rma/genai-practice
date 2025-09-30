from google import genai 
import os 

os.environ["GEMINI_API_KEY"] = "AIzaSyCkIWVQD0G_NqWgRriwwGasGtDaGZEMW1M"

client = genai.Client()

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = "How to ride scooter" 
)

print(response.text)