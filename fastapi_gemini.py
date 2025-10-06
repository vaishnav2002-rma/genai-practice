import os 
import json 
import google.generativeai as genai 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
from dotenv import load_dotenv 

load_dotenv()

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

app = FastAPI(text = "Gemini text processing api")

class TextInput(BaseModel):
    text: str 

@app.post("/v1/summarize")
def generate_summary(data: TextInput):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"Generate summary of the following text in 2-3 line:\n\n{data.text}"
        response = model.generate_content(prompt)
        summary = response.text.strip()
        return {"Summary":summary}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.post("/v1/entities")
def generate_entities(data: TextInput):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Extract and generate entities from the text below.
        Generate the entities in json format and include the fields entity, type, context.

        Text:
        {data.text}
        """
        response = model.generate_content(prompt)
        json_string = response.text.strip()
        entities_data = json.loads(json_string)
        return {"Entities": entities_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
