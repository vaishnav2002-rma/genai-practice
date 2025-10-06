import os
import json
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Gemini Text Processing API")

class TextInput(BaseModel):
    text: str

@app.post("/v1/summarize")
def generate_summary(data: TextInput):
    try:
        prompt = f"Generate a summary of the following text in 2-3 lines:\n\n{data.text}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        summary = response.text.strip()
        return {"Summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/entities")
def generate_entities(data: TextInput):
    try:
        prompt = f"""
        Extract and generate entities from the text below.
        Generate the entities in JSON format and include the fields: entity, type, and context.
        DO NOT include any markdown formatting (like ```json) or any surrounding text — only the raw JSON array.

        Text:
        {data.text}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        json_string = response.text.strip()

        entities_data = json.loads(json_string)

        return {"Entities": entities_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


