from google import genai 
from google.genai import types
import os 
from dotenv import load_dotenv 

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key 

essay_prompt = """
         You are a school student, write an essay with a few mistakes, like spellsing mistakes, punctuation errors and more. Ensure your essay includes a clear thesis statement. You should write only an essay, so do not include any notes.
         """

client = genai.Client()

student_reponse = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = "Write an essay on the benefits of regular exercises",
    config = types.GenerateContentConfig(
        system_instruction = essay_prompt
    )
)

essay = student_reponse.text 
print(essay)

teacher_prompt = """
                    As a teacher you are responsible for grading the students essays based on the following criterias

                    1. Evaluate the essays on the following criterias.
                     - Thesis statement,
                     - Clarity and precision of language,
                     - Grammer and punctuation
                     - Argumentation

                    2. Write a corrected version of the essay addressing any identified issues
                       in the original submission. Point out what changes were made.

                    3. Grade the essay written by the student from 1-5.
                 """

teacher_response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = essay,
    config = types.GenerateContentConfig(
        system_instruction = teacher_prompt   
    )
)

print(teacher_response.text)