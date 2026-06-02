from google import genai
import os 
from dotenv import load_dotenv
import pymupdf
import json

load_dotenv()

def extract_text(pdf):
    doc = pymupdf.open(pdf) 
    out = open("output.txt", "wb") 
    for page in doc: 
        text = page.get_text().encode("utf8")    
    out.close()

    return text

def profile_genrate(pdf):
    resume_text = extract_text(pdf)

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt= f"""
  Extract the candidate information and return ONLY valid JSON.

    {{
        "name":"",
        "email":"",
        "skills":[],
        "projects":[],
        "education":[],
        "experience":[]
    }}

    Resume:
    {resume_text}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    profile = json.loads(
        response.text.replace("```json", "").replace("```", "")
    )
    

    return profile
