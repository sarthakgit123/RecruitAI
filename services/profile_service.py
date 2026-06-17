from google import genai
import os
from dotenv import load_dotenv
import pymupdf
import json
from datetime import datetime

from experience_utils import get_verified_experience_years

load_dotenv()


def extract_text(pdf):
    """
    Extracts text from ALL pages of the PDF (previous version only
    kept the last page's text due to overwriting `text` in the loop).
    """
    doc = pymupdf.open(pdf)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def profile_generate(pdf):
    resume_text = extract_text(pdf)

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Giving Gemini today's date lets it correctly resolve "Present"
    # when computing total_experience_years.
    today_str = datetime.today().strftime("%Y-%m-%d")

    prompt = f"""
        Return ONLY valid JSON.
        {{
        Schema:

        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "current_role": "",
        "total_experience_years": 0,
        "skills": [],
        "projects": [
            {{
            "title": "",
            "technologies": [],
            "description": []
            }}
        ],
        "education": [
            {{
            "institution": "",
            "degree": "",
            "score": "",
            "year": ""
            }}
        ],
        "experience": [
            {{
            "role": "",
            "company": "",
            "duration": "",
            "start_date": "",
            "end_date": "",
            "description": []
            }}
        ]
        }}

        Rules:
        - Always return arrays where specified.
        - Never return strings where arrays are expected.
        - If a section is missing, return [].
        - If a text field is unavailable, return "".
        - If a numeric field is unavailable, return 0.
        - "current_role" is the title from the most recent job in "experience".
          If there is no work experience, return "".
        - "total_experience_years" is the SUM of duration (in years, rounded
          to 1 decimal) across all entries in "experience". Today's date is
          {today_str}. Treat "Present"/"Current" as today's date when
          calculating duration. If there is no experience, return 0.
        - "start_date" and "end_date" must be formatted as "YYYY-MM" when the
          information is available (use "Present" as end_date if the role is
          ongoing). If unavailable, return "".
        - "location" is the candidate's city/region if mentioned anywhere in
          the resume (e.g. address, contact section). If not mentioned,
          return "".
        - Output valid JSON only. No markdown, no commentary, no code fences.

    Resume:
    {resume_text}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw = response.text.strip()
    # Defensive cleanup in case Gemini still wraps output in code fences
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini did not return valid JSON for {pdf}: {e}\nRaw output:\n{raw}"
        )

    # Don't trust Gemini's self-reported total_experience_years blindly.
    # Recompute it in plain Python from start_date/end_date wherever
    # possible, and only fall back to Gemini's number when dates are
    # missing/unparseable. This matters because the chatbot will filter
    # candidates directly on this number (e.g. "3+ years experience"),
    # so it needs to be as accurate as possible.
    verified_years, was_verified = get_verified_experience_years(profile)
    profile["total_experience_years"] = verified_years
    profile["experience_years_verified"] = was_verified

    return profile


if __name__ == "__main__":
    # Quick manual test
    import sys
    if len(sys.argv) > 1:
        result = profile_generate(sys.argv[1])
        print(json.dumps(result, indent=2))