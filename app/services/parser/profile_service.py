"""
profile_service.py

Generates structured candidate profile JSON from PDF resume using OpenRouter LLM.
Features robust multi-model fallback, JSON auto-repair for truncated output,
and graceful heuristic fallbacks.
"""

import os
import time
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional
import pymupdf
from openai import OpenAI

from app.core.config import settings
from app.services.parser.experience_utils import get_verified_experience_years
from app.prompts import load_prompt


def extract_text(pdf_path: str) -> str:
    """
    Extracts text from PDF, cleans whitespace, and caps length.
    """
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    cleaned = re.sub(r'\n\s*\n', '\n', full_text).strip()

    if len(cleaned) > 3500:
        cleaned = cleaned[:3500]
    return cleaned


def _clean_and_repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Cleans markdown formatting and attempts to repair truncated JSON if needed.
    """
    if not raw:
        return None

    cleaned = raw.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

    # Direct JSON parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract JSON between first '{' and last '}'
    start_idx = cleaned.find("{")
    if start_idx == -1:
        return None

    json_candidate = cleaned[start_idx:]

    # Attempt direct parse from start_idx
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        pass

    # Auto-repair truncated JSON (e.g. cut off inside string/array/object)
    repaired = json_candidate

    # If inside an unterminated string, close it
    quote_count = repaired.count('"') - repaired.count(r'\"')
    if quote_count % 2 != 0:
        repaired += '"'

    # Close open brackets and braces
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets

    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _fallback_profile_extract(pdf_path: str, resume_text: str) -> Dict[str, Any]:
    """
    Heuristic fallback profile extractor when all LLMs fail.
    Extracts candidate name from filename, detects common skills via regex.
    """
    filename = os.path.basename(pdf_path)
    candidate_id = os.path.splitext(filename)[0]
    candidate_name = re.sub(r'[\d_.-]+', ' ', candidate_id).strip().title()
    if not candidate_name:
        candidate_name = candidate_id

    # Detect email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
    email = email_match.group(0) if email_match else ""

    # Detect phone
    phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
    phone = phone_match.group(0) if phone_match else ""

    # Common technical skills
    common_skills = [
        "Python", "JavaScript", "TypeScript", "SQL", "PostgreSQL", "MySQL",
        "React", "Node.js", "FastAPI", "Flask", "Django", "Docker", "AWS",
        "Git", "GitHub", "Pandas", "NumPy", "Scikit-learn", "Machine Learning",
        "Deep Learning", "NLP", "C++", "Java", "HTML", "CSS", "Excel"
    ]
    text_lower = resume_text.lower()
    found_skills = [s for s in common_skills if re.search(rf'\b{re.escape(s.lower())}\b', text_lower)]

    return {
        "name": candidate_name,
        "email": email,
        "phone": phone,
        "location": "",
        "current_role": "",
        "total_experience_years": 0.0,
        "experience_years_verified": False,
        "skills": found_skills or ["Python"],
        "projects": [],
        "education": [],
        "experience": []
    }


def profile_generate(pdf_path: str) -> Dict[str, Any]:
    """
    Generates structured profile JSON from PDF resume with multi-model fallback.
    """
    resume_text = extract_text(pdf_path)

    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )

    today_str = datetime.today().strftime("%Y-%m-%d")

    prompt_template = load_prompt("resume_profile")
    prompt = prompt_template.format(today_str=today_str, resume_text=resume_text)

    parsed_profile = None
    last_error = None

    for model_name in settings.OPENROUTER_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1,
                extra_body={"models": settings.OPENROUTER_MODELS[:3]}
            )
            raw = response.choices[0].message.content.strip()

            # Validate and parse JSON inside the retry loop
            profile = _clean_and_repair_json(raw)
            if profile and isinstance(profile, dict) and "name" in profile:
                parsed_profile = profile
                print(f"  Parsed successfully using model: {model_name}")
                break
            else:
                print(f"  Model {model_name} output was not valid JSON. Trying next model...")

        except Exception as e:
            last_error = e
            print(f"  Model {model_name} failed: {e}. Trying fallback...")
            time.sleep(1)

    if parsed_profile is None:
        print(f"  [!] All OpenRouter models failed for {pdf_path} ({last_error}). Using heuristic profile.")
        parsed_profile = _fallback_profile_extract(pdf_path, resume_text)

    # Ensure all required schema fields exist
    parsed_profile.setdefault("name", os.path.splitext(os.path.basename(pdf_path))[0])
    parsed_profile.setdefault("email", "")
    parsed_profile.setdefault("phone", "")
    parsed_profile.setdefault("location", "")
    parsed_profile.setdefault("current_role", "")
    parsed_profile.setdefault("skills", [])
    parsed_profile.setdefault("projects", [])
    parsed_profile.setdefault("education", [])
    parsed_profile.setdefault("experience", [])

    # Date math verification
    verified_years, was_verified = get_verified_experience_years(parsed_profile)
    parsed_profile["total_experience_years"] = verified_years
    parsed_profile["experience_years_verified"] = was_verified

    return parsed_profile
