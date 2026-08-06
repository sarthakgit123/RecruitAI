"""
profile_service.py

Generates structured candidate profile JSON from PDF resume using OpenRouter LLM.
"""

import os
import time
import json
import re
from datetime import datetime
from typing import Dict, Any
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


def profile_generate(pdf_path: str) -> Dict[str, Any]:
    """
    Generates structured profile JSON from PDF resume.
    """
    resume_text = extract_text(pdf_path)

    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )

    today_str = datetime.today().strftime("%Y-%m-%d")

    prompt_template = load_prompt("resume_profile")
    prompt = prompt_template.format(today_str=today_str, resume_text=resume_text)

    response = None
    last_error = None

    for model_name in settings.OPENROUTER_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.1,
                extra_body={"models": settings.OPENROUTER_MODELS[:3]}
            )
            break
        except Exception as e:
            last_error = e
            print(f"Model {model_name} failed: {e}. Trying fallback...")
            time.sleep(1)

    if response is None:
        raise ValueError(f"All OpenRouter fallback models failed for {pdf_path}: {last_error}")

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON for {pdf_path}: {e}\nRaw output:\n{raw}"
        )

    verified_years, was_verified = get_verified_experience_years(profile)
    profile["total_experience_years"] = verified_years
    profile["experience_years_verified"] = was_verified

    return profile
