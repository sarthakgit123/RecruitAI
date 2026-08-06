"""
jd_parser_service.py

Parses free-text Job Descriptions into structured JSON using OpenRouter LLM.
"""

import os
import json
import re
import time
from typing import Dict, Any
from openai import OpenAI
from app.core.config import settings
from app.prompts import load_prompt


def parse_jd(jd_text: str) -> Dict[str, Any]:
    """
    Parses a Job Description string into a structured JSON dict.
    """
    if not jd_text or not isinstance(jd_text, str):
        return _fallback_jd_parse(jd_text or "")

    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )

    prompt_template = load_prompt("jd_parse")
    prompt = prompt_template.format(jd_text=jd_text[:3500])

    response = None
    last_error = None

    for model_name in settings.OPENROUTER_MODELS:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1,
                extra_body={"models": settings.OPENROUTER_MODELS[:3]}
            )
            response = res.choices[0].message.content
            break
        except Exception as e:
            last_error = e
            print(f"JD Parser model {model_name} failed: {e}. Trying fallback...")
            time.sleep(0.5)

    if response is None:
        print(f"LLM JD parsing unavailable ({last_error}), using heuristic parser.")
        return _fallback_jd_parse(jd_text)

    raw = response.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        
        parsed["min_experience_years"] = float(parsed.get("min_experience_years") or 0.0)
        parsed["required_skills"] = list(parsed.get("required_skills") or [])
        parsed["preferred_skills"] = list(parsed.get("preferred_skills") or [])
        parsed["domain_keywords"] = list(parsed.get("domain_keywords") or [])
        parsed["key_responsibilities"] = list(parsed.get("key_responsibilities") or [])
        parsed["role_title"] = str(parsed.get("role_title") or "Software Engineer")
        parsed["education_requirements"] = str(parsed.get("education_requirements") or "")
        
        return parsed
    except json.JSONDecodeError:
        print("Failed to decode LLM JSON response for JD. Using fallback parser.")
        return _fallback_jd_parse(jd_text)


def _fallback_jd_parse(jd_text: str) -> Dict[str, Any]:
    text_lower = jd_text.lower()
    exp_match = re.search(r'(\d+)\+?\s*(?:-\s*\d+)?\s*(?:years?|yrs?)', text_lower)
    min_exp = float(exp_match.group(1)) if exp_match else 0.0
    
    common_tech = [
        "Python", "JavaScript", "TypeScript", "C++", "Java", "Go", "Rust",
        "React", "Node.js", "FastAPI", "Flask", "Django", "PostgreSQL",
        "MySQL", "MongoDB", "Redis", "AWS", "Docker", "Kubernetes",
        "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "SQL"
    ]
    
    found_skills = [tech for tech in common_tech if tech.lower() in text_lower]
    
    return {
        "role_title": "Software Engineer",
        "min_experience_years": min_exp,
        "required_skills": found_skills[:4] if found_skills else ["Python"],
        "preferred_skills": found_skills[4:] if len(found_skills) > 4 else [],
        "education_requirements": "Bachelor's degree in Computer Science or related field",
        "key_responsibilities": [line.strip() for line in jd_text.split('\n') if len(line.strip()) > 15][:3],
        "domain_keywords": found_skills
    }
