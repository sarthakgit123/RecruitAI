"""
metadata_service.py

Saves profile JSON files into settings.PROFILES_DIR.
"""

import os
import json
from typing import Dict, Any
from app.core.config import settings


def save_profile_metadata(profile: Dict[str, Any], candidate_id: str) -> str:
    """Saves candidate JSON profile to profiles directory."""
    os.makedirs(settings.PROFILES_DIR, exist_ok=True)
    json_filename = f"{candidate_id}.json"
    json_path = settings.PROFILES_DIR / json_filename

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)

    return str(json_path)
