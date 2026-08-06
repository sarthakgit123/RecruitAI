"""
pdf_service.py

Processes PDF resumes in settings.RESUMES_DIR into JSON profiles.
"""

import os
import json
from app.core.config import settings
from app.services.parser.profile_service import profile_generate


def process_all_resumes() -> None:
    os.makedirs(settings.PROFILES_DIR, exist_ok=True)

    if not os.path.isdir(settings.RESUMES_DIR):
        print(f"Upload folder not found: {settings.RESUMES_DIR}")
        return

    for root, dirs, files in os.walk(settings.RESUMES_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                json_filename = os.path.splitext(file)[0] + ".json"
                json_path = settings.PROFILES_DIR / json_filename

                # Skip LLM API call if profile JSON already exists and is non-empty
                if os.path.exists(json_path) and os.path.getsize(json_path) > 10:
                    print(f"Skipping (already processed): {file}")
                    continue

                print(f"Processing: {pdf_path}")

                try:
                    profile = profile_generate(pdf_path)

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(profile, f, indent=4, ensure_ascii=False)

                    print(f"Saved: {json_path}")

                except Exception as e:
                    print(f"Failed: {file}")
                    print(e)

    print("\nAll resumes processed.")
