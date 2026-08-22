"""
pdf_service.py

Processes PDF resumes in settings.RESUMES_DIR into JSON profiles.
Also saves candidate profiles to PostgreSQL candidates table.
"""

import os
import json
from typing import Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.parser.profile_service import profile_generate
from app.database.models import Candidate


def _save_candidate_to_db(db: Session, candidate_id: str, profile: dict) -> None:
    """
    Save a parsed candidate profile into the PostgreSQL candidates table.
    If the candidate already exists (by candidate_id), update their record.
    """
    existing = db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()

    if existing:
        # Update existing record with fresh profile data
        existing.candidate_name = profile.get("name", "")
        existing.email = profile.get("email", "")
        existing.phone = profile.get("phone", "")
        existing.location = profile.get("location", "")
        existing.current_role = profile.get("current_role", "")
        existing.total_experience_years = float(profile.get("total_experience_years", 0.0))
        existing.experience_verified = profile.get("experience_years_verified", False)
        existing.full_profile = profile
        print(f"  DB: Updated existing candidate: {candidate_id}")
    else:
        # Insert new candidate
        candidate = Candidate(
            candidate_id=candidate_id,
            candidate_name=profile.get("name", ""),
            email=profile.get("email", ""),
            phone=profile.get("phone", ""),
            location=profile.get("location", ""),
            current_role=profile.get("current_role", ""),
            total_experience_years=float(profile.get("total_experience_years", 0.0)),
            experience_verified=profile.get("experience_years_verified", False),
            full_profile=profile
        )
        db.add(candidate)
        print(f"  DB: Inserted new candidate: {candidate_id}")

    db.commit()


def process_all_resumes(db: Optional[Session] = None) -> None:
    """
    Parse all PDF resumes into JSON profiles and optionally save to PostgreSQL.
    
    Args:
        db: SQLAlchemy session. If provided, candidates are also saved to PostgreSQL.
            If None, only JSON file saving is performed (backwards compatible).
    """
    os.makedirs(settings.PROFILES_DIR, exist_ok=True)

    if not os.path.isdir(settings.RESUMES_DIR):
        print(f"Upload folder not found: {settings.RESUMES_DIR}")
        return

    for root, dirs, files in os.walk(settings.RESUMES_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                candidate_id = os.path.splitext(file)[0]
                json_filename = candidate_id + ".json"
                json_path = settings.PROFILES_DIR / json_filename

                # Skip LLM API call if profile JSON already exists and is non-empty
                if os.path.exists(json_path) and os.path.getsize(json_path) > 10:
                    print(f"Skipping (already processed): {file}")

                    # Still save to DB if not already there
                    if db is not None:
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                profile = json.load(f)
                            _save_candidate_to_db(db, candidate_id, profile)
                        except Exception as e:
                            print(f"  DB save failed for {file}: {e}")
                    continue

                print(f"Processing: {pdf_path}")

                try:
                    profile = profile_generate(pdf_path)

                    # Save JSON file (FAISS still needs these)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(profile, f, indent=4, ensure_ascii=False)
                    print(f"Saved: {json_path}")

                    # Save to PostgreSQL
                    if db is not None:
                        _save_candidate_to_db(db, candidate_id, profile)

                except Exception as e:
                    print(f"Failed: {file}")
                    print(e)

    print("\nAll resumes processed.")
