"""
reset_data.py

Completely cleans all local storage folders (uploads, profiles, faiss_db)
and deletes all rows from PostgreSQL tables for a fresh end-to-end test.
"""

import os
import shutil
from app.core.config import settings
from app.database.database import SessionLocal
from app.database.models import Candidate, JobDescription, MatchResult, ChatMessage


def clear_local_directories():
    print("[*] Cleaning local directories...")

    # Clear uploads/resumes
    if os.path.exists(settings.RESUMES_DIR):
        shutil.rmtree(settings.RESUMES_DIR, ignore_errors=True)
    os.makedirs(settings.RESUMES_DIR, exist_ok=True)
    print(f"  - Cleared: {settings.RESUMES_DIR}")

    # Clear profiles
    if os.path.exists(settings.PROFILES_DIR):
        shutil.rmtree(settings.PROFILES_DIR, ignore_errors=True)
    os.makedirs(settings.PROFILES_DIR, exist_ok=True)
    print(f"  - Cleared: {settings.PROFILES_DIR}")

    # Clear faiss_db
    if os.path.exists(settings.FAISS_DB_DIR):
        shutil.rmtree(settings.FAISS_DB_DIR, ignore_errors=True)
    os.makedirs(settings.FAISS_DB_DIR, exist_ok=True)
    print(f"  - Cleared: {settings.FAISS_DB_DIR}")


def clear_database_tables():
    print("\n[*] Cleaning PostgreSQL tables...")
    db = SessionLocal()
    try:
        # Delete in order of foreign key dependencies
        deleted_matches = db.query(MatchResult).delete()
        deleted_jds = db.query(JobDescription).delete()
        deleted_candidates = db.query(Candidate).delete()
        deleted_chats = db.query(ChatMessage).delete()
        db.commit()

        print(f"  - Deleted {deleted_matches} rows from match_results")
        print(f"  - Deleted {deleted_jds} rows from job_descriptions")
        print(f"  - Deleted {deleted_candidates} rows from candidates")
        print(f"  - Deleted {deleted_chats} rows from chat_messages")
    except Exception as e:
        db.rollback()
        print(f"  [!] Database reset error: {e}")
    finally:
        db.close()


def main():
    print("=" * 60)
    print("RECRUITAI FULL SYSTEM RESET")
    print("=" * 60)
    clear_local_directories()
    clear_database_tables()
    print("\n[OK] System fully reset! Ready for fresh end-to-end testing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
