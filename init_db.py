"""
init_db.py

One-time script to create all PostgreSQL tables defined in app/database/models.py.

Run this after setting up your PostgreSQL database and .env file:
    python init_db.py

What it does:
    1. Imports the SQLAlchemy engine (which reads DATABASE_URL from .env)
    2. Imports all ORM models (Candidate, JobDescription, MatchResult, ChatMessage)
    3. Calls Base.metadata.create_all() which generates CREATE TABLE SQL
       for every model and executes it against your PostgreSQL database.

Note: This is safe to run multiple times. If tables already exist,
      PostgreSQL will skip creating them (no data loss).
"""

from app.database.database import engine, Base

# This import is required even though we don't use the classes directly.
# It registers the model classes with Base.metadata so create_all() knows
# which tables to create.
import app.database.models  # noqa: F401


def init_db():
    print("=" * 50)
    print("RecruitAI Database Initialization")
    print("=" * 50)
    print(f"Connecting to: {engine.url}")
    print("Creating tables...")

    Base.metadata.create_all(bind=engine)

    print()
    print("Tables created:")
    for table_name in Base.metadata.tables:
        print(f"  - {table_name}")

    print()
    print("Done! Open pgAdmin and refresh recruitai_db to verify.")
    print("=" * 50)


if __name__ == "__main__":
    init_db()
