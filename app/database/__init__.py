"""
Database package.

Contains:
    - database.py: Engine, session factory, Base, and get_db dependency.
    - models.py: SQLAlchemy ORM model definitions for all PostgreSQL tables.
"""

from app.database.database import Base, SessionLocal, engine, get_db
from app.database.models import Candidate, JobDescription, MatchResult, ChatMessage
