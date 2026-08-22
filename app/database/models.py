"""
models.py

SQLAlchemy ORM models defining the database schema for RecruitAI.
Each class below maps directly to a PostgreSQL table.

Tables:
    1. candidates       — Parsed resume profiles with verified metadata
    2. job_descriptions  — Submitted JDs and their structured parse results
    3. match_results     — Hybrid scoring history linking candidates to JDs
    4. chat_messages     — Persistent per-session chatbot conversation logs
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Text, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database.database import Base


# ---------------------------------------------------------------------------
# Table 1: candidates
# ---------------------------------------------------------------------------
class Candidate(Base):
    """
    Stores candidate profile metadata and the full parsed resume JSON.

    Top-level fields (name, email, experience) are extracted into indexed
    SQL columns for fast filtering. The complete structured profile
    (skills, projects, education, experience entries) lives in the
    JSONB column 'full_profile' so PostgreSQL can query inside it.
    """
    __tablename__ = "candidates"

    # Primary key: auto-generated UUID (e.g., "c39a8b12-4e5f-...")
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Human-readable candidate identifier derived from PDF filename
    # e.g., "231230027_Hritik Singh" (from "231230027_Hritik Singh.pdf")
    # unique=True prevents duplicate inserts; index=True enables fast lookups
    candidate_id = Column(String(255), unique=True, index=True, nullable=False)

    # Extracted top-level metadata for fast SQL queries
    candidate_name = Column(String(255), index=True, nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(100), nullable=True)
    location = Column(String(255), nullable=True)
    current_role = Column(String(255), nullable=True)

    # Verified experience (computed by Python date math, not LLM)
    total_experience_years = Column(Float, default=0.0, index=True)
    experience_verified = Column(Boolean, default=False)

    # Full structured resume profile as JSONB
    # Contains: skills[], projects[], education[], experience[]
    # PostgreSQL can query inside this: WHERE full_profile->'skills' @> '["Python"]'
    full_profile = Column(JSONB, nullable=False)

    # Timestamp of when this candidate was first processed
    created_at = Column(DateTime, default=datetime.utcnow)

    # ORM Relationship: Access all match results for this candidate
    # cascade="all, delete-orphan" means deleting a candidate also deletes their match results
    match_results = relationship(
        "MatchResult", back_populates="candidate", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Table 2: job_descriptions
# ---------------------------------------------------------------------------
class JobDescription(Base):
    """
    Stores submitted Job Descriptions and their LLM-parsed structured output.

    raw_jd_text: The original free-text JD pasted by the recruiter.
    parsed_jd:   The structured JSON extracted by jd_parser_service.py
                 (role_title, required_skills, preferred_skills, min_experience_years, etc.)
    """
    __tablename__ = "job_descriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Extracted role title for display (e.g., "Python Backend Developer")
    role_title = Column(String(255), nullable=True)

    # Original raw JD text submitted by the user
    raw_jd_text = Column(Text, nullable=False)

    # LLM-parsed structured JD as JSONB
    # Contains: role_title, min_experience_years, required_skills[],
    #           preferred_skills[], education_requirements, domain_keywords[]
    parsed_jd = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # ORM Relationship: Access all match results for this JD
    match_results = relationship(
        "MatchResult", back_populates="job_description", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Table 3: match_results
# ---------------------------------------------------------------------------
class MatchResult(Base):
    """
    Stores the hybrid scoring output linking a Candidate to a JobDescription.

    Each row represents one candidate's ranking for one specific JD submission.
    This creates a many-to-many relationship between candidates and JDs,
    connected through the match_results table (an association table with data).

    Foreign Keys:
        job_id       → job_descriptions.id
        candidate_id → candidates.id
    """
    __tablename__ = "match_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys linking to parent tables
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("job_descriptions.id"), nullable=False
    )
    candidate_id = Column(
        UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False
    )

    # Ranking position (1 = best match)
    rank = Column(Integer, nullable=False)

    # Final 5-factor hybrid composite score (0.0 to 100.0)
    composite_score = Column(Float, nullable=False)

    # Detailed sub-score breakdown as JSONB
    # Contains: {semantic: 85.2, skills: 70.0, experience: 100.0, projects: 60.0, keywords: 45.5}
    score_breakdown = Column(JSONB, nullable=False)

    # Skill analysis results as JSONB arrays
    matched_skills = Column(JSONB, nullable=True)  # e.g., ["Python", "FastAPI", "PostgreSQL"]
    missing_skills = Column(JSONB, nullable=True)   # e.g., ["Docker", "AWS"]

    # Explainability insights as JSONB arrays of strings
    strengths = Column(JSONB, nullable=True)   # e.g., ["Exceeds min experience by 2 yrs"]
    weaknesses = Column(JSONB, nullable=True)  # e.g., ["Missing required skill: Docker"]

    # Natural language ranking explanation
    explanation = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # ORM Relationships: Navigate to parent JD and Candidate objects
    job_description = relationship("JobDescription", back_populates="match_results")
    candidate = relationship("Candidate", back_populates="match_results")


# ---------------------------------------------------------------------------
# Table 4: chat_messages
# ---------------------------------------------------------------------------
class ChatMessage(Base):
    """
    Replaces the in-memory _chat_history = [] array with persistent
    per-session conversation logs stored in PostgreSQL.

    Each row is one message (either from the user or the assistant).
    Messages are grouped by session_id so multiple recruiters can have
    independent conversation histories simultaneously.
    """
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Groups messages into conversations. Default "default" for single-user mode.
    # In future, this could be a browser cookie ID or authenticated user ID.
    session_id = Column(String(255), index=True, nullable=False, default="default")

    # "user" or "assistant"
    role = Column(String(50), nullable=False)

    # The actual message text
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
