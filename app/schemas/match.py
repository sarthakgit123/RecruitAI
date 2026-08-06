"""
match.py

Pydantic schemas for JD Match endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    semantic: float
    skills: float
    experience: float
    projects: float
    keywords: float


class CandidateResult(BaseModel):
    rank: int
    resume: str
    candidate_name: str
    similarity: float
    breakdown: Optional[ScoreBreakdown] = None
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    experience_years: float = 0.0
    current_role: str = ""


class MatchAPIResponse(BaseModel):
    status: str
    top_candidates: List[CandidateResult]
