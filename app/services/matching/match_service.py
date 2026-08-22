"""
match_service.py

Hybrid Resume-JD Matching Architecture:
1. Structured JD Parsing: Extracts role, min_exp, required/preferred skills, domain keywords.
2. FAISS Top-K Retrieval: Uses vector embeddings for fast initial candidate retrieval.
3. Hybrid Reranker: Calculates multi-factor composite scores.
4. Explainable Output: Generates matched/missing skills, strengths, weaknesses, and ranking reasons.
5. PostgreSQL Persistence: Saves JD, parsed JD, and match results to database.
"""

import os
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.parser.jd_parser_service import parse_jd
from app.services.matching.hybrid_matcher import compute_hybrid_score
from app.services.matching.explainability_service import generate_candidate_explanation
from app.database.models import JobDescription, MatchResult, Candidate

_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def _load_profile(resume_filename: str) -> Dict[str, Any]:
    candidate_id = os.path.splitext(resume_filename)[0]
    json_path = settings.PROFILES_DIR / f"{candidate_id}.json"
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "name": candidate_id.replace("_", " "),
        "current_role": "",
        "total_experience_years": 0.0,
        "skills": [],
        "projects": [],
        "experience": [],
        "education": []
    }


def _save_match_to_db(
    db: Session,
    jd_text: str,
    parsed_jd: Dict[str, Any],
    results: List[Dict[str, Any]]
) -> None:
    """
    Save the JD and all match results to PostgreSQL.
    
    Flow:
      1. Insert the raw JD text + parsed JD into job_descriptions table.
      2. For each ranked candidate, look up their UUID from the candidates table.
      3. Insert the rank, score, breakdown, and explanation into match_results table.
    """
    # Save Job Description
    jd_record = JobDescription(
        role_title=parsed_jd.get("role_title", ""),
        raw_jd_text=jd_text,
        parsed_jd=parsed_jd
    )
    db.add(jd_record)
    db.flush()  # flush to get jd_record.id assigned without committing yet

    # Save each match result
    for result in results:
        # Look up the candidate's UUID from the candidates table
        resume_name = result.get("resume", "")
        candidate_id_str = os.path.splitext(resume_name)[0]
        candidate_row = (
            db.query(Candidate)
            .filter(Candidate.candidate_id == candidate_id_str)
            .first()
        )

        if candidate_row is None:
            # Candidate not in DB yet (edge case: uploaded via JSON only, not through upload-zip)
            print(f"  DB: Skipping match result for {candidate_id_str} (not in candidates table)")
            continue

        match_record = MatchResult(
            job_id=jd_record.id,
            candidate_id=candidate_row.id,
            rank=result.get("rank", 0),
            composite_score=result.get("similarity", 0.0),
            score_breakdown=result.get("breakdown", {}),
            matched_skills=result.get("matched_skills", []),
            missing_skills=result.get("missing_skills", []),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            explanation=result.get("explanation", "")
        )
        db.add(match_record)

    db.commit()
    print(f"  DB: Saved JD '{parsed_jd.get('role_title', 'N/A')}' + {len(results)} match results")


def process_resumes_and_match(
    jd_text: str,
    db: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """
    Main matching pipeline.
    
    Args:
        jd_text: Raw job description text from the user.
        db: SQLAlchemy session. If provided, JD and match results are saved to PostgreSQL.
    
    Returns:
        List of ranked candidate result dicts.
    """
    if not os.path.exists(settings.RESUME_INDEX_PATH) or not os.path.exists(settings.RESUME_NAMES_PATH):
        return []

    parsed_jd = parse_jd(jd_text)

    index = faiss.read_index(str(settings.RESUME_INDEX_PATH))
    with open(settings.RESUME_NAMES_PATH, "rb") as f:
        resume_names = pickle.load(f)

    if not resume_names:
        return []

    model = _get_embedding_model()
    jd_embedding = model.encode(jd_text, convert_to_numpy=True).astype(np.float32)
    jd_embedding = np.expand_dims(jd_embedding, axis=0)
    faiss.normalize_L2(jd_embedding)

    top_k = min(len(resume_names), max(10, len(resume_names)))
    scores, indices = index.search(jd_embedding, top_k)

    scored_candidates = []

    for rank_idx, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(resume_names):
            continue

        resume_file = resume_names[idx]
        profile = _load_profile(resume_file)
        raw_similarity = float(scores[0][rank_idx] * 100.0)

        scoring_res = compute_hybrid_score(parsed_jd, profile, raw_semantic_score=raw_similarity)
        explanation_res = generate_candidate_explanation(parsed_jd, profile, scoring_res)

        candidate_name = profile.get("name") or os.path.splitext(resume_file)[0].replace("_", " ")

        scored_candidates.append({
            "resume": resume_file,
            "candidate_name": candidate_name,
            "similarity": scoring_res["composite_score"],
            "breakdown": {
                "semantic": scoring_res["semantic_score"],
                "skills": scoring_res["skill_score"],
                "experience": scoring_res["experience_score"],
                "projects": scoring_res["project_score"],
                "keywords": scoring_res["keyword_score"]
            },
            "matched_skills": explanation_res["matched_skills"],
            "missing_skills": explanation_res["missing_skills"],
            "strengths": explanation_res["strengths"],
            "weaknesses": explanation_res["weaknesses"],
            "explanation": explanation_res["explanation"],
            "experience_years": profile.get("total_experience_years", 0.0),
            "current_role": profile.get("current_role", "")
        })

    scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)

    final_results = []
    for rank, cand in enumerate(scored_candidates, start=1):
        final_results.append({
            "rank": rank,
            **cand
        })

    # Save JD and match results to PostgreSQL
    if db is not None:
        try:
            _save_match_to_db(db, jd_text, parsed_jd, final_results)
        except Exception as e:
            print(f"  DB: Failed to save match results: {e}")

    return final_results
