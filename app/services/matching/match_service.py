"""
match_service.py

Hybrid Resume-JD Matching Architecture:
1. Structured JD Parsing: Extracts role, min_exp, required/preferred skills, domain keywords.
2. FAISS Top-K Retrieval: Uses vector embeddings for fast initial candidate retrieval.
3. Hybrid Reranker: Calculates multi-factor composite scores.
4. Explainable Output: Generates matched/missing skills, strengths, weaknesses, and ranking reasons.
"""

import os
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.parser.jd_parser_service import parse_jd
from app.services.matching.hybrid_matcher import compute_hybrid_score
from app.services.matching.explainability_service import generate_candidate_explanation

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


def process_resumes_and_match(jd_text: str) -> List[Dict[str, Any]]:
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

    return final_results
