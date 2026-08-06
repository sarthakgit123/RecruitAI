"""
hybrid_matcher.py

Multi-Factor Hybrid Scoring Engine.
Calculates weighted composite score combining:
  1. Semantic Vector Similarity (25%)
  2. Weighted Skill Match (35%)
  3. Experience Verification Match (20%)
  4. Project Relevance (10%)
  5. Keyword Coverage (10%)
"""

from typing import Dict, Any
from app.services.matching.skill_normalizer import match_skills, normalize_skill_list


def compute_hybrid_score(parsed_jd: Dict[str, Any], candidate_profile: Dict[str, Any], raw_semantic_score: float = 50.0) -> Dict[str, Any]:
    sem_score = max(0.0, min(100.0, float(raw_semantic_score)))

    req_skills = parsed_jd.get("required_skills") or []
    pref_skills = parsed_jd.get("preferred_skills") or []
    cand_skills = candidate_profile.get("skills") or []

    skill_details = match_skills(cand_skills, req_skills, pref_skills)
    skill_score = skill_details["weighted_skill_score"]

    min_exp = float(parsed_jd.get("min_experience_years") or 0.0)
    cand_exp = float(candidate_profile.get("total_experience_years") or 0.0)

    if min_exp <= 0:
        exp_score = 100.0
    elif cand_exp >= min_exp:
        exp_score = 100.0
    else:
        ratio = cand_exp / min_exp
        exp_score = max(25.0, round(ratio * 100.0, 1))

    projects = candidate_profile.get("projects") or []
    domain_kws = set(s.lower() for s in (parsed_jd.get("domain_keywords") or []))
    all_jd_skills = set(s.lower() for s in normalize_skill_list(req_skills + pref_skills))

    proj_matches = 0
    total_proj_techs = 0

    for proj in projects:
        techs = normalize_skill_list(proj.get("technologies") or [])
        title = (proj.get("title") or "").lower()
        desc = " ".join(proj.get("description") or []).lower()

        for tech in techs:
            total_proj_techs += 1
            if tech.lower() in all_jd_skills or tech.lower() in domain_kws:
                proj_matches += 1

        for kw in domain_kws:
            if kw in title or kw in desc:
                proj_matches += 1

    if total_proj_techs > 0 or projects:
        proj_score = min(100.0, max(30.0, round((proj_matches / max(1, total_proj_techs)) * 100.0, 1)))
    else:
        proj_score = 50.0

    text_corpus = (
        " ".join(candidate_profile.get("skills") or []) + " " +
        candidate_profile.get("current_role", "") + " " +
        " ".join(exp.get("role", "") + " " + " ".join(exp.get("description") or []) for exp in candidate_profile.get("experience") or [])
    ).lower()

    target_keywords = set(s.lower() for s in (parsed_jd.get("domain_keywords") or []) + req_skills + pref_skills)
    if target_keywords:
        kw_hits = sum(1 for kw in target_keywords if kw in text_corpus)
        kw_score = round((kw_hits / len(target_keywords)) * 100.0, 1)
    else:
        kw_score = 70.0

    composite = (
        (0.25 * sem_score) +
        (0.35 * skill_score) +
        (0.20 * exp_score) +
        (0.10 * proj_score) +
        (0.10 * kw_score)
    )
    
    composite_score = round(max(0.0, min(100.0, composite)), 2)

    return {
        "composite_score": composite_score,
        "semantic_score": round(sem_score, 1),
        "skill_score": round(skill_score, 1),
        "experience_score": round(exp_score, 1),
        "project_score": round(proj_score, 1),
        "keyword_score": round(kw_score, 1),
        "skill_details": skill_details
    }
