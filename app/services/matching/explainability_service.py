"""
explainability_service.py

Generates explainable recruiter insights for candidate match rankings.
"""

from typing import Dict, Any, List


def generate_candidate_explanation(parsed_jd: Dict[str, Any], candidate_profile: Dict[str, Any], scoring_result: Dict[str, Any]) -> Dict[str, Any]:
    skill_details = scoring_result.get("skill_details") or {}
    matched_req = skill_details.get("matched_required") or []
    missing_req = skill_details.get("missing_required") or []
    matched_pref = skill_details.get("matched_preferred") or []
    missing_pref = skill_details.get("missing_preferred") or []

    all_matched = skill_details.get("all_matched") or []
    all_missing = skill_details.get("all_missing") or []

    min_exp = float(parsed_jd.get("min_experience_years") or 0.0)
    cand_exp = float(candidate_profile.get("total_experience_years") or 0.0)
    composite_score = scoring_result.get("composite_score", 0.0)

    strengths = []
    weaknesses = []

    if min_exp > 0:
        if cand_exp >= min_exp:
            diff = round(cand_exp - min_exp, 1)
            if diff > 0:
                strengths.append(f"Exceeds minimum experience requirement ({cand_exp} yrs vs {min_exp} yrs required).")
            else:
                strengths.append(f"Meets exact experience requirement ({cand_exp} yrs required).")
        else:
            gap = round(min_exp - cand_exp, 1)
            weaknesses.append(f"Experience gap: candidate has {cand_exp} yrs vs {min_exp} yrs required.")
    elif cand_exp > 0:
        strengths.append(f"Has {cand_exp} years of relevant industry experience.")

    if matched_req:
        strengths.append(f"Matches required core skills: {', '.join(matched_req[:4])}.")
    if matched_pref:
        strengths.append(f"Has preferred skills: {', '.join(matched_pref[:3])}.")

    if missing_req:
        weaknesses.append(f"Missing required core skills: {', '.join(missing_req[:3])}.")
    if missing_pref:
        weaknesses.append(f"Missing preferred skills: {', '.join(missing_pref[:3])}.")

    current_role = candidate_profile.get("current_role")
    if current_role:
        role_title = parsed_jd.get("role_title") or ""
        if role_title.lower() in current_role.lower():
            strengths.append(f"Current role ('{current_role}') directly aligns with target role.")

    if not strengths:
        strengths.append("Demonstrates foundational background in software development.")
    if not weaknesses:
        weaknesses.append("No critical missing skills or experience gaps identified.")

    cand_name = candidate_profile.get("name") or "Candidate"
    
    if composite_score >= 75:
        fit_level = "High-priority match"
    elif composite_score >= 50:
        fit_level = "Moderate candidate match"
    else:
        fit_level = "Partial match"

    req_count = len(matched_req)
    total_req = len(matched_req) + len(missing_req)
    req_summary = f"{req_count}/{total_req} required skills matched" if total_req > 0 else "Skills evaluated"

    explanation = (
        f"{cand_name} is ranked as a {fit_level} with a {composite_score}% composite hybrid score. "
        f"{req_summary}. Candidate holds {cand_exp} years of total experience "
        f"{f'(vs {min_exp} yrs required)' if min_exp > 0 else ''}. "
        f"{'Strongest alignment in ' + ', '.join(matched_req[:3]) + '.' if matched_req else ''}"
    )

    return {
        "matched_skills": all_matched,
        "missing_skills": all_missing,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "explanation": explanation
    }
