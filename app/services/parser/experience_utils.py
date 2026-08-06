"""
experience_utils.py

Recomputes total_experience_years in plain Python from each experience
entry's start_date / end_date, instead of trusting the number LLM
self-reports in the JSON.
"""

from datetime import datetime
from typing import List, Dict, Any, Tuple


def _parse_yyyymm(value: str) -> datetime | None:
    if not value:
        return None

    value = value.strip()

    if value.lower() in ("present", "current", "ongoing", "now"):
        return datetime.today().replace(day=1)

    try:
        return datetime.strptime(value, "%Y-%m")
    except ValueError:
        return None


def compute_total_experience_years(experience_list: List[Dict[str, Any]]) -> Tuple[float, bool]:
    if not experience_list:
        return 0.0, True

    total_months = 0
    all_valid = True

    for entry in experience_list:
        start = _parse_yyyymm(entry.get("start_date", ""))
        end = _parse_yyyymm(entry.get("end_date", ""))

        if start is None or end is None:
            all_valid = False
            continue

        months = (end.year - start.year) * 12 + (end.month - start.month)
        months += 1

        if months < 0:
            all_valid = False
            continue

        total_months += months

    total_years = round(total_months / 12, 1)
    return total_years, all_valid


def get_verified_experience_years(profile: Dict[str, Any]) -> Tuple[float, bool]:
    computed_years, all_valid = compute_total_experience_years(
        profile.get("experience", [])
    )

    if all_valid:
        return computed_years, True

    fallback = profile.get("total_experience_years", 0)
    try:
        fallback = float(fallback)
    except (TypeError, ValueError):
        fallback = 0.0

    return fallback, False
