"""
experience_utils.py

Recomputes total_experience_years in plain Python from each experience
entry's start_date / end_date, instead of trusting the number Gemini
self-reports in the JSON. LLM arithmetic is not reliable enough to use
directly for filtering decisions (e.g. "candidates with 3+ years exp"),
so we treat Gemini's number as a fallback only, and prefer the
recomputed value whenever the dates are present and parseable.
"""

from datetime import datetime


def _parse_yyyymm(value):
    """
    Parses a 'YYYY-MM' string into a datetime (first of that month).
    Returns None if it can't be parsed (missing, malformed, etc).
    Handles 'Present' / 'Current' as today's date.
    """
    if not value:
        return None

    value = value.strip()

    if value.lower() in ("present", "current", "ongoing", "now"):
        return datetime.today().replace(day=1)

    try:
        return datetime.strptime(value, "%Y-%m")
    except ValueError:
        return None


def compute_total_experience_years(experience_list):
    """
    Sums the duration (in years) of every experience entry that has
    parseable start_date and end_date fields. Entries with missing or
    unparseable dates are skipped (not counted), since we can't safely
    guess their length.

    Returns a tuple: (total_years, all_entries_had_valid_dates)
    - total_years: float, rounded to 1 decimal
    - all_entries_had_valid_dates: True only if every experience entry
      contributed a valid duration. False means at least one entry was
      skipped due to missing/bad dates, so the total is a partial sum
      and the caller may want to fall back to Gemini's self-reported
      number for that candidate instead.
    """
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
        # Count the starting month itself (e.g. Jun-Aug = 3 months, not 2)
        months += 1

        if months < 0:
            # End date before start date - bad data, skip rather than
            # subtract incorrectly.
            all_valid = False
            continue

        total_months += months

    total_years = round(total_months / 12, 1)
    return total_years, all_valid


def get_verified_experience_years(profile):
    """
    Given a parsed resume profile dict, returns the experience-years
    figure to actually use for filtering:
    - If every experience entry had valid start/end dates, use the
      Python-computed total (trustworthy, exact).
    - Otherwise, fall back to Gemini's self-reported
      'total_experience_years' field, since we can't fully verify it
      but it's better than a partial/wrong sum.

    Also returns a flag indicating whether the figure was verified,
    so calling code (or the chatbot) can be transparent about
    confidence if needed.
    """
    computed_years, all_valid = compute_total_experience_years(
        profile.get("experience", [])
    )

    if all_valid:
        return computed_years, True  # verified

    fallback = profile.get("total_experience_years", 0)
    try:
        fallback = float(fallback)
    except (TypeError, ValueError):
        fallback = 0.0

    return fallback, False  # not fully verified, using Gemini's number


if __name__ == "__main__":
    # Quick manual test using the sample profile structure
    sample_experience = [
        {
            "role": "Agent Architect Intern",
            "company": "GlobalLogic",
            "start_date": "2025-06",
            "end_date": "2025-08",
        },
        {
            "role": "Data Science and AI Intern",
            "company": "Incedo Inc",
            "start_date": "2024-06",
            "end_date": "2024-08",
        },
    ]

    years, valid = compute_total_experience_years(sample_experience)
    print(f"Computed years: {years}, all_valid: {valid}")