"""
chatbot_retrieval.py

Handles the "retrieval" half of the chatbot's RAG pipeline:

1. Intent extraction - one small Gemini call reads the user's question
   and pulls out structured filters (skill, min_years, role) plus
   whatever semantic/fuzzy part remains.
2. Metadata filtering - plain Python, loops over profiles/*.json and
   keeps only candidates that pass the hard filters. No AI involved -
   fast and exact for numeric/boolean conditions that embeddings are
   bad at.
3. FAISS semantic search - embeds the question (or its semantic
   remainder) and searches chatbot_index.faiss, restricted to chunks
   belonging to the filtered candidate set (or all candidates if no
   hard filters applied).

This module does NOT generate the final natural-language answer - see
chatbot_service.py for that. This module only decides WHICH candidates
and WHICH chunks are relevant to a question.
"""

import os
import json
import re
import pickle
import time
import faiss
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PROFILES_DIR = os.path.join(PROJECT_ROOT, "profiles")
FAISS_DB_DIR = os.path.join(PROJECT_ROOT, "faiss_db")

CHATBOT_INDEX_PATH = os.path.join(FAISS_DB_DIR, "chatbot_index.faiss")
CHATBOT_METADATA_PATH = os.path.join(FAISS_DB_DIR, "chatbot_metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loaded once per process, reused across questions - loading these on
# every call would be slow and pointless since the data doesn't change
# between questions in the same run.
_embedding_model = None
_faiss_index = None
_chunk_metadata = None
_all_profiles_cache = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_faiss_index_and_metadata():
    global _faiss_index, _chunk_metadata
    if _faiss_index is None:
        if not os.path.exists(CHATBOT_INDEX_PATH):
            raise FileNotFoundError(
                f"Chatbot FAISS index not found at {CHATBOT_INDEX_PATH}. "
                "Run chatbot_index_service.py first."
            )
        _faiss_index = faiss.read_index(CHATBOT_INDEX_PATH)
        with open(CHATBOT_METADATA_PATH, "rb") as f:
            _chunk_metadata = pickle.load(f)
    return _faiss_index, _chunk_metadata


def _load_all_profiles():
    """
    Loads every profile JSON into memory as {candidate_id: profile_dict}.
    Cached after first call within a process - profiles/ doesn't change
    mid-conversation, so re-reading from disk every question is wasted
    work once you're at 50-300 resumes.
    """
    global _all_profiles_cache
    if _all_profiles_cache is not None:
        return _all_profiles_cache

    profiles = {}
    if not os.path.isdir(PROFILES_DIR):
        return profiles

    for filename in os.listdir(PROFILES_DIR):
        if not filename.endswith(".json"):
            continue
        candidate_id = os.path.splitext(filename)[0]
        with open(os.path.join(PROFILES_DIR, filename), "r", encoding="utf-8") as f:
            profiles[candidate_id] = json.load(f)

    _all_profiles_cache = profiles
    return profiles


def extract_intent(question):
    """
    Uses Gemini to turn a free-text question into structured filters.

    Returns a dict:
        {
            "skill": "Python" or null,
            "min_years": 3 or null,
            "role": "backend engineer" or null,
            "semantic_query": the part of the question that isn't a
                               hard filter, used for FAISS search.
                               Usually the question itself, sometimes
                               trimmed down.
        }

    Kept deliberately simple - only the three filter types we said
    matter most (skill, years, role). Anything more open-ended falls
    through to semantic_query untouched, which FAISS handles via
    similarity search.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
    Return ONLY valid JSON, no markdown, no commentary.

    Read this question about job candidates and extract structured filters.

    Schema:
    {{
        "skill": "" or a specific skill/technology name mentioned, else null,
        "min_years": null or a number if the question mentions a minimum
                      years of experience requirement (e.g. "3+ years",
                      "at least 2 years" -> 2),
        "role": "" or a job title mentioned that the candidate must have
                 ACTUALLY HELD in the past (e.g. "who has worked as a
                 backend engineer" -> "backend engineer"), else null.
                 IMPORTANT: only extract this when the question asks
                 about a role someone has held/done before. Do NOT
                 extract this for questions about fit, suitability, or
                 potential for a role (e.g. "who would be a good fit
                 for a backend role", "who could work as a designer")
                 - those are judgment questions, not factual filters,
                 and must return null here so they go through semantic
                 search instead.
        "semantic_query": a short descriptive phrase capturing what the
                           question is actually about, used for semantic
                           search. This must stay meaningful and specific -
                           NEVER strip it down to a generic phrase like
                           "who knows" or "who has". If the question is
                           "who knows Python", semantic_query should be
                           something like "Python programming skills",
                           not "who knows". If the question has no fuzzy
                           part beyond the hard filters (e.g. "who knows
                           Python" with nothing else), repeat the skill/
                           role here anyway so semantic search still has
                           something specific to match against.
    }}

    Rules:
    - Only fill skill/min_years/role if explicitly present in the question.
    - Do not guess or infer values that aren't stated.
    - For "role": only extract it if the question is asking about a role
      someone has ACTUALLY HELD (past/current employment fact). If the
      question is asking who would be SUITABLE, a GOOD FIT, or COULD do
      a role, that is a judgment call, not a factual filter - leave
      "role" as null and let semantic_query carry the full question.
    - semantic_query must never be empty or vague - it should always
      contain the specific subject of the question (skill name, role,
      project type, etc).
    - Output valid JSON only.

    Question: {question}
    """

    response = None
    last_error = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            break
        except genai_errors.ServerError as e:
            # Gemini occasionally returns 503 UNAVAILABLE under high
            # demand - this is transient, not a bug in our code.
            # Retry a few times with short backoff before giving up.
            last_error = e
            if attempt < max_retries - 1:
                wait_seconds = 2 ** attempt  # 1s, 2s, 4s
                print(f"Gemini server busy (attempt {attempt + 1}/{max_retries}), retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                print(f"Gemini still unavailable after {max_retries} attempts: {e}")

    if response is None:
        # Fail safe rather than crashing the whole chatbot - treat the
        # question as pure semantic search with no hard filters.
        return {
            "skill": None,
            "min_years": None,
            "role": None,
            "semantic_query": question,
        }

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        # If intent extraction fails for any reason, fail safe: treat
        # the whole question as semantic with no hard filters, rather
        # than crashing the chat.
        intent = {"skill": None, "min_years": None, "role": None, "semantic_query": question}

    # Defensive guard: even with explicit prompt instructions, the LLM
    # can still mistakenly extract a "role" out of a fit/suitability
    # question (e.g. "good fit for a backend role" -> role="backend
    # role"), which then wrongly hard-filters on job titles nobody
    # actually holds. If the extracted role or the original question
    # contains fit/suitability language, null it out and let the
    # question go through semantic search instead.
    fit_language = ("good fit", "suitable", "suited", "would fit", "could work as",
                     "best fit", "right fit", "fit for")
    question_lower = question.lower()
    if intent.get("role") and any(phrase in question_lower for phrase in fit_language):
        intent["role"] = None

    # Defensive fallback: don't trust the LLM's semantic_query blindly.
    # If it came back empty or suspiciously short (a sign the model
    # stripped out the actual subject), rebuild it from the extracted
    # filters plus the original question so FAISS search still has
    # something meaningful to embed.
    semantic_query = (intent.get("semantic_query") or "").strip()
    if len(semantic_query) < 5:
        parts = [p for p in [intent.get("skill"), intent.get("role")] if p]
        intent["semantic_query"] = " ".join(parts) if parts else question
    else:
        intent["semantic_query"] = semantic_query

    return intent


def apply_hard_filters(intent):
    """
    Loops over all candidate profiles in plain Python and keeps only
    those matching the hard filters extracted by extract_intent().

    Returns a list of candidate_ids that pass. If no hard filters were
    present in the intent (skill/min_years/role all null), returns ALL
    candidate_ids - meaning the FAISS search step runs unrestricted.
    """
    profiles = _load_all_profiles()

    skill = (intent.get("skill") or "").strip().lower()
    min_years = intent.get("min_years")
    role = (intent.get("role") or "").strip().lower()

    no_filters = not skill and min_years is None and not role
    if no_filters:
        return list(profiles.keys())

    matching_ids = []

    for candidate_id, profile in profiles.items():
        if skill:
            candidate_skills = [s.lower().strip() for s in profile.get("skills", [])]
            if skill not in candidate_skills:
                continue

        if min_years is not None:
            years = profile.get("total_experience_years", 0)
            try:
                years = float(years)
            except (TypeError, ValueError):
                years = 0.0
            if years < float(min_years):
                continue

        if role:
            current_role = (profile.get("current_role") or "").lower()
            past_roles = " ".join(
                exp.get("role", "") for exp in profile.get("experience", [])
            ).lower()
            if role not in current_role and role not in past_roles:
                continue

        matching_ids.append(candidate_id)

    return matching_ids


def semantic_search(query_text, allowed_candidate_ids, top_k=8):
    """
    Embeds query_text and searches the chatbot FAISS index, restricted
    to chunks whose candidate_id is in allowed_candidate_ids.

    Why filter AFTER the FAISS search rather than building a separate
    index per filter combination: FAISS doesn't support arbitrary
    pre-filtering natively in IndexFlatIP, and at 50-300 resumes
    (a few thousand chunks at most) searching the full index then
    discarding non-matching results is fast enough that it's not worth
    the complexity of partitioned indexes.

    Returns a list of chunk dicts (the metadata records), best matches
    first, up to top_k results that belong to an allowed candidate.
    """
    index, metadata = _get_faiss_index_and_metadata()
    model = _get_embedding_model()

    allowed_set = set(allowed_candidate_ids)

    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    # Search more than top_k since some results will be filtered out
    # by candidate_id - over-fetch to make sure we still end up with
    # enough results after filtering.
    search_k = min(len(metadata), max(top_k * 5, 30))
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        if chunk["candidate_id"] not in allowed_set:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def retrieve(question, top_k=8):
    """
    Full retrieval pipeline for one question: intent extraction ->
    hard filtering -> semantic search restricted to the filtered set.

    Returns a dict with everything chatbot_service.py needs to build
    the final answer:
        {
            "intent": the extracted intent dict,
            "matching_candidate_ids": candidates that passed hard filters,
            "chunks": retrieved chunk dicts with scores,
            "no_hard_filter_matches": True if hard filters were present
                                      but matched zero candidates
        }
    """
    intent = extract_intent(question)
    matching_ids = apply_hard_filters(intent)

    no_hard_filter_matches = False
    has_hard_filters = bool(intent.get("skill") or intent.get("min_years") or intent.get("role"))
    if has_hard_filters and len(matching_ids) == 0:
        no_hard_filter_matches = True
        chunks = []
    else:
        semantic_query = intent.get("semantic_query") or question
        chunks = semantic_search(semantic_query, matching_ids, top_k=top_k)

    return {
        "intent": intent,
        "matching_candidate_ids": matching_ids,
        "chunks": chunks,
        "no_hard_filter_matches": no_hard_filter_matches,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Who knows Python?"

    result = retrieve(q)
    print(json.dumps(result, indent=2, default=str))