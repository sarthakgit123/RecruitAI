"""
chatbot_retrieval.py

Handles the retrieval phase of chatbot RAG: intent extraction, hard filtering, FAISS chunk search.
"""

import os
import json
import re
import pickle
import time
from typing import Dict, Any, List
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.prompts import load_prompt

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model = None
_faiss_index = None
_chunk_metadata = None
_all_profiles_cache = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_faiss_index_and_metadata():
    global _faiss_index, _chunk_metadata
    if _faiss_index is None:
        if not os.path.exists(settings.CHATBOT_INDEX_PATH):
            raise FileNotFoundError(
                f"Chatbot FAISS index not found at {settings.CHATBOT_INDEX_PATH}. "
                "Run chatbot_index_service.py first."
            )
        _faiss_index = faiss.read_index(str(settings.CHATBOT_INDEX_PATH))
        with open(settings.CHATBOT_METADATA_PATH, "rb") as f:
            _chunk_metadata = pickle.load(f)
    return _faiss_index, _chunk_metadata


def _load_all_profiles() -> Dict[str, Any]:
    global _all_profiles_cache
    if _all_profiles_cache is not None:
        return _all_profiles_cache

    profiles = {}
    if not os.path.isdir(settings.PROFILES_DIR):
        return profiles

    for filename in os.listdir(settings.PROFILES_DIR):
        if not filename.endswith(".json"):
            continue
        candidate_id = os.path.splitext(filename)[0]
        with open(settings.PROFILES_DIR / filename, "r", encoding="utf-8") as f:
            profiles[candidate_id] = json.load(f)

    _all_profiles_cache = profiles
    return profiles


def extract_intent(question: str) -> Dict[str, Any]:
    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )

    prompt_template = load_prompt("chatbot_intent")
    prompt = prompt_template.format(question=question)

    response = None
    last_error = None

    for model_name in settings.OPENROUTER_MODELS:
        try:
            result = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,
                extra_body={"models": settings.OPENROUTER_MODELS[:3]}
            )
            response = result.choices[0].message.content
            break
        except Exception as e:
            last_error = e
            print(f"Intent model {model_name} failed: {e}. Trying fallback...")
            time.sleep(1)

    if response is None:
        return {
            "skill": None,
            "min_years": None,
            "role": None,
            "semantic_query": question,
        }

    raw = response.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        intent = {"skill": None, "min_years": None, "role": None, "semantic_query": question}

    fit_language = ("good fit", "suitable", "suited", "would fit", "could work as",
                     "best fit", "right fit", "fit for")
    question_lower = question.lower()
    if intent.get("role") and any(phrase in question_lower for phrase in fit_language):
        intent["role"] = None

    semantic_query = (intent.get("semantic_query") or "").strip()
    if len(semantic_query) < 5:
        parts = [p for p in [intent.get("skill"), intent.get("role")] if p]
        intent["semantic_query"] = " ".join(parts) if parts else question
    else:
        intent["semantic_query"] = semantic_query

    return intent


def apply_hard_filters(intent: Dict[str, Any]) -> List[str]:
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


def semantic_search(query_text: str, allowed_candidate_ids: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
    index, metadata = _get_faiss_index_and_metadata()
    model = _get_embedding_model()

    allowed_set = set(allowed_candidate_ids)

    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)

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


def retrieve(question: str, top_k: int = 4) -> Dict[str, Any]:
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
