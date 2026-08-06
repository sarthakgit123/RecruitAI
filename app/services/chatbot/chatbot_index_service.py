"""
chatbot_index_service.py

Builds section-level FAISS index and metadata for the chatbot RAG pipeline.
"""

import os
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.core.config import settings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _candidate_id_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0]


def build_chunks_for_profile(profile: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    chunks = []
    name = profile.get("name", "")

    current_role = profile.get("current_role", "")
    total_years = profile.get("total_experience_years", 0)
    if name or current_role:
        summary_text = f"{name}. Current role: {current_role}. Total experience: {total_years} years."
        chunks.append({
            "candidate_id": candidate_id,
            "chunk_type": "summary",
            "text": summary_text,
            "meta": {"name": name, "current_role": current_role, "total_experience_years": total_years},
        })

    skills = profile.get("skills", [])
    if skills:
        skills_text = f"{name} has these skills: " + ", ".join(skills)
        chunks.append({
            "candidate_id": candidate_id,
            "chunk_type": "skills",
            "text": skills_text,
            "meta": {"name": name, "skills": skills},
        })

    for project in profile.get("projects", []):
        title = project.get("title", "")
        tech = ", ".join(project.get("technologies", []))
        desc = " ".join(project.get("description", []))
        project_text = f"{name} worked on project '{title}'. Technologies: {tech}. {desc}"
        chunks.append({
            "candidate_id": candidate_id,
            "chunk_type": "project",
            "text": project_text,
            "meta": {"name": name, "title": title, "technologies": project.get("technologies", [])},
        })

    for exp in profile.get("experience", []):
        role = exp.get("role", "")
        company = exp.get("company", "")
        duration = exp.get("duration", "")
        desc = " ".join(exp.get("description", []))
        exp_text = f"{name} worked as {role} at {company} ({duration}). {desc}"
        chunks.append({
            "candidate_id": candidate_id,
            "chunk_type": "experience",
            "text": exp_text,
            "meta": {"name": name, "role": role, "company": company, "duration": duration},
        })

    education = profile.get("education", [])
    if education:
        edu_parts = [
            f"{edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('year', '')})"
            for edu in education
        ]
        edu_text = f"{name}'s education: " + "; ".join(edu_parts)
        chunks.append({
            "candidate_id": candidate_id,
            "chunk_type": "education",
            "text": edu_text,
            "meta": {"name": name, "education": education},
        })

    return chunks


def build_chatbot_index() -> None:
    os.makedirs(settings.FAISS_DB_DIR, exist_ok=True)

    if not os.path.isdir(settings.PROFILES_DIR):
        print(f"Profiles folder not found: {settings.PROFILES_DIR}")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_chunks = []
    profile_files = [f for f in os.listdir(settings.PROFILES_DIR) if f.endswith(".json")]

    if not profile_files:
        print(f"No JSON profiles found in {settings.PROFILES_DIR}")
        return

    for filename in profile_files:
        json_path = settings.PROFILES_DIR / filename

        with open(json_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        candidate_id = _candidate_id_from_filename(filename)
        chunks = build_chunks_for_profile(profile, candidate_id)

        if not chunks:
            print(f"Skipped (no extractable content): {filename}")
            continue

        all_chunks.extend(chunks)
        print(f"Chunked: {filename} -> {len(chunks)} chunks")

    if not all_chunks:
        print("No chunks were generated. Nothing to index.")
        return

    print(f"\nEmbedding {len(all_chunks)} chunks total...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(settings.CHATBOT_INDEX_PATH))

    with open(settings.CHATBOT_METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\nChatbot FAISS index created successfully")
    print(f"Total chunks indexed: {len(all_chunks)}")
