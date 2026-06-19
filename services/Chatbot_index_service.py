"""
chatbot_index_service.py

Builds a SEPARATE FAISS index for the chatbot, distinct from the
whole-resume index used by the JD-matcher (resume_index.faiss).

Why a separate index:
- The JD-matcher embeds one vector per whole resume - good for "how
  well does this resume match this JD" holistic comparisons.
- The chatbot needs section-level retrieval - "who used LangChain in a
  project" should match the PROJECT chunk, not just "this whole resume
  is generally similar." So each resume is split into multiple chunks
  (skills, each project, each experience entry, education), and each
  chunk gets its own vector.

Every vector is paired with a metadata record so a FAISS match can
always be traced back to: which candidate, which section, and the
exact text that was embedded. This is the link that was missing in the
original resume_names.pkl (which only stored a filename per resume).

Output files (in faiss_db/):
- chatbot_index.faiss     -> the FAISS index itself
- chatbot_metadata.pkl    -> list of dicts, one per vector, same order
                             as vectors in the index (position i in the
                             list corresponds to vector i in the index)
"""

import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Anchor paths to this file's location so the script works regardless
# of the current working directory it's run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PROFILES_DIR = os.path.join(PROJECT_ROOT, "profiles")
FAISS_DB_DIR = os.path.join(PROJECT_ROOT, "faiss_db")

CHATBOT_INDEX_PATH = os.path.join(FAISS_DB_DIR, "chatbot_index.faiss")
CHATBOT_METADATA_PATH = os.path.join(FAISS_DB_DIR, "chatbot_metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _candidate_id_from_filename(filename):
    """Use the JSON filename (without extension) as a stable candidate id."""
    return os.path.splitext(filename)[0]


def build_chunks_for_profile(profile, candidate_id):
    """
    Splits one candidate's profile JSON into a list of chunk dicts.
    Each chunk dict has:
        - candidate_id: links back to the source resume/JSON file
        - chunk_type: "skills" | "project" | "experience" | "education" | "summary"
        - text: the actual text that gets embedded
        - meta: small dict of extra structured fields useful for display
                or filtering (kept lightweight - the full profile JSON
                is loaded separately when needed, this is just enough
                context to make a retrieved chunk self-explanatory)

    Empty sections are skipped entirely - no empty chunks are created.
    """
    chunks = []
    name = profile.get("name", "")

    # --- Summary chunk: name + current role + total experience ---
    # This gives FAISS something sensible to match on for very general
    # questions like "who is a software engineer" without needing a
    # specific skill/project mention.
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

    # --- Skills chunk: one chunk for the whole skills list ---
    skills = profile.get("skills", [])
    if skills:
        skills_text = f"{name} has these skills: " + ", ".join(skills)
        chunks.append({
            "candidate_id": candidate_id,
            "chunk_type": "skills",
            "text": skills_text,
            "meta": {"name": name, "skills": skills},
        })

    # --- One chunk per project (not one chunk for all projects) ---
    # Splitting per-project means a FAISS match can point to the exact
    # project that's relevant, not a vague "something in this resume
    # matched" blob covering every project at once.
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

    # --- One chunk per experience entry ---
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

    # --- Education chunk: one chunk for the whole education list ---
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


def build_chatbot_index():
    os.makedirs(FAISS_DB_DIR, exist_ok=True)

    if not os.path.isdir(PROFILES_DIR):
        print(f"Profiles folder not found: {PROFILES_DIR}")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_chunks = []  # metadata, one entry per vector, same order as embeddings

    profile_files = [f for f in os.listdir(PROFILES_DIR) if f.endswith(".json")]

    if not profile_files:
        print(f"No JSON profiles found in {PROFILES_DIR}")
        return

    for filename in profile_files:
        json_path = os.path.join(PROFILES_DIR, filename)

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

    faiss.write_index(index, CHATBOT_INDEX_PATH)

    with open(CHATBOT_METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\nChatbot FAISS index created successfully")
    print(f"Total chunks indexed: {len(all_chunks)}")
    print(f"Index saved to: {CHATBOT_INDEX_PATH}")
    print(f"Metadata saved to: {CHATBOT_METADATA_PATH}")


if __name__ == "__main__":
    build_chatbot_index()