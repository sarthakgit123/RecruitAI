"""
faiss_service.py

Builds whole-resume FAISS vector index for fast Top-K retrieval.
"""

import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings


def build_faiss_index() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = []
    resume_names = []

    if not os.path.exists(settings.PROFILES_DIR):
        print(f"Profiles folder not found: {settings.PROFILES_DIR}")
        return

    profile_files = [f for f in os.listdir(settings.PROFILES_DIR) if f.endswith(".json")]

    for file in profile_files:
        json_path = settings.PROFILES_DIR / file

        with open(json_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        skills_text = " ".join(profile.get("skills", []))

        projects_text = " ".join([
            f"""
            {project.get('title', '')}
            {project.get('description', [])}
            {project.get('technologies', [])}
            """
            for project in profile.get("projects", [])
        ])

        education_text = " ".join([
            f"""
            {edu.get('degree', '')}
            {edu.get('institution', '')}
            {edu.get('location', '')}
            {edu.get('dates', '')}
            """
            for edu in profile.get("education", [])
        ])

        experience_text = " ".join([
            f"""
            {exp.get('title', '')}
            {exp.get('company', '')}
            {exp.get('location', '')}
            {exp.get('dates', '')}
            {exp.get('description', '')}
            """
            for exp in profile.get("experience", [])
        ])

        profile_text = f"""
        Candidate Name:
        {profile.get('name', '')}

        Skills:
        {skills_text}

        Projects:
        {projects_text}

        Experience:
        {experience_text}

        Education:
        {education_text}
        """

        embedding = model.encode(profile_text)
        embeddings.append(embedding)
        resume_names.append(file.replace(".json", ".pdf"))

        print(f"Indexed: {file}")

    if not embeddings:
        print("No profiles found to index.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(settings.FAISS_DB_DIR, exist_ok=True)
    faiss.write_index(index, str(settings.RESUME_INDEX_PATH))

    with open(settings.RESUME_NAMES_PATH, "wb") as f:
        pickle.dump(resume_names, f)

    print("\nFAISS Index Created Successfully")
    print(f"Total Resumes Indexed: {len(resume_names)}")
