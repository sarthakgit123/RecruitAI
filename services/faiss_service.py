import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def build_faiss_index():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = []
    resume_names = []

    profile_folder = r"C:\Users\sarth\OneDrive\Desktop\AI-Resume-JD-Matcher\profiles"

    for file in os.listdir(profile_folder):

        if not file.endswith(".json"):
            continue

        json_path = os.path.join(profile_folder, file)

        with open(json_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # ----------------------------
        # Skills
        # ----------------------------
        skills_text = " ".join(
            profile.get("skills", [])
        )

        # ----------------------------
        # Projects
        # ----------------------------
        projects_text = " ".join(
                [
                    f"""
                    {project.get('title', '')}
                    {project.get('description', [])}
                    {project.get('technologies', [])}
                    """
                    for project in profile.get("projects", [])
                ]
            )


        # ----------------------------
        # Education
        # ----------------------------
        education_text = " ".join(
            [
                f"""
                {edu.get('degree', '')}
                {edu.get('institution', '')}
                {edu.get('location', '')}
                {edu.get('dates', '')}
                """
                for edu in profile.get("education", [])
            ]
        )

        # ----------------------------
        # Experience
        # ----------------------------
        experience_text = " ".join(
            [
                f"""
                {exp.get('title', '')}
                {exp.get('company', '')}
                {exp.get('location', '')}
                {exp.get('dates', '')}
                {exp.get('description', '')}
                """
                for exp in profile.get("experience", [])
            ]
        )

        # ----------------------------
        # Final Resume Text
        # ----------------------------
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

        resume_names.append(
            file.replace(".json", ".pdf")
        )

        print(f"Indexed: {file}")

    # ----------------------------
    # Convert to NumPy
    # ----------------------------
    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Inner Product Index
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    # ----------------------------
    # Save Index
    # ----------------------------
    faiss.write_index(
        index,
        "faiss_db/resume_index.faiss"
    )

    with open(
        "faiss_db/resume_names.pkl",
        "wb"
    ) as f:
        pickle.dump(
            resume_names,
            f
        )

    
    print("\nFAISS Index Created Successfully")
    print(f"Total Resumes Indexed: {len(resume_names)}")

