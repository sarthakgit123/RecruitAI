import os
import json
import pickle

PROFILE_DIR = "profiles"
OUTPUT_FILE = "faiss_db/metadata.pkl"


def build_metadata():

    candidates = []

    for filename in os.listdir(PROFILE_DIR):

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(PROFILE_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        candidate = {
            "name": data.get("name", ""),
            "email": data.get("email", ""),
            "location": data.get("location", ""),
            "current_role": data.get("current_role", ""),
            "experience": data.get("total_experience_years", 0),
            "skills": data.get("skills", []),
            "json_path": filepath
        }

        candidates.append(candidate)

    with open("faiss_db/metadata.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4)

    print(f"Saved {len(candidates)} candidates")

