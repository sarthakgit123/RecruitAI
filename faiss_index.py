import os
import json
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = []
resume_names = []

profile_folder = "profiles"

for file in os.listdir(profile_folder):

    if file.endswith(".json"):

        json_path = os.path.join(
            profile_folder,
            file
        )

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            profile = json.load(f)

     
        profile_text = f"""
        Name: {profile.get('name', '')}

        Skills:
        {' '.join(profile.get('skills', []))}

        Projects:
        {' '.join(profile.get('projects', []))}

        Education:
        {' '.join(profile.get('education', []))}

        Experience:
        {' '.join(profile.get('experience', []))}
        """

        embedding = model.encode(
            profile_text
        )

        embeddings.append(embedding)

        resume_names.append(
            file.replace(".json", ".pdf")
        )

        print(f"Indexed: {file}")


embeddings = np.array(
    embeddings
).astype("float32")

print("Embedding Shape:", embeddings.shape)


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

# Save FAISS index
faiss.write_index(
    index,
    "resume_index.faiss"
)


with open(
    "resume_names.pkl",
    "wb"
) as f:

    pickle.dump(
        resume_names,
        f
    )

print("\nFAISS Index Created Successfully")
print(f"Total Resumes Indexed: {len(resume_names)}")