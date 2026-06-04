import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


def process_resumes_and_match(jd):

    index = faiss.read_index(
        "faiss_db/resume_index.faiss"
    )

    with open(
        "faiss_db/resume_names.pkl",
        "rb"
    ) as f:
        resume_names = pickle.load(f)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    jd_embedding = model.encode(
        jd,
        convert_to_numpy=True
    ).astype(np.float32)

    jd_embedding = np.expand_dims(
        jd_embedding,
        axis=0
    )

    faiss.normalize_L2(jd_embedding)


    k = min(5, len(resume_names))

    scores, indices = index.search(
    jd_embedding,
    k
    )

    results = []

    for rank, idx in enumerate(indices[0], start=1):

        similarity = float(
            scores[0][rank - 1] * 100
        )

        results.append(
            {
                "rank": rank,
                "resume": resume_names[idx],
                "similarity": round(similarity, 2)
            }
        )

    return results