import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from services.pdf_service import process_all_resumes
from services.faiss_service import build_faiss_index


def process_resumes_and_match(jd):

    # Generate profiles from PDFs
    process_all_resumes()

    # Create embeddings + FAISS index
    build_faiss_index()

    # Load FAISS index
    index = faiss.read_index(
        "faiss_db/resume_index.faiss"
    )

    # Load resume names
    with open(
        "faiss_db/resume_names.pkl",
        "rb"
    ) as f:

        resume_names = pickle.load(f)

    # Load embedding model
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # JD embedding
    jd_embedding = model.encode(
        jd,
        convert_to_numpy=True
    ).astype(np.float32)

    jd_embedding = np.expand_dims(
        jd_embedding,
        axis=0
    )

    faiss.normalize_L2(
        jd_embedding
    )

    # Search
    k = min(
        5,
        len(resume_names)
    )

    scores, indices = index.search(
        jd_embedding,
        k
    )

    results = []

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        similarity = float(
            scores[0][rank - 1] * 100
        )

        results.append(
            {
                "rank": rank,
                "resume": resume_names[idx],
                "similarity": round(
                    similarity,
                    2
                )
            }
        )

    return results