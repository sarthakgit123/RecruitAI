import os

from services.pdf_service import (
    process_all_resumes
)

from services.faiss_service import (
    build_faiss_index
)


UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


async def process_uploaded_resumes(
    resumes
):

    uploaded_files = []

    for resume in resumes:

        file_path = os.path.join(
            UPLOAD_DIR,
            resume.filename
        )

        with open(
            file_path,
            "wb"
        ) as f:

            content = await resume.read()

            f.write(content)

        uploaded_files.append(
            resume.filename
        )

    # Generate profile JSONs
    process_all_resumes()

    # Build FAISS index
    build_faiss_index()

    return {
        "status": "success",
        "uploaded_files": uploaded_files,
        "total_files": len(
            uploaded_files
        )
    }