import os
import zipfile
import shutil

UPLOAD_FOLDER = "uploads"
RESUME_FOLDER = "uploads/resumes"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESUME_FOLDER, exist_ok=True)


async def process_uploaded_zip(zip_file):

    # Clear previous resumes
    shutil.rmtree(
        RESUME_FOLDER,
        ignore_errors=True
    )

    os.makedirs(
        RESUME_FOLDER,
        exist_ok=True
    )

    # Save uploaded zip
    zip_path = os.path.join(
        UPLOAD_FOLDER,
        zip_file.filename
    )

    with open(zip_path, "wb") as f:
        content = await zip_file.read()
        f.write(content)

    # Extract zip
    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            RESUME_FOLDER
        )

    # Delete uploaded zip
    os.remove(zip_path)

    # Collect PDF names
    extracted_files = []

    for file in os.listdir(
        RESUME_FOLDER
    ):

        if file.lower().endswith(
            ".pdf"
        ):
            extracted_files.append(
                file
            )

    return {
        "status": "success",
        "uploaded_files": extracted_files,
        "total_files": len(
            extracted_files
        )
    }