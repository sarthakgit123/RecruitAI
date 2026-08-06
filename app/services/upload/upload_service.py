"""
upload_service.py

Handles saving uploaded zip files and extracting PDF resumes.
"""

import os
import zipfile
import shutil
from typing import Dict, Any
from fastapi import UploadFile
from app.core.config import settings


async def process_uploaded_zip(zip_file: UploadFile) -> Dict[str, Any]:
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    os.makedirs(settings.RESUMES_DIR, exist_ok=True)

    # Clear previous resumes
    shutil.rmtree(settings.RESUMES_DIR, ignore_errors=True)
    os.makedirs(settings.RESUMES_DIR, exist_ok=True)

    # Save uploaded zip
    zip_path = settings.UPLOADS_DIR / zip_file.filename

    with open(zip_path, "wb") as f:
        content = await zip_file.read()
        f.write(content)

    # Extract zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(settings.RESUMES_DIR)

    # Delete uploaded zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Collect PDF names
    extracted_files = [
        file for file in os.listdir(settings.RESUMES_DIR)
        if file.lower().endswith(".pdf")
    ]

    return {
        "status": "success",
        "uploaded_files": extracted_files,
        "total_files": len(extracted_files)
    }
