"""
upload.py

FastAPI route for ZIP resume pool upload.
Passes a database session to pdf_service for PostgreSQL candidate persistence.
"""

import os
from fastapi import APIRouter, Request, UploadFile, File, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.upload.upload_service import process_uploaded_zip
from app.services.upload.pdf_service import process_all_resumes
from app.services.vectorstore.faiss_service import build_faiss_index
from app.services.chatbot.chatbot_index_service import build_chatbot_index
from app.database.database import get_db

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


@router.post("/upload-zip")
async def upload_zip(
    request: Request,
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        await process_uploaded_zip(zip_file)
        process_all_resumes(db=db)
        build_faiss_index()
        build_chatbot_index()

        profile_count = len([
            f for f in os.listdir(settings.PROFILES_DIR) if f.endswith(".json")
        ]) if os.path.isdir(settings.PROFILES_DIR) else 0

        return templates.TemplateResponse(
            request=request,
            name="Fork.html",
            context={
                "count": profile_count,
                "title": "RecruitAI — Intake Logged"
            }
        )
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
