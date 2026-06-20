from fastapi import (
    FastAPI,
    Form,
    Request,
    UploadFile,
    File
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from fastapi.templating import (
    Jinja2Templates
)

from fastapi.staticfiles import (
    StaticFiles
)

from typing import List
import os

from services.match_service import (
    process_resumes_and_match
)

from services.upload_service import (
    process_uploaded_zip
)

from services.pdf_service import (
    process_all_resumes
)

from services.faiss_service import (
    build_faiss_index
)

from services.Chatbot_index_service import (
    build_chatbot_index
)

from chatbot_routes import router as chatbot_router

import zipfile
import shutil

app = FastAPI(
    title="AI Resume JD Matcher",
    version="1.0.0"
)

# -----------------------------
# Chatbot routes
# -----------------------------

app.include_router(chatbot_router)

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# -----------------------------
# Static + Templates
# -----------------------------

templates = Jinja2Templates(
    directory="templates"
)

app.mount(
    "/static",
    StaticFiles(
        directory="static"
    ),
    name="static"
)

# -----------------------------
# Create folders
# -----------------------------

os.makedirs(
    "uploads",
    exist_ok=True
)

os.makedirs(
    "profiles",
    exist_ok=True
)

os.makedirs(
    "faiss_db",
    exist_ok=True
)

def cleanup():

    folders = [
        "uploads/resumes",
        "profiles",
        "faiss_db"
    ]

    for folder in folders:

        shutil.rmtree(
            folder,
            ignore_errors=True
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# -----------------------------
# Home Page
# -----------------------------

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Resume Matcher"
        }
    )

# -----------------------------
# JD Match Page
# -----------------------------

@app.get("/jd-match")
async def jd_match_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="jd_match.html",
        context={
            "title": "JD Match"
        }
    )

# -----------------------------
# Upload Resumes
# -----------------------------

@app.post("/upload-zip")
async def upload_zip(
    request: Request,
    zip_file: UploadFile = File(...)
):

    try:

        zip_path = os.path.join(
            "uploads",
            zip_file.filename
        )

        with open(zip_path, "wb") as f:
            f.write(await zip_file.read())

        # Clear previous resumes

        shutil.rmtree(
            "uploads/resumes",
            ignore_errors=True
        )

        os.makedirs(
            "uploads/resumes",
            exist_ok=True
        )

        # Extract ZIP

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                "uploads/resumes"
            )

        # Parse every resume PDF into structured JSON (profiles/*.json).
        # This also runs the experience-years verification step inside
        # profile_service.py, so total_experience_years is trustworthy
        # by the time either the JD-matcher or the chatbot reads it.
        process_all_resumes()

        # Build the JD-matcher's whole-resume FAISS index
        # (faiss_db/resume_index.faiss). Previously this ran inside
        # match_service.py on every JD submission - moved here so it
        # only runs once, right after upload, instead of on every match.
        build_faiss_index()

        # Build the chatbot's separate chunked FAISS index
        # (faiss_db/chatbot_index.faiss) from the same JSON profiles.
        # This is a different index from the one above - the JD-matcher
        # needs one vector per whole resume, the chatbot needs section-
        # level chunks for precise retrieval. Building both here means
        # the chatbot is immediately usable even if the user never runs
        # a JD match.
        build_chatbot_index()

        profile_count = len([
            f for f in os.listdir("profiles") if f.endswith(".json")
        ]) if os.path.isdir("profiles") else 0

        return templates.TemplateResponse(
            request=request,
            name="fork.html",
            context={
                "count": profile_count
            }
        )

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
# -----------------------------
# Match UI
# -----------------------------

@app.post("/match-ui")
async def match_ui(
    request: Request,
    jd: str = Form(...)
):

    try:

        results = process_resumes_and_match(
            jd
        )
         
       

        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "results": results
            }
        )

    except Exception as e:

        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "results": [],
                "error": str(e)
            }
        )

# -----------------------------
# API Endpoint
# -----------------------------

@app.post("/match")
async def match_api(
    jd: str = Form(...)
):

    try:

        results = process_resumes_and_match(
            jd
        )



        return {
            "status": "success",
            "top_candidates": results
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }