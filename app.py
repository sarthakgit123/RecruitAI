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

import zipfile
import shutil

app = FastAPI(
    title="AI Resume JD Matcher",
    version="1.0.0"
)

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

        return templates.TemplateResponse(
            request=request,
            name="upload_success.html",
            context={
                "result":
                "ZIP uploaded successfully"
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
         
        cleanup()

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

        cleanup()

        return {
            "status": "success",
            "top_candidates": results
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }