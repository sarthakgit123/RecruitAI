"""
main.py

Main FastAPI application entry point.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import api_router

# Ensure runtime directories exist outside app/
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.RESUMES_DIR, exist_ok=True)
os.makedirs(settings.PROFILES_DIR, exist_ok=True)
os.makedirs(settings.FAISS_DB_DIR, exist_ok=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=str(settings.STATIC_DIR)),
    name="static"
)

# Include API routes
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
