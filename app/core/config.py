"""
config.py

Central application configuration module.
Anchors all relative paths to the project root directory.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Project root directory (3 levels up from app/core/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")


class Settings(BaseModel):
    PROJECT_NAME: str = "RecruitAI"
    VERSION: str = "1.0.0"

    # Directory Paths
    BASE_DIR: Path = BASE_DIR
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    RESUMES_DIR: Path = BASE_DIR / "uploads" / "resumes"
    PROFILES_DIR: Path = BASE_DIR / "profiles"
    FAISS_DB_DIR: Path = BASE_DIR / "faiss_db"
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"

    # FAISS File Paths
    RESUME_INDEX_PATH: Path = BASE_DIR / "faiss_db" / "resume_index.faiss"
    RESUME_NAMES_PATH: Path = BASE_DIR / "faiss_db" / "resume_names.pkl"
    CHATBOT_INDEX_PATH: Path = BASE_DIR / "faiss_db" / "chatbot_index.faiss"
    CHATBOT_METADATA_PATH: Path = BASE_DIR / "faiss_db" / "chatbot_metadata.pkl"

    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODELS: list[str] = [
        "nvidia/nemotron-nano-9b-v2:free",
        "openrouter/free",
        "meta-llama/llama-3.3-70b-instruct:free"
    ]


settings = Settings()
