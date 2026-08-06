"""
API Routes Package.
Includes all router submodules.
"""

from fastapi import APIRouter
from app.api.routes.match import router as match_router
from app.api.routes.upload import router as upload_router
from app.api.routes.chat import router as chat_router

api_router = APIRouter()

api_router.include_router(match_router)
api_router.include_router(upload_router)
api_router.include_router(chat_router)
