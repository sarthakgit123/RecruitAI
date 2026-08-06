"""
match.py

FastAPI routes for JD Match UI and JSON API.
"""

from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.services.matching.match_service import process_resumes_and_match
from app.schemas.match import MatchAPIResponse

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": settings.PROJECT_NAME}
    )


@router.get("/jd-match")
async def jd_match_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="jd_match.html",
        context={"title": f"{settings.PROJECT_NAME} — JD Match"}
    )


@router.post("/match-ui")
async def match_ui(request: Request, jd: str = Form(...)):
    try:
        results = process_resumes_and_match(jd)
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "results": results,
                "jd": jd,
                "title": f"{settings.PROJECT_NAME} — Match Results"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "results": [],
                "jd": jd,
                "error": str(e),
                "title": f"{settings.PROJECT_NAME} — Match Error"
            }
        )


@router.post("/match", response_model=MatchAPIResponse)
async def match_api(jd: str = Form(...)):
    try:
        results = process_resumes_and_match(jd)
        return {
            "status": "success",
            "top_candidates": results
        }
    except Exception as e:
        return {
            "status": "error",
            "top_candidates": [],
            "message": str(e)
        }
