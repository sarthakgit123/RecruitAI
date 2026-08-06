"""
run.py

Top-level server launcher for RecruitAI FastAPI application.
Run using: python run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
