"""
skill_normalizer.py

Normalizes skill names to canonical forms and performs skill matching
between JD requirements and candidate profiles.
"""

import re
from typing import List, Dict, Any, Set

SYNONYM_MAP = {
    # Programming Languages
    "python": "Python", "py": "Python", "python3": "Python",
    "javascript": "JavaScript", "js": "JavaScript", "ecmascript": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "c++": "C++", "cpp": "C++", "c#": "C#", "csharp": "C#", "c": "C",
    "java": "Java", "golang": "Go", "go": "Go", "rust": "Rust",
    "php": "PHP", "ruby": "Ruby", "swift": "Swift", "kotlin": "Kotlin",
    "html": "HTML/CSS", "html5": "HTML/CSS", "css": "HTML/CSS", "css3": "HTML/CSS",

    # Web & Backend Frameworks
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django",
    "express": "Express.js", "expressjs": "Express.js", "express.js": "Express.js",
    "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "spring": "Spring Boot", "springboot": "Spring Boot", "spring boot": "Spring Boot",
    "laravel": "Laravel",

    # Frontend Frameworks
    "react": "React", "reactjs": "React", "react.js": "React",
    "next": "Next.js", "nextjs": "Next.js", "next.js": "Next.js",
    "vue": "Vue.js", "vuejs": "Vue.js", "vue.js": "Vue.js",
    "angular": "Angular", "angularjs": "Angular",

    # Databases & Storage
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "postgres db": "PostgreSQL",
    "mysql": "MySQL", "sqlite": "SQLite", "mongo": "MongoDB", "mongodb": "MongoDB",
    "redis": "Redis", "dynamodb": "DynamoDB", "oracle": "Oracle",
    "sql server": "MS SQL Server", "mssql": "MS SQL Server", "sql": "SQL",

    # APIs & Protocols
    "rest": "REST API", "rest api": "REST API", "restful": "REST API", "restful api": "REST API",
    "graphql": "GraphQL", "grpc": "gRPC", "json": "JSON",

    # Cloud & DevOps
    "aws": "AWS", "amazon web services": "AWS",
    "gcp": "Google Cloud (GCP)", "google cloud": "Google Cloud (GCP)",
    "azure": "Microsoft Azure", "microsoft azure": "Microsoft Azure",
    "docker": "Docker", "k8s": "Kubernetes", "kubernetes": "Kubernetes",
    "ci/cd": "CI/CD", "cicd": "CI/CD", "terraform": "Terraform",
    "git": "Git", "github": "GitHub",

    # AI, ML & Data Science
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "ai": "Artificial Intelligence", "artificial intelligence": "Artificial Intelligence",
    "nlp": "Natural Language Processing", "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn",
    "pandas": "Pandas", "numpy": "NumPy", "keras": "Keras",
    "opencv": "OpenCV", "spacy": "spaCy", "nltk": "NLTK",
    "rag": "RAG (Retrieval-Augmented Generation)", "llm": "LLMs", "llms": "LLMs",
    "faiss": "FAISS",

    # Soft Skills & Engineering Concepts
    "microservices": "Microservices", "system design": "System Design",
    "oop": "Object-Oriented Programming", "agile": "Agile/Scrum", "scrum": "Agile/Scrum"
}


def _clean_str(s: str) -> str:
    if not s:
        return ""
    cleaned = s.lower().strip()
    cleaned = re.sub(r'[^\w\s\+\#\/\.\-]', '', cleaned)
    return cleaned


def normalize_skill(skill_raw: str) -> str:
    if not skill_raw or not isinstance(skill_raw, str):
        return ""
    
    key = _clean_str(skill_raw)
    if key in SYNONYM_MAP:
        return SYNONYM_MAP[key]
    
    return skill_raw.strip().title()


def normalize_skill_list(skills_list: List[str]) -> List[str]:
    if not skills_list:
        return []
    
    normalized = []
    seen: Set[str] = set()
    
    for s in skills_list:
        canonical = normalize_skill(s)
        if canonical and canonical.lower() not in seen:
            seen.add(canonical.lower())
            normalized.append(canonical)
            
    return normalized


def match_skills(candidate_skills: List[str], required_skills: List[str], preferred_skills: List[str] = None) -> Dict[str, Any]:
    cand_norm = set(s.lower() for s in normalize_skill_list(candidate_skills))
    req_norm = normalize_skill_list(required_skills or [])
    pref_norm = normalize_skill_list(preferred_skills or [])

    matched_req = [s for s in req_norm if s.lower() in cand_norm]
    missing_req = [s for s in req_norm if s.lower() not in cand_norm]

    matched_pref = [s for s in pref_norm if s.lower() in cand_norm]
    missing_pref = [s for s in pref_norm if s.lower() not in cand_norm]

    req_ratio = len(matched_req) / len(req_norm) if req_norm else 1.0
    pref_ratio = len(matched_pref) / len(pref_norm) if pref_norm else 1.0

    if req_norm and pref_norm:
        weighted_score = (req_ratio * 70.0) + (pref_ratio * 30.0)
    elif req_norm:
        weighted_score = req_ratio * 100.0
    elif pref_norm:
        weighted_score = pref_ratio * 100.0
    else:
        weighted_score = 75.0

    return {
        "matched_required": matched_req,
        "missing_required": missing_req,
        "matched_preferred": matched_pref,
        "missing_preferred": missing_pref,
        "all_matched": list(set(matched_req + matched_pref)),
        "all_missing": list(set(missing_req + missing_pref)),
        "required_ratio": req_ratio,
        "preferred_ratio": pref_ratio,
        "weighted_skill_score": round(weighted_score, 2)
    }
