# AI-Powered Resume Screening & ATS Ranking System

An intelligent resume screening platform that leverages semantic embeddings, vector search, and Large Language Models (LLMs) to evaluate candidate-job fit beyond traditional keyword matching.

The system analyzes resumes against job descriptions, computes ATS-style match scores, identifies skill gaps, ranks candidates, and generates explainable hiring insights using AI.

---

## Features

### Semantic Resume Matching

* Uses transformer-based embeddings to capture contextual meaning.
* Matches resumes with job descriptions beyond exact keyword overlap.
* Handles skill synonyms and related technologies effectively.

### ATS Score Generation

* Calculates resume-job compatibility scores.
* Evaluates skill alignment and keyword coverage.
* Provides recruiter-friendly scoring metrics.

### Candidate Ranking

* Indexes resumes using FAISS vector search.
* Retrieves Top-K most relevant candidates.
* Supports bulk resume evaluation and comparison.

### Skill Gap Analysis

* Detects missing skills required by the job description.
* Highlights strengths and weaknesses in candidate profiles.
* Provides actionable improvement recommendations.

### AI-Powered Explanations

* Uses Gemini API to generate:

  * Candidate-fit summaries
  * Skill-gap insights
  * Match reasoning
  * Resume improvement suggestions

### Fast Resume Processing

* PDF resume ingestion and parsing.
* Embedding generation and vector indexing.
* Real-time candidate ranking and scoring.

---

## System Architecture

```text
Resume PDFs
      │
      ▼
 Resume Parser
      │
      ▼
Embedding Generation
(Sentence Transformers)
      │
      ▼
 FAISS Vector Store
      │
      ▼
Similarity Search
      │
      ▼
ATS Scoring Engine
      │
      ▼
 Gemini Analysis
      │
      ▼
Final Candidate Ranking
```

---

## Tech Stack

### Backend

* Python
* FastAPI

### AI / Machine Learning

* Sentence Transformers
* NLP
* Semantic Embeddings
* Vector Search

### Retrieval

* FAISS

### LLM

* Google Gemini API

### Data Processing

* PDF Parsing
* Text Extraction

---

## Example Workflow

1. Upload multiple resumes.
2. Enter a job description.
3. Generate semantic embeddings.
4. Search candidates using FAISS similarity search.
5. Calculate ATS match scores.
6. Identify missing skills.
7. Generate AI-powered explanations.
8. Display ranked candidate list.

---

## Sample Output

```text
Candidate: John Doe

ATS Match Score: 89%

Matched Skills:
✓ Python
✓ FastAPI
✓ SQL
✓ REST APIs

Missing Skills:
✗ Docker
✗ AWS

AI Summary:
Strong backend candidate with relevant API development
experience and database knowledge. Consider improving
cloud deployment skills.
```

---

## Future Enhancements

* Multi-JD candidate matching
* Resume recommendation engine
* Recruiter dashboard
* Interview question generation
* Candidate clustering
* Hybrid Search (BM25 + Semantic Search)
* Analytics dashboard
* Resume optimization assistant

---

## Key Highlights

* Semantic Matching using Transformer Embeddings
* Explainable AI-based Candidate Evaluation
* ATS Score Generation
* Skill Gap Detection
* Top-K Candidate Retrieval using FAISS
* LLM-Powered Hiring Insights
* Scalable FastAPI Backend

---

## Author

**Sarthak Kumar Seth**

GitHub: https://github.com/sarthakgit123
