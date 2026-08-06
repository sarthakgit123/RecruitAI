# RecruitAI

An AI-powered recruitment assistant that ranks resumes against job descriptions and lets you chat with your entire candidate pool.

Upload a zip of resumes → get structured profiles, ranked matches, and a chatbot that answers questions like *"Who knows Python and has 2+ years experience?"*

Built with **FastAPI**, **OpenRouter LLMs**, **Sentence-Transformers**, and **FAISS**.

---

## Features

- **Resume Upload** — Upload a `.zip` of PDF resumes. Each resume is parsed into structured JSON (name, skills, projects, education, experience) using LLM.
- **Hybrid JD Matching** — Paste a job description and get candidates ranked by a weighted composite score:
  - Semantic similarity (25%)
  - Skill matching with synonym normalization (35%)
  - Experience verification (20%)
  - Project relevance (10%)
  - Keyword coverage (10%)
- **Explainable Rankings** — Each candidate shows matched/missing skills, strengths, weaknesses, and a plain-English ranking explanation.
- **Resume Chatbot (RAG)** — Ask free-form questions across the whole candidate pool with intent extraction, hard filtering, and grounded answers.
- **Skill Normalization** — 80+ synonym mappings (e.g., `postgres` → `PostgreSQL`, `js` → `JavaScript`) for accurate matching.
- **Experience Verification** — Total experience years are recomputed from start/end dates in Python, not blindly trusted from LLM output.
- **Centralized Prompts** — All LLM prompts live in editable `.txt` files under `app/prompts/`. Change prompts without touching code.

---

## How It Works

```
Upload .zip of resumes
        │
        ▼
Parse each PDF → structured JSON (LLM)
        │
        ▼
Verify experience years (Python date math)
        │
        ├──► Build whole-resume FAISS index ──► JD Matcher (hybrid scoring)
        │
        └──► Chunk by section (skills / project / experience / education)
                     │
                     ▼
             Build chunked FAISS index ──► Resume Chatbot (RAG)
```

### JD Matching Pipeline

1. Parse JD into structured JSON (role, required/preferred skills, experience, keywords)
2. Retrieve Top-K candidates via FAISS vector similarity
3. Rerank using hybrid scoring across 5 factors
4. Generate explainability report for each candidate

### Chatbot Pipeline

1. Extract intent from question (skill filter, min years, role, semantic query)
2. Apply hard metadata filters on structured profiles
3. Semantic search on filtered candidates via FAISS
4. Generate grounded answer from retrieved context

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| Templates | Jinja2 |
| LLM | OpenRouter (Nemotron, LLaMA 3.3 70B) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (IndexFlatIP, cosine similarity) |
| PDF Extraction | PyMuPDF |
| Frontend | Vanilla HTML / CSS / JS |

---

## Project Structure

```
RecruitAI/
├── app/
│   ├── main.py                              # FastAPI entry point, CORS, static mounts
│   ├── api/routes/
│   │   ├── upload.py                        # POST /upload-zip
│   │   ├── match.py                         # GET /, GET /jd-match, POST /match, POST /match-ui
│   │   └── chat.py                          # GET /chat, POST /chat-api, POST /chat-reset
│   ├── core/
│   │   └── config.py                        # Central settings, .env loading, path anchors
│   ├── prompts/
│   │   ├── resume_profile.txt               # Resume → JSON extraction prompt
│   │   ├── jd_parse.txt                     # JD → structured JSON prompt
│   │   ├── chatbot_intent.txt               # Intent extraction prompt
│   │   └── chatbot_answer.txt               # RAG answer generation prompt
│   ├── services/
│   │   ├── upload/                          # Zip extraction, PDF processing
│   │   ├── parser/                          # Profile generation, JD parsing, experience utils
│   │   ├── matching/                        # Hybrid scorer, skill normalizer, explainability
│   │   ├── chatbot/                         # RAG retrieval, answer generation, index builder
│   │   └── vectorstore/                     # FAISS index builder
│   └── schemas/                             # Pydantic response models
│
├── templates/                               # Jinja2 HTML templates
├── static/                                  # CSS and JS
├── uploads/                                 # Uploaded zips + extracted resumes (gitignored)
├── profiles/                                # Parsed resume JSONs (gitignored)
├── faiss_db/                                # FAISS indexes + metadata (gitignored)
├── run.py                                   # Server launcher
├── requirements.txt
└── .env                                     # API keys (gitignored)
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/sarthakgit123/RecruitAI.git
cd RecruitAI
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a key from [OpenRouter](https://openrouter.ai/).

### 4. Run the app

```bash
python run.py
```

Visit **http://127.0.0.1:8000**.

---

## Usage

1. **Upload** — Zip your resume PDFs and upload on the home page.
2. The app parses every resume, verifies experience, and builds FAISS indexes automatically.
3. **Choose a path:**
   - **JD Match** → Paste a job description → Get ranked candidates with scores and explanations.
   - **Chatbot** → Ask anything about the candidate pool.

### Example Chatbot Questions

```
Who knows Python?
Who has 2+ years of experience?
Who has worked as an intern?
Who would be a good fit for a backend role?
Compare candidates for a full-stack position.
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home / Upload page |
| `POST` | `/upload-zip` | Upload resume zip file |
| `GET` | `/jd-match` | JD Match input page |
| `POST` | `/match` | Match API (returns JSON) |
| `POST` | `/match-ui` | Match UI (returns HTML results) |
| `GET` | `/chat` | Chatbot page |
| `POST` | `/chat-api` | Chat API (returns JSON) |
| `POST` | `/chat-reset` | Reset chat history |

---

## Design Decisions

**Why two FAISS indexes?**
The JD matcher needs whole-resume vectors for holistic comparison. The chatbot needs section-level chunks (skills, projects, experience) for precise retrieval. One index can't serve both well.

**Why verify experience in Python?**
LLM arithmetic on dates is unreliable. `experience_utils.py` recomputes total years from each role's start/end dates and only falls back to the LLM's figure when dates are missing.

**Why hybrid retrieval?**
Pure vector search can't handle exact constraints like "3+ years experience." Pure filtering can't reason about "who'd be a good fit." Combining both gives accurate answers for both cases.

**Why centralized prompts?**
Keeping prompts in `.txt` files means you can iterate on prompt engineering without modifying any Python code. Just edit the template and restart.

---

## Known Limitations

- **Free-tier API quota** is limited. Each chatbot question costs 2 LLM calls (intent + answer), and each resume upload costs 1 call per resume.
- **Chat history is in-memory and global**, not per-session. Fine for solo use, not for concurrent users.
- **Re-uploading rebuilds everything** — no incremental indexing yet.
- Tested at a scale of **50–300 resumes**.

---

## License

This project is for educational and demonstration purposes.
