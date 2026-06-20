# AI Resume–JD Matcher & Chatbot

An AI-powered hiring assistant that does two things with a single pool of resumes:

1. **JD Matching** — score every resume against a job description and rank candidates by similarity.
2. **Resume Chatbot (RAG)** — ask free-form questions across the whole candidate pool ("who knows Python", "who has 2+ years experience", "who'd be a good fit for a backend role") and get grounded, multi-candidate answers.

Built with **FastAPI**, **Gemini** (resume parsing + chat reasoning), **Sentence-Transformers** (embeddings), and **FAISS** (vector search).

---

## How it works

```
Upload .zip of resumes
        │
        ▼
Parse each PDF → structured JSON (Gemini)
        │
        ▼
Verify experience years in plain Python (don't trust LLM math blindly)
        │
        ├──► Build whole-resume FAISS index ──► JD Matcher
        │
        └──► Chunk by section (skills / project / experience / education)
                     │
                     ▼
             Build chunked FAISS index ──► Resume Chatbot (RAG)
```

After upload, the user picks one of two paths:

- **JD Match** — paste a job description, get the top candidates ranked by similarity.
- **Chatbot** — ask anything about the pool. Each question runs through:
  1. **Intent extraction** (Gemini) — pulls out hard filters (skill / min years / role) and a semantic remainder.
  2. **Metadata filtering** (plain Python) — exact filtering on the verified, structured resume data. No AI guesswork for numbers.
  3. **FAISS semantic search** — restricted to the filtered candidate pool, for the fuzzy/open-ended part of the question.
  4. **Answer generation** (Gemini) — grounded only in the retrieved context, instructed to consider every relevant candidate, not just one.

This hybrid design exists because pure vector similarity is bad at exact constraints ("3+ years of experience"), and pure keyword filtering can't handle open-ended judgment questions ("who'd be a good fit for this role"). Combining both gives accurate answers for both cases.

---

## Features

- Upload a single `.zip` of resumes — any number of PDFs.
- Automatic parsing into structured JSON via Gemini (name, skills, projects, education, experience).
- Self-verified experience calculation — total years of experience are recomputed in Python from each role's start/end dates, rather than trusting the LLM's arithmetic.
- JD Matcher — paste a job description, get ranked candidates by cosine similarity.
- Resume Chatbot — cross-resume question answering with:
  - Exact skill / experience / role filtering
  - Open-ended fit and comparison reasoning
  - Conversation history for natural follow-ups
  - Honest "no candidates match" responses instead of hallucinated answers
- Clean separation between the two FAISS indexes — one tuned for whole-resume JD matching, one chunked by section for precise chatbot retrieval.
- Resilient to transient Gemini API overload (503) with automatic retry + backoff.

---

## Tech stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| Templates | Jinja2 |
| Resume parsing | Gemini API (`gemini-2.5-flash`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (`IndexFlatIP`, cosine similarity via L2-normalized vectors) |
| PDF text extraction | PyMuPDF |
| Frontend | Vanilla HTML / CSS / JS (no framework) |

---

## Project structure

```
AI-Resume-JD-Matcher/
├── app.py                       # FastAPI app entrypoint, routes
├── chatbot_routes.py             # /chat, /chat-api, /chat-reset routes
├── services/
│   ├── __init__.py
│   ├── pdf_service.py             # Extracts all resumes in a zip → JSON profiles
│   ├── profile_service.py         # Gemini PDF → structured JSON parsing
│   ├── experience_utils.py        # Python-side verification of total experience years
│   ├── faiss_service.py           # Builds the JD-matcher's whole-resume FAISS index
│   ├── match_service.py           # Searches the JD-matcher index against a job description
│   ├── chatbot_index_service.py   # Builds the chatbot's chunked FAISS index
│   ├── chatbot_retrieval.py       # Intent extraction + hybrid filtering + semantic search
│   └── chatbot_service.py         # Final answer generation + chat history
├── templates/
│   ├── index.html                 # Upload page (Exhibit A)
│   ├── fork.html                  # Post-upload choice: JD Match or Chatbot
│   ├── jd_match.html              # Job description input (Exhibit B)
│   ├── results.html                # JD match results
│   └── chat.html                   # Chatbot interface (Exhibit C)
├── static/
│   ├── style.css                   # Shared "case file" design system
│   ├── fork.css
│   └── chat.css / chat.js
├── uploads/                        # Uploaded zips + extracted resumes (gitignored)
├── profiles/                        # Parsed resume JSON (gitignored)
├── faiss_db/                        # Both FAISS indexes + metadata (gitignored)
├── .env                              # GEMINI_API_KEY (gitignored)
└── requirements.txt
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd AI-Resume-JD-Matcher
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/).

### 4. Run the app

```bash
uvicorn app:app --reload
```

Visit **http://127.0.0.1:8000**.

---

## Usage

1. **Upload a pool** — zip up resume PDFs, upload on the home page.
2. The app parses every resume, verifies experience numbers, and builds both FAISS indexes automatically.
3. **Choose a path:**
   - **Run JD Match** → paste a job description → get ranked candidates.
   - **Open the Interview** → ask the chatbot anything about the pool.

### Example chatbot questions

```
Who knows Python?
Who has 2+ years of experience?
Who has worked as an intern?
Who would be a good fit for a backend role?
I need a candidate to build an LLM-based platform. Who should I hire and why?
```

---

## Design notes

**Why two separate FAISS indexes?**
The JD-matcher needs a holistic, whole-resume comparison against a job description — one vector per resume. The chatbot needs precision at the section level (skills vs. a specific project vs. a specific role), so it's chunked into multiple vectors per resume. Reusing one index for both jobs would hurt one use case or the other.

**Why verify experience years in Python instead of trusting Gemini's number?**
LLM arithmetic on dates isn't reliable enough to filter candidates on. `experience_utils.py` recomputes total experience from each role's `start_date`/`end_date` and only falls back to Gemini's self-reported figure when dates are missing or unparseable.

**Why hybrid retrieval instead of pure RAG or pure filtering?**
Pure vector search struggles with exact constraints like "3+ years experience." Pure keyword filtering can't reason about open-ended questions like "who'd be a good culture fit." Combining hard metadata filters with semantic search on the filtered pool gets the best of both.

---

## Known limitations

- **Free-tier Gemini quota** is limited (20–250 requests/day depending on model and tier as of writing). Heavy testing can exhaust it quickly — each chatbot question costs 2 Gemini calls (intent extraction + answer generation), and each resume upload costs 1 call per resume.
- **Chat history is in-memory and global**, not per-session — fine for solo use/demos, but concurrent users would share one conversation history.
- **Re-uploading a zip rebuilds everything from scratch** — there's no incremental indexing yet.
- Designed and tested at a scale of **50–300 resumes**; not yet optimized for much larger pools.

---

## Roadmap ideas

- [ ] Per-session chat history (so multiple users don't share one conversation)
- [ ] Incremental indexing (only re-parse new/changed resumes)
- [ ] Support multiple skills per filter query (e.g. "LangChain and LangGraph")
- [ ] Swap to `gemini-2.5-flash-lite` or combine intent extraction + answer generation into one call to reduce quota usage
- [ ] Async upload processing with a progress indicator for large pools

