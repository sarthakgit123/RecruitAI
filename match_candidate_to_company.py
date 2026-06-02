from sentence_transformers import SentenceTransformer
from candidate_profile import profile_genrate

profile = profile_genrate("Sarthak_resume.pdf")

job_description = """
Looking for a Python Backend Intern.

Requirements:
- Python
- FastAPI
- REST APIs
- SQL
- Machine Learning experience preferred
"""

profile_text = f"""
Skills:
{profile['skills']}

Projects:
{profile['projects']}

Education:
{profile['education']}

Experience:
{profile['experience']}
"""

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

candidate_embedding = model.encode(profile_text)
job_embedding = model.encode(job_description)

score = model.similarity(
    [candidate_embedding],
    [job_embedding]
)

print(f"Match Score: {float(score[0][0])*100:.2f}%")