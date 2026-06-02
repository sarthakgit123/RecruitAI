import os
import json
from candidate_profile import profile_genrate

# Create profiles folder if it doesn't exist
os.makedirs("profiles", exist_ok=True)

# Process all resumes
for file in os.listdir("resumes"):

    if file.endswith(".pdf"):

        pdf_path = os.path.join("resumes", file)

        print(f"Processing: {file}")

        try:
            profile = profile_genrate(pdf_path)

            json_filename = os.path.splitext(file)[0] + ".json"

            json_path = os.path.join(
                "profiles",
                json_filename
            )

            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    profile,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Saved: {json_path}")

        except Exception as e:

            print(f"Failed: {file}")
            print(e)

print("\nAll resumes processed.")