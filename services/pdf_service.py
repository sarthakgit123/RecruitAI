import os
import json
from services.profile_service import profile_genrate

def process_all_resumes():
    os.makedirs("profiles", exist_ok=True)

    # Process all resumes
    for file in os.listdir("uploads"):

        if file.endswith(".pdf"):

            pdf_path = os.path.join("uploads", file)

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