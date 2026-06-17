import os
import json

from profile_service import (
    profile_generate
)

# Anchor all paths to this file's location (services/), then go up one
# level to the project root. This makes the script work no matter where
# it's run from (services/ directly, project root via -m, etc).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads", "resumes", "resumes")
PROFILES_DIR = os.path.join(PROJECT_ROOT, "profiles")


def process_all_resumes():

    os.makedirs(
        PROFILES_DIR,
        exist_ok=True
    )

    if not os.path.isdir(UPLOADS_DIR):
        print(f"Upload folder not found: {UPLOADS_DIR}")
        return

    for root, dirs, files in os.walk(
        UPLOADS_DIR
    ):

        for file in files:

            if file.endswith(".pdf"):

                pdf_path = os.path.join(
                    root,
                    file
                )

                print(
                    f"Processing: {pdf_path}"
                )

                try:

                    profile = profile_generate(
                        pdf_path
                    )

                    json_filename = (
                        os.path.splitext(file)[0]
                        + ".json"
                    )

                    json_path = os.path.join(
                        PROFILES_DIR,
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

                    print(
                        f"Saved: {json_path}"
                    )

                except Exception as e:

                    print(
                        f"Failed: {file}"
                    )

                    print(e)

    print(
        "\nAll resumes processed."
    )


if __name__ == "__main__":
    process_all_resumes()