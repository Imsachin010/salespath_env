from huggingface_hub import HfApi
import os

REPO_ID = "Imsachin010/salespath-env"
FOLDER_PATH = "."

IGNORE_PATTERNS = [
    "*.pyc",
    "**/__pycache__/**",
    ".git/**",
    ".spa/**",
    ".SPA/**",
    "*.egg-info/**",
    "push_to_hub.py",
    "salespath_env/server/Dockerfile",  # root Dockerfile is used instead
    # Training scripts ARE included for HF Spaces GPU training
    # "training/**",
]

def main():
    api = HfApi()

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )

    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy SalesPath Environment",
    )

    print(
        f"Live Space URL:\n"
        f"https://{REPO_ID.replace('/', '-')}.hf.space"
    )

if __name__ == "__main__":
    main()