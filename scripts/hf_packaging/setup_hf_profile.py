import os
import sys
from huggingface_hub import HfApi

def setup_profile():
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        print("[ERROR] HF_TOKEN is not set in the environment.")
        print("Please set it first via: $env:HF_TOKEN=\"hf_...\"")
        sys.exit(1)

    api = HfApi(token=token.strip())
    repo_id = "neitpm/neitpm"

    profile_content = """---
title: PHAM MINH TIEN
emoji: 🧠
colorFrom: blue
colorTo: indigo
pinned: false
---

# PHAM MINH TIEN

### AI Engineer

I build deep learning systems for **visual perception and time-series modeling**, with an emphasis on ONNX edge inference and reproducible evaluation.

## Links

- **GitHub:** [github.com/neitpm](https://github.com/neitpm)
- **Email:** [neit.pm.work@gmail.com](mailto:neit.pm.work@gmail.com)
"""

    print(f"[INFO] Creating/Verifying profile repo: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"[INFO] Uploading profile README.md to {repo_id}...")
    api.upload_file(
        path_or_fileobj=profile_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Create professional profile README",
    )

    print("[INFO] Verifying repository status...")
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"[SUCCESS] Profile README verified! Repo ID: {info.id}")
    print(f"[SUCCESS] Public URL: https://huggingface.co/{repo_id}")
    print(f"[SUCCESS] Profile Page: https://huggingface.co/neitpm")

if __name__ == "__main__":
    setup_profile()
