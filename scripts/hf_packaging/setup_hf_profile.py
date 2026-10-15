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
title: Pham Minh Tien
emoji: 🧠
colorFrom: blue
colorTo: indigo
pinned: false
---

# Pham Minh Tien

### AI Engineer · Computer Vision · Time-Series Forecasting

I build deep learning systems for computer vision, document AI, biometric ROI detection,
and multi-horizon time-series forecasting, with an emphasis on reproducible evaluation,
model packaging, and inference pipelines.

## Featured Model Releases

- **[HybridTriNet Oil Forecasting](https://huggingface.co/neitpm/hybridtrinet-oil-forecasting)**  
  Multi-horizon forecasting bundle with GUMNet and HybridTriNet architectures,
  per-horizon checkpoints, scalers, inference code, metrics, and SHA-256 verification.

- **[SSDLite Palm-Vein ROI](https://huggingface.co/neitpm/ssdlite-palm-vein-roi)**  
  CPU-based ONNX inference prototype for SSDLite-MobileNetV3 with PyTorch–ONNX
  numerical parity verification and reproducible latency profiling.

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
