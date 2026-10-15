import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("[ERROR] huggingface_hub is not installed. Please install it using: pip install huggingface_hub")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BUNDLE_DIR = ROOT_DIR / "hf_releases" / "hybridtrinet-oil-forecasting"


def upload_bundle(repo_id: str, private: bool = False):
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        print("[ERROR] HF_TOKEN environment variable not found or empty.")
        print("Please set your Hugging Face token in the environment before uploading:")
        print("  Windows PowerShell: $env:HF_TOKEN=\"your_hf_token\"")
        print("  Linux/macOS:        export HF_TOKEN=\"your_hf_token\"")
        sys.exit(1)

    if not BUNDLE_DIR.exists():
        print(f"[ERROR] Release bundle directory does not exist: {BUNDLE_DIR}")
        print("Run scripts/hf_packaging/package_bundle.py first.")
        sys.exit(1)

    api = HfApi(token=token.strip())

    print(f"[INFO] Creating/Verifying Hugging Face model repository: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[INFO] Repository ready: https://huggingface.co/{repo_id}")

    print(f"[INFO] Uploading staged files from {BUNDLE_DIR} to {repo_id}...")
    api.upload_folder(
        folder_path=str(BUNDLE_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release HybridTriNet multi-horizon oil forecasting bundle with verified scalers and inference pipeline",
    )
    print("[SUCCESS] Model bundle uploaded successfully!")
    print(f"[INFO] View model card at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload staged HybridTriNet bundle to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, default="neitpm/hybridtrinet-oil-forecasting", help="Target Hugging Face repository ID")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    upload_bundle(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
