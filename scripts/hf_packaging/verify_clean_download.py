import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed. Please install it using: pip install huggingface_hub")
    sys.exit(1)


def verify_remote_repo(repo_id: str):
    print(f"\n[CLEAN-VERIFY] Downloading latest release from Hugging Face: {repo_id}...")
    temp_dir = tempfile.mkdtemp(prefix="hf_verify_")
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[CLEAN-VERIFY] Download completed to clean directory: {download_path}")

        verify_script = Path(download_path) / "inference" / "verify_bundle.py"
        if not verify_script.exists():
            raise FileNotFoundError(f"Verification script not found at {verify_script}")

        print("[CLEAN-VERIFY] Executing verify_bundle.py on freshly downloaded repository...")
        res = subprocess.run(
            [sys.executable, str(verify_script), "--bundle_dir", str(download_path)],
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise RuntimeError(f"Clean download verification failed with exit code {res.returncode}")

        print("[SUCCESS] Post-upload clean download verification PASSED! Repository is production-ready.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download and verify published Hugging Face repository in a clean environment.")
    parser.add_argument("--repo_id", type=str, default="neitpm/hybridtrinet-oil-forecasting", help="Hugging Face repository ID")
    args = parser.parse_args()
    verify_remote_repo(args.repo_id)


if __name__ == "__main__":
    main()
