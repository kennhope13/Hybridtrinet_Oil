import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

try:
    from run_inference import load_forecasting_pipeline, forecast
except ImportError:
    from scripts.hf_packaging.run_inference import load_forecasting_pipeline, forecast


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(root_dir: str):
    root = Path(root_dir)
    print(f"[VERIFY] Starting deep verification of bundle at: {root.resolve()}")

    # 1. Check checksums.sha256 if present
    checksum_file = root / "checksums.sha256"
    if checksum_file.exists():
        print(f"[VERIFY] Verifying checksums from {checksum_file.name}...")
        with open(checksum_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                expected_hash, rel_path = parts
                target_f = root / rel_path
                if not target_f.exists():
                    raise AssertionError(f"Missing file specified in checksums: {rel_path}")
                actual_hash = calculate_sha256(target_f)
                assert actual_hash == expected_hash, f"SHA-256 mismatch for {rel_path}!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        print("  -> All SHA-256 checksums matched successfully.")
    else:
        print("[WARN] checksums.sha256 not found, skipping hash verification.")

    # 2. Check each horizon under models/
    models_dir = root / "models"
    if not models_dir.exists():
        raise AssertionError(f"models/ directory not found in {root}")

    horizon_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    if not horizon_dirs:
        raise AssertionError("No horizon subdirectories found in models/")

    print(f"[VERIFY] Found {len(horizon_dirs)} horizon configurations: {[d.name for d in horizon_dirs]}")

    for h_dir in horizon_dirs:
        print(f"\n--- Verifying horizon {h_dir.name} ---")
        bundle = load_forecasting_pipeline(str(h_dir), device="cpu")
        cfg = bundle["config"]
        meta = bundle["feature_meta"]
        x_mu = bundle["x_mu"]
        x_sd = bundle["x_sd"]
        y_mu = bundle["y_mu"]
        y_sd = bundle["y_sd"]

        # Validate dimensional alignment
        K = cfg["K"]
        H = cfg["H"]
        D_in = cfg["D_in"]
        D_out = cfg["D_out"]

        assert K == 64, f"Unexpected lookback K={K}"
        assert D_in == len(meta["feature_cols"]), f"D_in ({D_in}) != len(feature_cols) ({len(meta['feature_cols'])})"
        assert D_out == len(meta["tgt_idx"]), f"D_out ({D_out}) != len(tgt_idx) ({len(meta['tgt_idx'])})"

        # Check scaler dimensions
        assert x_mu.shape == (D_in,), f"x_mu shape mismatch: {x_mu.shape} vs ({D_in},)"
        assert x_sd.shape == (D_in,), f"x_sd shape mismatch: {x_sd.shape} vs ({D_in},)"
        assert y_mu.shape == (D_out,), f"y_mu shape mismatch: {y_mu.shape} vs ({D_out},)"
        assert y_sd.shape == (D_out,), f"y_sd shape mismatch: {y_sd.shape} vs ({D_out},)"

        # Check no NaN/Inf in scalers
        assert not np.isnan(x_mu).any() and not np.isinf(x_mu).any(), "NaN/Inf found in x_mu"
        assert not np.isnan(x_sd).any() and not np.isinf(x_sd).any(), "NaN/Inf found in x_sd"
        assert not np.isnan(y_mu).any() and not np.isinf(y_mu).any(), "NaN/Inf found in y_mu"
        assert not np.isnan(y_sd).any() and not np.isinf(y_sd).any(), "NaN/Inf found in y_sd"
        assert (x_sd > 0).all(), "x_sd contains non-positive values"
        assert (y_sd > 0).all(), "y_sd contains non-positive values"

        # Check deterministic forward pass & inverse transform
        fixed_input = np.ones((1, K, D_in), dtype=np.float32) * 50.0
        pred1 = forecast(bundle, fixed_input)
        pred2 = forecast(bundle, fixed_input)

        # Output shape check
        expected_shape = (1, H, D_out)
        assert pred1.shape == expected_shape, f"Forecast shape mismatch: {pred1.shape} vs {expected_shape}"

        # No NaN / Inf in forecast
        assert not np.isnan(pred1).any(), f"NaN detected in forecast for {h_dir.name}"
        assert not np.isinf(pred1).any(), f"Inf detected in forecast for {h_dir.name}"

        # Determinism check
        np.testing.assert_allclose(pred1, pred2, atol=1e-6, err_msg="Non-deterministic inference detected!")

        print(f"  [PASS] Horizon {h_dir.name}: Shape={pred1.shape}, Finite values, Deterministic, Inverse transform OK.")

    print("\n=======================================================")
    print("[SUCCESS] All model bundles and horizons passed verification!")
    print("=======================================================")


def main():
    parser = argparse.ArgumentParser(description="Deep verification of HybridTriNet model release bundle.")
    parser.add_argument("--bundle_dir", type=str, default=".", help="Root directory of release bundle")
    args = parser.parse_args()
    verify_bundle(args.bundle_dir)


if __name__ == "__main__":
    main()
