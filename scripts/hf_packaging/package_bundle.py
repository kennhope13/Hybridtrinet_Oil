import hashlib
import json
import os
import shutil
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmark" / "retrain_clean_15feat"
BUNDLE_DIR = ROOT_DIR / "hf_releases" / "hybridtrinet-oil-forecasting"
SCRIPTS_DIR = ROOT_DIR / "scripts" / "hf_packaging"

HORIZONS = [1, 5, 10, 15, 20, 30, 60]

ARCH_HYPERPARAMS = {
    "d_feat": 96,
    "kan_M": 8,
    "kan_depth": 2,
    "kan_drop": 0.2,
    "gru_hidden": 128,
    "gru_layers": 1,
    "gru_drop": 0.1,
    "attn_dmodel": 64,
    "attn_heads": 4,
    "attn_layers": 2,
    "attn_drop": 0.2,
    "patch_len": 16,
    "stride": 8,
}


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package():
    print(f"[PACKAGING] Staging Hugging Face model bundle from clean 15-feat run to: {BUNDLE_DIR}")
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Setup subdirectories
    models_dir = BUNDLE_DIR / "models"
    inference_dir = BUNDLE_DIR / "inference"
    metrics_dir = BUNDLE_DIR / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy inference code
    print("[PACKAGING] Copying standalone inference and verification scripts...")
    shutil.copy2(SCRIPTS_DIR / "model_architecture.py", inference_dir / "model_architecture.py")
    shutil.copy2(SCRIPTS_DIR / "run_inference.py", inference_dir / "run_inference.py")
    shutil.copy2(SCRIPTS_DIR / "verify_bundle.py", inference_dir / "verify_bundle.py")

    # 3. Copy metrics
    print("[PACKAGING] Copying clean holdout metrics and training history...")
    shutil.copy2(BENCHMARK_DIR / "holdout_metrics.csv", metrics_dir / "holdout_metrics.csv")
    shutil.copy2(BENCHMARK_DIR / "training_history.json", metrics_dir / "training_history.json")

    # 4. Process each horizon
    for hz in HORIZONS:
        print(f"[PACKAGING] Assembling horizon h{hz}...")
        hz_dir = models_dir / f"h{hz}"
        hz_dir.mkdir(parents=True, exist_ok=True)

        ckpt_source = BENCHMARK_DIR / f"hybrid_h{hz}.pt"
        meta_source_dir = BENCHMARK_DIR / f"hybrid_h{hz}_meta"

        shutil.copy2(ckpt_source, hz_dir / "model.pt")
        for fname in ["x_mu.npy", "x_sd.npy", "y_mu.npy", "y_sd.npy"]:
            shutil.copy2(meta_source_dir / fname, hz_dir / fname)

        # Read feature_cols
        with open(meta_source_dir / "feature_cols.json", "r", encoding="utf-8") as f:
            feat_meta = json.load(f)

        with open(hz_dir / "feature_cols.json", "w", encoding="utf-8") as f:
            json.dump(feat_meta, f, indent=2)

        # Generate model_config.json
        model_config = {
            "model_type": "HybridTriNet",
            "horizon": hz,
            "K": feat_meta["K"],
            "H": hz,
            "D_in": len(feat_meta["feature_cols"]),
            "D_out": len(feat_meta["tgt_idx"]),
            "target_columns": [feat_meta["feature_cols"][i] for i in feat_meta["tgt_idx"]],
            **ARCH_HYPERPARAMS,
        }
        with open(hz_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

    # 5. Generate LICENSE
    license_content = """MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DISCLAIMER: The checkpoints and model architecture are released strictly for
research, benchmark evaluation, and educational reproducibility purposes.
Raw dataset redistribution is not included.
"""
    with open(BUNDLE_DIR / "LICENSE", "w", encoding="utf-8") as f:
        f.write(license_content.strip() + "\n")

    # 6. Generate Model Card README.md
    readme_content = """---
language:
- en
- vi
license: mit
library_name: pytorch
tags:
- time-series
- forecasting
- oil-price
- multi-horizon
- kan
- transformer
- gru
- mlops
metrics:
- mae
- mape
- rmse
---

# HybridTriNet: Multi-Horizon Oil Price Forecasting

This repository hosts official checkpoints, architecture definitions, and isolated preprocessing metadata for **HybridTriNet**, a tri-branch deep neural network designed for multi-horizon oil and refined product price forecasting.

> **Disclaimer**: The checkpoints and model architecture are released strictly for research, benchmark evaluation, and educational reproducibility purposes. Raw dataset redistribution is not included.

---

## 1. Architecture Overview

HybridTriNet fuses three complementary representation learning branches via a **Dynamic Gated Fusion** network:
1. **KAN Branch (Kolmogorov-Arnold Network)**: Captures complex non-linear feature interactions using learnable Radial Basis Function expansions ($M=8$, depth=2).
2. **GRU Branch**: Models sequential temporal dependencies and autoregressive momentum (hidden dimension=128).
3. **Patch Transformer Branch**: Captures long-range multi-scale attention patterns across segmented temporal patches (patch length=16, stride=8, $d_{model}=64$, 4 heads, 2 encoder layers).

Predictions are generated simultaneously across multiple forecasting horizons ($H \\in \\{1, 5, 10, 15, 20, 30, 60\\}$ steps) with input lookback window $K = 64$ trading steps.

---

## 2. Input/Output Inference Contract

To ensure valid forecasting without schema mismatch, inputs must conform strictly to the following contract:

- **Lookback Window ($K$)**: Exactly 64 consecutive trading steps.
- **Input Dimension ($D_{in} = 15$)**: The input tensor consists of 15 authentic market, calendar, and risk features (synthetic temporal indices such as index counters are deliberately omitted to avoid temporal trend shortcut learning):
  1. `BRT DTD` (Brent Dated Benchmark)
  2. `BRT KH` (Brent Futures/Forward)
  3. `WTI` (West Texas Intermediate Benchmark)
  4. `NgayTrongTuan` (Day of week: 0–6)
  5. `ThangTrongNam` (Month of year: 1–12)
  6. `QuyTrongNam` (Quarter: 1–4)
  7. `Nam` (Year)
  8. `NgayLe` (Public holiday indicator: 0 or 1)
  9. `SuKienDacBiet` (Special geopolitical/economic event indicator)
  10. `USD_Index` (US Dollar Index)
  11. `GPRD` (Geopolitical Risk Daily Index)
  12. `MG95` (Mogas 95 Refined Gasoline Price)
  13. `MG92` (Mogas 92 Refined Gasoline Price)
  14. `DO 0.001%` (Diesel Oil 0.001% Price)
  15. `DO 0.05%` (Diesel Oil 0.05% Price)
- **Target Columns ($D_{out} = 4$)**:
  - `MG95` (Index 11)
  - `MG92` (Index 12)
  - `DO 0.001%` (Index 13)
  - `DO 0.05%` (Index 14)
- **Lagged Target Semantics**: Target columns are provided strictly as historical lagged inputs within the 64-step lookback window; future target values are never used as input.
- **Output Scale**: Predictions are output in the **original target units defined by the dataset metadata** following automated inverse standardization (`out_real = out_norm * y_sd + y_mu`).

---

## 3. Holdout Evaluation Benchmark

Models were trained and evaluated chronologically on out-of-sample holdout test windows (Test Period: `2023-09-11` to `2026-05-25`).

> **Note**: Accuracy metrics reflect out-of-sample holdout performance during benchmark training. The verification script (`verify_bundle.py`) tests bundle integrity, determinism, and software pipeline correctness, not future empirical forecasting accuracy.

| Model | Horizon ($H$) | MAE | MAPE (%) | RMSE | Windows Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| HybridTriNet | 1-step | 8.12 | 6.76% | 19.15 | 699 |
| HybridTriNet | 5-step | 9.58 | 8.40% | 18.09 | 695 |
| HybridTriNet | 10-step | 9.26 | 7.82% | 20.19 | 690 |
| HybridTriNet | 15-step | 10.27 | 8.85% | 21.48 | 685 |
| HybridTriNet | 20-step | 9.86 | 8.46% | 21.11 | 680 |
| HybridTriNet | 30-step | 10.58 | 9.23% | 22.22 | 670 |
| HybridTriNet | 60-step | 10.89 | 9.95% | 21.49 | 640 |

Complete metric tables and comparison baselines are available in `metrics/holdout_metrics.csv` and `metrics/training_history.json`.

---

## 4. Repository Structure

```text
hybridtrinet-oil-forecasting/
├── README.md
├── LICENSE
├── checksums.sha256
├── inference/
│   ├── model_architecture.py     # Standalone PyTorch architecture
│   ├── run_inference.py          # Runnable inference pipeline with denormalization
│   └── verify_bundle.py          # Deep verification suite (integrity, determinism, shapes)
├── models/
│   ├── h1/                       # Horizon 1 model + metadata
│   ├── h5/                       # Horizon 5 model + metadata
│   ├── h10/                      # Horizon 10 model + metadata
│   ├── h15/                      # Horizon 15 model + metadata
│   ├── h20/                      # Horizon 20 model + metadata
│   ├── h30/                      # Horizon 30 model + metadata
│   └── h60/                      # Horizon 60 model + metadata
└── metrics/
    ├── holdout_metrics.csv       # Multi-horizon evaluation benchmark
    └── training_history.json     # Epoch-level training logs
```

Each horizon folder (`models/h<H>/`) is strictly self-contained and binds:
- `model.pt`: Checkpoint weights.
- `model_config.json`: Model hyperparameters ($K, H, D_{in}, D_{out}$, layers, attention heads).
- `feature_cols.json`: Input feature ordering and target indices.
- `x_mu.npy`, `x_sd.npy`: Input standardization parameters.
- `y_mu.npy`, `y_sd.npy`: Target inverse standardization parameters.

---

## 5. Quickstart & Inference

```python
import numpy as np
from inference.run_inference import load_forecasting_pipeline, forecast

# 1. Load pipeline for 60-step horizon
bundle = load_forecasting_pipeline("models/h60", device="cpu")

# 2. Input shape: (Batch, K=64, D_in=15) in raw feature scale
sample_raw_input = np.random.normal(loc=bundle["x_mu"], scale=bundle["x_sd"], size=(1, 64, 15))

# 3. Run inference (standardization -> forward pass -> inverse standardization)
forecast_real = forecast(bundle, sample_raw_input)
print("Forecast shape (B, H=60, D_out=4):", forecast_real.shape)
print("Products:", bundle["target_cols"])
```

Or run directly via command line:

```bash
python inference/run_inference.py --model_dir models/h60
```

To run deep integrity and determinism verification:

```bash
python inference/verify_bundle.py --bundle_dir .
```
"""
    with open(BUNDLE_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")

    # Clean any __pycache__ before computing checksums
    pycache = inference_dir / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    # 7. Compute SHA-256 for all staged files
    print("[PACKAGING] Generating checksums.sha256...")
    checksum_lines = []
    for fpath in sorted(BUNDLE_DIR.rglob("*")):
        if fpath.is_file() and fpath.name != "checksums.sha256" and "__pycache__" not in str(fpath):
            rel_path = fpath.relative_to(BUNDLE_DIR).as_posix()
            sha256_hash = calculate_sha256(fpath)
            checksum_lines.append(f"{sha256_hash}  {rel_path}")

    with open(BUNDLE_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    print(f"[SUCCESS] Packaged {len(checksum_lines)} artifacts into {BUNDLE_DIR}")


if __name__ == "__main__":
    package()
