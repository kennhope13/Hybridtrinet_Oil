# HybridTriNet Oil Price Forecasting Platform

An enterprise-grade, reproducible multi-horizon time-series forecasting research platform and inference pipeline. The repository features **HybridTriNet**—a tri-branch deep neural network combining Kolmogorov-Arnold Networks (KAN), Recurrent Neural Networks (GRU), and Patch Transformers with dynamic gated fusion—alongside benchmark baselines, comprehensive data pipelines, standalone inference artifacts, and a full Streamlit decision-support dashboard.

> [!IMPORTANT]
> **Research & Engineering Status**: This project is developed strictly for research, quantitative benchmarking, and educational reproducibility. It does **not** constitute financial advice, investment recommendations, or an endorsement of algorithmic trading strategies. All benchmark holdout metrics reflect historical chronological evaluation and do not guarantee future empirical forecasting performance.

---

## 1. Project Overview

Crude oil and refined petroleum markets are subject to high volatility, geopolitical shocks, and non-linear macroeconomic dependencies. Traditional statistical models often struggle to balance high-frequency temporal momentum with multi-scale non-linear interactions across diverse market regimes.

**HybridTriNet** addresses this challenge through:
- **Multi-Horizon Forecasting**: Simultaneous point forecasting across 7 distinct operational horizons: **1, 5, 10, 15, 20, 30, and 60 trading days/steps** ($H \in \{1, 5, 10, 15, 20, 30, 60\}$).
- **Fixed Lookback Window ($K=64$)**: Predictions are conditioned on a historical lookback window of 64 consecutive trading observations.
- **Multi-Asset Target Output ($D_{out}=4$)**: Jointly forecasts 4 refined petroleum commodities:
  1. `MG95` (Mogas 95 Gasoline)
  2. `MG92` (Mogas 92 Gasoline)
  3. `DO 0.001%` (Diesel Oil 0.001% Sulfur)
  4. `DO 0.05%` (Diesel Oil 0.05% Sulfur)
- **Denormalized Physical Units**: Automated pipeline inverse-transforms predictions back into original market scale defined by dataset metadata (USD/barrel / commodity units).

---

## 2. Model Pipeline

The end-to-end forecasting pipeline enforces strict causal chronological boundaries to prevent lookahead leakage, shortcut learning, and out-of-distribution drift.

```mermaid
flowchart LR
    A["Raw Market Time Series<br/>(Energy, Macro, Risk)"] --> B["Causal Preprocessing<br/>(Forward-fill only)"]
    B --> C["Chronological Split<br/>(Train 70% / Val 15% / Holdout 15%)"]
    C --> D["Windowing<br/>(Lookback K=64, Horizon H)"]
    D --> E["Per-Feature Standardization<br/>(Fit on Train split only)"]
    E --> F["HybridTriNet Forward Pass<br/>(KAN + GRU + Patch Transformer)"]
    F --> G["Dynamic Gated Fusion<br/>(Learned branch weights)"]
    G --> H["Inverse Standardization<br/>(Denormalization to real scale)"]
    H --> I["Evaluation & Dashboard<br/>(Holdout Metrics, Streamlit UI, PDF Export)"]
```

<p align="center">
  <img src="docs/figures/forecasting_pipeline.png" alt="HybridTriNet Forecasting Pipeline" width="900"/>
</p>

---

## 3. Model Architecture

HybridTriNet is structured into three parallel representation learning branches that capture orthogonal characteristics of market time-series:

1. **KAN Branch (Kolmogorov-Arnold Network)**:
   - Replaces traditional multi-layer perceptrons with learnable univariate non-linear activation functions parameterized via Radial Basis Function (RBF) expansions.
   - Initial linear projection: maps $(B, K \times D_{in})$ down to $(B, D_{in})$.
   - 2 stacked `KANBlock` modules ($M=8$ centers, hidden dimension=96, dropout=0.2).
   - Generates feature representation $f_k \in \mathbb{R}^{B \times 96}$.
2. **GRU Branch (Sequential Temporal Momentum)**:
   - Processes $(B, K, D_{in})$ sequentially using a Gated Recurrent Unit (hidden dimension=128, layers=1).
   - Extracts the final hidden state $h_T$, followed by LayerNorm and linear projection to $f_g \in \mathbb{R}^{B \times 96}$.
3. **Patch Transformer Branch (Multi-Scale Temporal Attention)**:
   - Segments the lookback series into overlapping temporal patches (`patch_len=16`, `stride=8`), projecting raw sub-windows to dimension $d_{model}=64$.
   - Augments patch embeddings with sinusoidal positional encodings (`PosEnc`).
   - Processes representations through a 2-layer `TransformerEncoder` (4 attention heads, feedforward dimension=256, dropout=0.2).
   - Global temporal average pooling projects patch states to $f_a \in \mathbb{R}^{B \times 96}$.
4. **Dynamic Gated Fusion**:
   - Each branch produces an independent horizon prediction: $y_k, y_g, y_a \in \mathbb{R}^{B \times H \times D_{out}}$.
   - Concatenated representation $[f_k, f_g, f_a] \in \mathbb{R}^{B \times 288}$ passes through a 2-layer gating MLP with GELU activations to output time-step-specific gating weights $w \in \mathbb{R}^{B \times H \times 3}$.
   - Softmax normalization ensures a partition of unity across branches:
     $$\hat{y}_t = w_{t, 1} y_{k, t} + w_{t, 2} y_{g, t} + w_{t, 3} y_{a, t}$$

<p align="center">
  <img src="docs/figures/hybridtrinet_architecture.png" alt="HybridTriNet Neural Architecture" width="900"/>
</p>

---

## 4. Architecture Diagram & Reproducibility Script

The architecture and pipeline visual diagrams are programmatically synthesized directly from model metadata and configuration files to guarantee fidelity with actual codebase parameters:

```powershell
python scripts/visualize_architecture.py
```
Outputs:
- Architecture schematic: `docs/figures/hybridtrinet_architecture.png`
- Pipeline diagram: `docs/figures/forecasting_pipeline.png`

---

## 5. Feature Contract

To ensure mathematical and operational consistency, inference inputs must conform strictly to the following contract:

- **Lookback Length ($K$)**: 64 consecutive trading observations.
- **Input Dimension ($D_{in} = 15$)**: Exactly 15 market, macroeconomic, calendar, and risk features (synthetic temporal indices such as index counters are deliberately omitted to avoid temporal trend shortcut learning):
  1. `BRT DTD`: Brent Dated Benchmark
  2. `BRT KH`: Brent Futures/Forward Benchmark
  3. `WTI`: West Texas Intermediate Benchmark
  4. `NgayTrongTuan`: Day of week (0 to 6)
  5. `ThangTrongNam`: Month of year (1 to 12)
  6. `QuyTrongNam`: Quarter of year (1 to 4)
  7. `Nam`: Calendar year
  8. `NgayLe`: Public holiday binary flag (0 or 1)
  9. `SuKienDacBiet`: Geopolitical / market event indicator (0 or 1)
  10. `USD_Index`: US Dollar Currency Index
  11. `GPRD`: Geopolitical Risk Daily Index
  12. `MG95`: Historical Mogas 95 price
  13. `MG92`: Historical Mogas 92 price
  14. `DO 0.001%`: Historical Diesel 0.001% price
  15. `DO 0.05%`: Historical Diesel 0.05% price
- **Lagged Target Semantics**: Target columns (indices 11 to 14) are provided strictly as historical autoregressive inputs within the $K=64$ lookback window; future target values are never used as input.
- **Horizon-Specific Scalers**: Each horizon binds its own isolated standardization parameters (`x_mu.npy`, `x_sd.npy`, `y_mu.npy`, `y_sd.npy`) fitted exclusively on training windows.
- **Output Tensor Shape**: $(B, H, D_{out})$ where $B$ is batch size, $H \in \{1, 5, 10, 15, 20, 30, 60\}$, and $D_{out} = 4$.

---

## 6. Training and Evaluation

### Chronological Holdout Benchmark
Training, validation, and holdout splits are strictly partitioned chronologically (70% train / 15% validation / 15% holdout) over the dataset span (`2008-05-01` to `2026-05-25`). 
- **Holdout Test Period**: `2023-09-11` to `2026-05-25`.
- **Training Recipe**: AdamW optimizer, OneCycleLR learning rate schedule, Cosine annealing, Smooth L1 Loss, early stopping patience on validation loss.

### Benchmark Metrics (Clean 15-Feature Release)
The table below documents out-of-sample holdout performance evaluated across all rolling origins within the holdout period:

| Model | Horizon ($H$) | MAE (Original Units) | MAPE (%) | RMSE (Original Units) | Evaluated Windows |
| :--- | :---: | :---: | :---: | :---: | :---: |
| HybridTriNet | **1 day** | 8.12 | 6.76% | 19.15 | 699 |
| HybridTriNet | **5 days** | 9.58 | 8.40% | 18.09 | 695 |
| HybridTriNet | **10 days** | 9.26 | 7.82% | 20.19 | 690 |
| HybridTriNet | **15 days** | 10.27 | 8.85% | 21.48 | 685 |
| HybridTriNet | **20 days** | 9.86 | 8.46% | 21.11 | 680 |
| HybridTriNet | **30 days** | 10.58 | 9.23% | 22.22 | 670 |
| HybridTriNet | **60 days** | 10.89 | 9.95% | 21.49 | 640 |

> **Evaluation Distinction**: The cleaned 15-feature retraining removes the synthetic temporal index and preserves competitive holdout performance across all seven forecast horizons without systematic degradation. Software verification scripts (`verify_bundle.py`) verify model packaging, determinism, and software pipeline correctness, and are separate from future empirical forecasting accuracy evaluations. Checkpoints from legacy runs containing synthetic temporal indices are archived separately as experimental artifacts.

---

## 7. Reproducible Inference

The repository provides standalone, self-contained inference modules that run independently of external training infrastructure.

### Quick Inference (Python API)
```python
import numpy as np
from inference.run_inference import load_forecasting_pipeline, forecast

# 1. Load pipeline for 60-day horizon on CPU
bundle = load_forecasting_pipeline("hf_releases/hybridtrinet-oil-forecasting/models/h60", device="cpu")

# 2. Prepare raw input window: shape (1, K=64, D_in=15)
synthetic_input = np.random.normal(loc=bundle["x_mu"], scale=bundle["x_sd"], size=(1, 64, 15)).astype(np.float32)

# 3. Execute inference (standardization -> forward pass -> inverse standardization)
predictions = forecast(bundle, synthetic_input)
print("Forecast shape (B, H=60, D_out=4):", predictions.shape)
print("Predicted Commodities:", bundle["target_cols"])
```

### Command Line Execution
```powershell
# Run sample inference on horizon 60
python inference/run_inference.py --model_dir hf_releases/hybridtrinet-oil-forecasting/models/h60

# Run deep bundle verification across all 7 horizons
python inference/verify_bundle.py --bundle_dir hf_releases/hybridtrinet-oil-forecasting
```

---

## 8. Packaging & Hugging Face Distribution

Release artifacts are staged under `hf_releases/hybridtrinet-oil-forecasting` following modern MLOps model distribution best practices:
- **`package_bundle.py`**: Packages checkpoints, isolated `model_config.json`, per-horizon scalers, and computes cryptographically signed `checksums.sha256`.
- **`upload_to_hf.py`**: Securely uploads staged bundles to Hugging Face Hub (reads token exclusively from `$env:HF_TOKEN`).
- **`verify_clean_download.py`**: Post-upload verification script that pulls snapshots into an isolated clean temporary directory to confirm repository integrity.

*(Note: Model packaging is complete and ready for deployment. Checkpoint weights are distributed via Hugging Face Hub to keep Git history lightweight).*

---

## 9. Model Limitations

1. **Not Financial or Trading Advice**: Predictions reflect statistical pattern matching on historical market data and should not be used as the primary basis for financial or commercial commitments.
2. **Distribution Drift & Regime Shifts**: Macroeconomic shocks, policy changes (e.g. OPEC+ quotas, trade tariffs), or unprecedented geopolitical crises can alter underlying dynamics beyond the training distribution.
3. **Data Dependency**: Model metrics are conditional on the specific historical dataset window; out-of-sample performance may vary across different market eras.
4. **Data Privacy**: Raw proprietary datasets are excluded from public Git repositories. Checkpoints and architecture code are released strictly under MIT research licenses.

---

## 10. Repository Structure

```text
Hybridtrinet_Oil/
├── benchmark/                        # Benchmark holdout metrics and training runs
│   ├── retrain_clean_15feat/         # Official clean 15-feature training artifacts
│   └── retrain_2026-10-14/           # Legacy experimental 16-feature artifacts
├── docs/                             # Documentation assets and diagrams
│   ├── figures/                      # Architecture and pipeline diagrams (PNG/SVG)
│   └── images/                       # UI screenshots and visual plots
├── hf_releases/                      # Staged release bundle for Hugging Face Hub
│   └── hybridtrinet-oil-forecasting/ # Checkpoints, scalers, and model card
├── inference/                        # Standalone, runnable inference suite
│   ├── model_architecture.py         # Independent PyTorch HybridTriNet implementation
│   ├── run_inference.py              # Self-contained forecast execution script
│   └── verify_bundle.py              # Automated deep bundle verification suite
├── scripts/                          # Automation, packaging, and visualization
│   ├── hf_packaging/                 # Packaging, upload, and clean download scripts
│   └── visualize_architecture.py     # Programmatic architecture diagram generator
├── src/                              # Application model modules and training utilities
├── tests/                            # Pytest test suite (unit tests, bundle tests)
├── app_main.py                       # Streamlit web dashboard application
├── data_pipeline.py                  # Ingestion, validation, and feature preparation
├── pipeline_engine.py                # Background execution engine
├── train_all_horizons.py             # Multi-horizon automated training runner
├── Dockerfile                        # Containerization spec
├── docker-compose.yml                # Multi-service container orchestration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation and portfolio guide
```
