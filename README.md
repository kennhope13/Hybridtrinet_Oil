# Crude Oil Multi-Horizon Forecasting Platform

A multi-horizon crude oil forecasting system for 1, 5, 10, 15, 20, 30, and 60 days. It includes two deep learning architectures, GUMNet and HybridTriNet, a background GUMNet candidate evaluation workflow, and PostgreSQL-backed ingestion audit history.

---

## 1. System Architecture

The application separates the Streamlit interface, background orchestration, forecasting models, and persistent storage:

```mermaid
flowchart TD
    subgraph UI_Layer["Streamlit Web UI (Port 8502)"]
        UI["app_main.py (Multi-Tab Dashboard)"]
        Tab1["Live Forecast"]
        Tab2["Backtest Metrics"]
        Tab3["Upload Audit History"]
        Tab4["Visualizations"]
        UI --- Tab1 & Tab2 & Tab3 & Tab4
    end

    subgraph Service_Orchestration["MLOps and Orchestration Layer"]
        Engine["pipeline_engine.py\n(Background Orchestrator)"]
        Lock[".pipeline.lock\n(Process Lock and Stale Watchdog)"]
        Status[".pipeline_status.json\n(Persistent State)"]
        Worker["backtest_worker.py\n(Backtest Simulation)"]
        Trainer["train_all_horizons.py\n(Candidate Trainer)"]
        Engine <--> Lock
        Engine <--> Status
        Engine --> Worker
        Engine --> Trainer
    end

    subgraph DL_Models["Deep Learning Models"]
        GUMNet["GUMNet (Retraining and Quantile Forecasting)\nWavelet-KAN + Dilated CNN + GRU-Attention"]
        HybridTriNet["HybridTriNet (Multi-Branch Model)\nRBF-KAN + Patch Transformer + GRU"]
    end

    subgraph Storage_Layer["Separated Persistence Layer"]
        Postgres[("PostgreSQL 16\n• ingestion_batches\n• ingestion_files (JSONB)\n• app_metadata")]
        FileStore[("Filesystem and Docker Volumes\n• Datasets: datasets/\n• Checkpoints: checkpoints_multi/\n• Candidates: checkpoints_candidate/")]
    end

    UI -->|"Trigger evaluation and ingestion"| Engine
    UI -->|"Load GUMNet or HybridTriNet"| DL_Models
    UI <-->|"Audit queries and ingestion logs"| Postgres
    Engine <-->|"Promote or restore checkpoints"| FileStore
    Trainer -->|"Save candidate weights"| FileStore
```

---

## 2. Background Candidate Evaluation Workflow

`pipeline_engine.py` evaluates whether the current GUMNet should be retrained after new data is ingested:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit UI
    participant DB as PostgreSQL 16
    participant Pipe as Pipeline Engine
    participant BW as Backtest Worker
    participant TR as Candidate Trainer
    participant FS as Checkpoint Storage

    User->>UI: Upload new price data (Excel or CSV)
    UI->>DB: Record batch and file metadata (JSONB)
    UI->>UI: Merge price series and apply trading-day calendar
    UI->>Pipe: Start background evaluation
    Note over Pipe: Evaluate trigger conditions:\nnew_rows >= 5 and eval_pairs >= 10

    alt Not enough data for evaluation
        Pipe-->>UI: Skip retraining and keep current checkpoints
    else Enough data
        Pipe->>Pipe: Acquire .pipeline.lock
        Pipe->>BW: Backtest current model on new observations
        alt MAPE <= 10% (below the retraining trigger threshold)
            Pipe-->>UI: Finish evaluation without training a candidate
        else MAPE > 10%
            Pipe->>TR: Train an isolated candidate in checkpoints_candidate/
            Pipe->>BW: Compare candidate and current production checkpoints
            alt Candidate has lower MAPE
                Pipe->>FS: Back up current production checkpoints
                Pipe->>FS: Promote candidate checkpoints to checkpoints_multi/
                Pipe-->>UI: Report that the candidate was applied
            else Candidate does not improve or an error occurs
                Pipe->>FS: Keep or restore the current production checkpoints
                Pipe-->>UI: Report that current checkpoints were retained
            end
        end
        Pipe->>Pipe: Release .pipeline.lock
    end
    UI-->>User: Show forecasts and evaluation results
```

---

## 3. Model Architectures

### GUMNet: Wavelet-KAN, Dilated CNN, and GRU-Attention

GUMNet is used by the background candidate workflow and produces quantile forecasts:

- **Wavelet-KAN branch:** Uses a Mexican Hat wavelet basis, combined with SiLU activation:

  $$\text{wavelet} = (1 - x_{\text{norm}}^2) \cdot e^{-0.5 \cdot x_{\text{norm}}^2}$$

- **Dilated CNN branch:** Uses 1D convolution with `dilation=2` and `AdaptiveAvgPool1d` to capture local patterns over a wider receptive field.
- **GRU and attention branch:** Uses a two-layer GRU and `nn.MultiheadAttention` with four heads.
- **Gating and quantile head:** A softmax gating layer combines the three branches. The output includes quantiles at $q = 0.1, 0.5, 0.9$.

```mermaid
flowchart LR
    X["Input\n[Batch, Seq_Len, Features]"]
    subgraph GUMNet_Architecture["GUMNet Architecture"]
        CNN["Dilated Conv1D\n(dilation=2) + Pool"]
        GRU["Two-Layer GRU\n+ MultiheadAttention"]
        KAN["Wavelet-KAN\n(Mexican Hat) + Pool"]
        Gate["Learnable Softmax Gating"]
        Fusion["Weighted Feature Fusion\nw_cnn·CNN + w_gru·GRU + w_kan·KAN"]
        Out["Quantile Output Layer\n(Horizon × quantiles 0.1, 0.5, 0.9)"]
        X --> CNN & GRU & KAN
        CNN & GRU & KAN --> Gate
        CNN & GRU & KAN --> Fusion
        Gate -.->|"Branch weights"| Fusion
        Fusion --> Out
    end
```

<p align="center">
  <img src="docs/images/gumnet_architecture.png" alt="GUMNet Architecture Diagram" width="850"/>
</p>

### HybridTriNet: RBF-KAN, Patch Transformer, and GRU

HybridTriNet is a multi-branch forecasting model combining:

- **RBF-KAN branch:** Uses radial basis function expansion to learn nonlinear feature mappings.
- **Patch Transformer Encoder:** Splits the time series into patches (`patch_len=16`, `stride=8`), adds positional encoding, and applies `torch.nn.TransformerEncoder`.
- **GRU branch:** Captures sequential patterns in the time series.
- **Adaptive softmax gating:** Learns how to combine branch representations for each forecast horizon.

<p align="center">
  <img src="docs/images/hybridtrinet_architecture.png" alt="HybridTriNet Architecture Diagram" width="850"/>
</p>

---

## 4. Database and Deployment

The application separates relational audit data from time-series files and model checkpoints:

- **PostgreSQL 16:** Stores ingestion audit history and metadata.
  - `ingestion_batches`: Summary information for each upload batch.
  - `ingestion_files`: Per-file status, SHA-256 hash, and additional metadata in `details_json` (`JSONB`).
  - Existing records in `ingestion_history.json` are migrated to the database on first startup when that file is present.
- **Filesystem and Docker volumes:** Store time-series datasets in `datasets/` and PyTorch checkpoints and scalers in `checkpoints_multi/`.

### Run with Docker Compose

1. Create a local environment file:

   Linux or macOS:
   ```bash
   cp .env.example .env
   ```

   Windows PowerShell:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Build and start the app and database:

   ```bash
   docker compose up -d --build app
   ```

   The app waits for the database health check before starting.

3. Open the Streamlit dashboard at `http://localhost:8502`.

---

## 5. Streamlit Dashboard

1. **Live Forecast:** Upload data and view forecasts for horizons $H \in \{1, 5, 10, 15, 20, 30, 60\}$ days.
2. **Backtest Metrics:** Review MAE, MAPE, and error-threshold indicators.
3. **Upload History:** Review ingestion audit records stored in PostgreSQL.
4. **Visualizations:** Compare forecast trends with observed prices.
5. **Detailed Results:** Inspect and export evaluation data.

---

## 6. Project Structure

```text
├── checkpoints_multi/         # PyTorch checkpoints (.pt), scalers (.npy), and training history
├── datasets/                  # Time-series datasets
├── oil_forecast_research_new/ # GUMNet model and DataProcessor source
├── Hybridtrinet_Oil/src/      # HybridTriNet model source
├── app_main.py                # Main Streamlit application
├── pipeline_engine.py         # Background orchestration, state, and process lock
├── backtest_worker.py         # Backtest simulation and MAE/MAPE evaluation
├── database.py                # PostgreSQL 16 access (psycopg and JSONB)
├── docker-compose.yml         # App and PostgreSQL services
├── Dockerfile                 # Python, PyTorch, and Streamlit image
└── requirements.txt           # Python dependencies
```