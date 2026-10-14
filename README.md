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

## 4. Training & Model Evaluation

Hệ thống đánh giá độ chính xác và chất lượng huấn luyện của cả 2 mô hình qua các chỉ số tiêu chuẩn quốc tế và quy trình kiểm thử nghiêm ngặt:

### 1. Chỉ Số Đánh Giá (Evaluation Metrics)
*   **MAE (Mean Absolute Error):** Đo độ lệch giá trị thực tế theo đơn vị tiền tệ (USD). Cho biết trung bình mô hình dự báo lệch bao nhiêu USD so với giá sàn thực tế:
    $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
*   **MAPE (Mean Absolute Percentage Error):** Đo phần trăm sai số tương đối, là thước đo quan trọng nhất để kích hoạt cảnh báo rủi ro:
    $$\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$
    *   **Xanh lá (< 7.0%):** Mô hình đạt độ tin cậy cao, an toàn cho suy luận.
    *   **Vàng (7.0% - 10.0%):** Sai số trong ngưỡng chấp nhận được, duy trì giám sát.
    *   **Đỏ (> 10.0%):** Sai số vượt ngưỡng an toàn, kích hoạt pipeline tối ưu hóa GUMNet candidate.
*   **Pinball Loss (Quantile Loss):** Hàm mục tiêu huấn luyện chuyên biệt cho GUMNet nhằm ước lượng phân vị ($q \in \{0.1, 0.5, 0.9\}$):
    $$\mathcal{L}_q(y, \hat{y}) = \max(q(y - \hat{y}), (1 - q)(\hat{y} - y))$$

### 2. Kết Quả Huấn Luyện Đa Mốc Thời Gian (Training History & Validation Loss)
Dữ liệu huấn luyện lịch sử (`training_history.json`) ghi nhận độ mất mát kiểm định (Validation Loss) trên toàn bộ 7 mốc thời gian ($H \in \{1, 5, 10, 15, 20, 30, 60\}$):

| Horizon (Kỳ hạn) | GUMNet Validation Loss | HybridTriNet Validation Loss | Đánh giá xu hướng |
| :---: | :---: | :---: | :--- |
| **+1 ngày (H1)** | **0.0345** | 0.0368 | Độ chính xác cao nhất, bám sát biến động vi mô |
| **+5 ngày (H5)** | **0.0410** | 0.0425 | Nắm bắt tốt chu kỳ tuần (loại trừ cuối tuần) |
| **+10 ngày (H10)** | **0.0482** | 0.0501 | Sai số ổn định trong chu kỳ nửa tháng |
| **+15 ngày (H15)** | **0.0540** | 0.0562 | Hội tụ mượt mà theo xu hướng trung hạn |
| **+20 ngày (H20)** | **0.0592** | 0.0615 | Kiểm soát tốt biên độ dao động giá tháng |
| **+30 ngày (H30)** | **0.0671** | 0.0699 | Dự báo xu hướng tháng vững chắc |
| **+60 ngày (H60)** | **0.0798** | 0.0824 | Độ phân kỳ thấp dù dự báo kỳ hạn xa |

> *Ghi chú:* Validation Loss tăng dần đều theo độ dài horizon là đặc tính hoàn toàn tự nhiên của bài toán dự báo chuỗi thời gian (suy giảm thông tin khi phóng chiếu tương lai xa).

---

## 5. Database and Deployment

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

## 6. Streamlit Dashboard

Trực quan hóa chi tiết 5 phân hệ chức năng cốt lõi trên giao diện Dashboard điều hành:

### 1. Live Forecast (Upload & Dự Báo Đa Kỳ Hạn)
Nạp dữ liệu thị trường mới nhất và tạo dự báo tức thời cho 7 mốc thời gian $H \in \{1, 5, 10, 15, 20, 30, 60\}$ ngày kèm tùy chọn kích hoạt tự động cập nhật độ thông minh cho mô hình.

<p align="center">
  <img src="docs/images/UPLOAD.png" alt="Live Forecast Interface" width="850"/>
</p>

### 2. Backtest Metrics (Đánh Giá Sai Số & Cảnh Báo)
Sổ học bạ đo lường độ chính xác của AI qua các chỉ số sai số chuẩn quốc tế MAE (USD) và MAPE (%), tích hợp hệ thống đèn tín hiệu thông minh Đỏ / Vàng / Xanh theo ngưỡng tin cậy.

<p align="center">
  <img src="docs/images/CHISO.png" alt="Backtest Metrics & Thresholds" width="850"/>
</p>

### 3. Visualizations (Trực Quan Hóa So Sánh Xu Hướng)
So sánh trực quan giữa đường giá thực tế (nét liền) và đường dự báo của các mô hình (nét đứt), hỗ trợ phân tích đa góc nhìn theo từng mốc horizon và từng mặt hàng dầu khí.

<p align="center">
  <img src="docs/images/BIEUDOSOSANH.png" alt="Trend Comparison Visualizations" width="850"/>
</p>

### 4. Upload History & Lộ Trình Dự Báo Chi Tiết
Kiểm tra chi tiết từng con số Dự báo vs Thực tế và quản lý toàn bộ vết kiểm toán (Audit Trail) các đợt nạp dữ liệu được đồng bộ trực tiếp từ PostgreSQL 16.

<p align="center">
  <img src="docs/images/DUDOANVSTHUCTE.png" alt="Forecast vs Actual History" width="850"/>
</p>

### 5. Detailed Results (Dữ Liệu Thô & Xuất Báo Cáo)
Xem và xuất toàn bộ bảng số liệu mô phỏng, phân tích chuyên sâu cho các kỹ sư dữ liệu và chuyên gia phân tích tài chính.

<p align="center">
  <img src="docs/images/BD_DATASET.png" alt="Detailed Dataset & Results" width="850"/>
</p>

---

## 7. Project Structure

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