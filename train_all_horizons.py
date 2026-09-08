"""
train_all_horizons.py
Huấn luyện cả GUMNet và HybridTriNet cho 6 mốc horizon: 1, 5, 10, 30, 60, 100 ngày.
Mỗi mốc tạo 1 checkpoint riêng, lưu vào thư mục checkpoints_multi/.
"""

import os, sys, json, random, time, warnings, uuid
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore") # Tắt các cảnh báo dư thừa để log sạch sẽ
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from project_io import load_checkpoint
from torch.utils.data import DataLoader

def safe_torch_save(obj, path, retries=6, delay=1.5):
    """Lưu checkpoint an toàn trước lỗi khóa file trên Windows (vd: app Streamlit đang mở/đọc
    đúng file này để hiển thị lúc job huấn luyện cố ghi đè -> torch.save ném RuntimeError
    'cannot be opened'). Ghi ra 1 file tạm trước, rồi mới đổi tên đè lên file thật (os.replace) —
    nếu bước nào bị khóa thì đợi rồi thử lại vài lần thay vì làm chết cả job giữa chừng.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)  # đổi tên là thao tác nguyên tử, ít bị khóa hơn ghi đè trực tiếp
            return
        except Exception as e:
            last_err = e
            flush_print(f"   ⚠️ Không ghi được checkpoint {path.name} (lần {attempt}/{retries}): {e} — thử lại sau {delay:.0f}s...")
            time.sleep(delay)
    raise RuntimeError(f"Không thể lưu checkpoint {path} sau {retries} lần thử: {last_err}")

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "oil_forecast_research_new-main" / "data" / "processed" / "clean_data_exo_ver1.csv"
OUT_DIR = ROOT / "checkpoints_multi"
OUT_DIR.mkdir(exist_ok=True)

# Quản lý lock huấn luyện đồng bộ giữa app_main.py và train_all_horizons.py
TRAIN_LOCK_FILE = ROOT / ".training.lock"

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # Tiến trình tồn tại nhưng khác quyền truy cập
    except OSError:
        return False
    except Exception:
        return False

def _read_lock_info():
    if not TRAIN_LOCK_FILE.exists():
        return None
    try:
        return json.loads(TRAIN_LOCK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None

def _atomic_write_lock(lock_data):
    """Ghi lock file bằng file tạm rồi rename nguyên tử."""
    tmp = TRAIN_LOCK_FILE.with_suffix(".lock.tmp")
    tmp.write_text(json.dumps(lock_data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, TRAIN_LOCK_FILE)

def acquire_or_takeover_lock(job_id, models, horizons):
    """
    Tiếp quản lock nếu job_id trùng khớp, hoặc tạo lock mới theo cách nguyên tử (mode='x').
    Nếu có job khác đang chạy (PID alive), báo lỗi và sys.exit(1).
    """
    my_pid = os.getpid()
    info = _read_lock_info()

    if info:
        existing_pid = info.get("pid")
        existing_job_id = info.get("job_id")

        # 1. Tiếp quản lock nếu job_id trùng khớp (app đã giữ chỗ trước Popen)
        if job_id and existing_job_id == job_id:
            info["pid"] = my_pid
            info["models"] = models
            info["horizons"] = horizons
            _atomic_write_lock(info)
            flush_print(f"🔒 Đã tiếp quản lock cho Job ID: {job_id} (PID {my_pid})")
            return job_id

        # 2. Xung đột nếu có job khác đang chạy
        if existing_pid and existing_pid != my_pid and _pid_alive(existing_pid):
            flush_print(
                f"❌ Đã có một Job Huấn luyện khác đang chạy (PID {existing_pid}, Job ID: {existing_job_id}, "
                f"bắt đầu lúc {info.get('started_at', '?')}). Dừng ngay để tránh ghi đè checkpoint."
            )
            sys.exit(1)
        else:
            # Lock rác (PID đã chết) -> xóa bỏ an toàn
            try:
                TRAIN_LOCK_FILE.unlink(missing_ok=True)
            except Exception:
                pass

    # Tạo lock mới nguyên tử
    actual_job_id = job_id or uuid.uuid4().hex[:12]
    lock_content = json.dumps({
        "job_id": actual_job_id,
        "pid": my_pid,
        "models": models,
        "horizons": horizons,
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2)

    try:
        with open(TRAIN_LOCK_FILE, mode="x", encoding="utf-8") as f:
            f.write(lock_content)
        flush_print(f"🔒 Đã tạo khóa huấn luyện nguyên tử: Job ID {actual_job_id} (PID {my_pid})")
    except FileExistsError:
        info = _read_lock_info()
        if info and info.get("job_id") == actual_job_id:
            info["pid"] = my_pid
            _atomic_write_lock(info)
            return actual_job_id
        elif info and info.get("pid") and _pid_alive(info.get("pid")):
            flush_print(f"❌ Xung đột khóa: Job khác ({info.get('job_id')}) vừa chiếm quyền chạy.")
            sys.exit(1)
        else:
            _atomic_write_lock(json.loads(lock_content))

    return actual_job_id

def release_own_lock(job_id):
    """Chỉ xóa lock nếu job_id trong file lock khớp với job_id hiện tại."""
    try:
        info = _read_lock_info()
        if info and info.get("job_id") == job_id:
            TRAIN_LOCK_FILE.unlink(missing_ok=True)
            flush_print(f"🔓 Đã giải phóng khóa huấn luyện: Job ID {job_id}")
    except Exception:
        pass

DATE_COL = "Ngày"
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
MASTER_HORIZON = 60  # Mốc dài nhất
HORIZONS = [1, 5, 10, 15, 20, 30, 60]



# ─── Config ───
GUMNET_SEQ_LEN = 30
GUMNET_EPOCHS  = 150
GUMNET_LR      = 2e-4  # Giảm LR để Finetune ổn định hơn
GUMNET_BATCH   = 32

HYBRID_SEQ_LEN = 64
HYBRID_EPOCHS  = 200
HYBRID_LR      = 2e-4  # Giảm LR để Finetune ổn định hơn
HYBRID_BATCH   = 32

VAL_RATIO = 0.2
SEED = 42

import argparse, sys

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, default=None, help="Mã định danh job được cấp phát từ app")
    parser.add_argument("--update_data", action="store_true", help="Cập nhật dữ liệu vào CSV gốc")
    parser.add_argument("--new_file", type=str, help="Đường dẫn file mới nhất để nạp lẻ")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch huấn luyện")
    parser.add_argument("--models", nargs="+", default=["GUMNet"], help="Danh sách mô hình")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS, help="Danh sách chân trời")
    parser.add_argument("--force_retrain", action="store_true", help="Huấn luyện lại từ đầu (bảo toàn checkpoint cũ đến khi kiểm tra thành công)")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục xuất checkpoint (dành cho candidate cô lập)")
    return parser.parse_args()



def update_training_data(specific_file=None):
    """Gộp dữ liệu mới vào clean_data_exo_ver1.csv một cách nguyên tử và an toàn."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu gốc tại: {DATA_PATH}")

    try:
        base_df = pd.read_csv(DATA_PATH)
    except Exception as e:
        raise RuntimeError(f"Không thể đọc file dữ liệu gốc {DATA_PATH.name}: {e}")

    if DATE_COL not in base_df.columns:
        raise ValueError(f"Tập dữ liệu gốc {DATA_PATH.name} thiếu cột ngày '{DATE_COL}'")

    base_df[DATE_COL] = pd.to_datetime(base_df[DATE_COL], errors="coerce")
    base_df = base_df.dropna(subset=[DATE_COL])
    for c in TARGET_COLS:
        if c in base_df.columns:
            base_df[c] = pd.to_numeric(base_df[c], errors="coerce")

    if specific_file:
        files = [Path(specific_file)]
        flush_print(f"🎯 Chỉ nạp lẻ file mới: {files[0].name}")
    else:
        flush_print("🔄 Quét toàn bộ thư mục datasets...")
        data_dir = ROOT / "datasets"
        if not data_dir.exists():
            flush_print("⚠️ Thư mục datasets không tồn tại.")
            return
        files = sorted(
            [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"] and not f.name.startswith("~$")],
            key=lambda x: x.stat().st_mtime
        )

    if not files:
        flush_print("ℹ️ Không có file dữ liệu mới nào để cập nhật.")
        return

    # Sử dụng index theo DATE_COL để cập nhật chọn lọc theo từng cột
    base_df = base_df.set_index(DATE_COL)

    for f in files:
        flush_print(f"   📂 Đang xử lý file: {f.name}")
        try:
            if f.suffix.lower() == ".csv":
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception as e:
            raise ValueError(f"Lỗi đọc file {f.name}: {e}")

        # Tìm cột ngày
        dcol = None
        for c in df.columns:
            if str(c).strip().lower() in ["ngày", "ngay", "date"]:
                dcol = c
                break

        if not dcol:
            raise ValueError(f"File {f.name} không chứa cột ngày hợp lệ (cần cột 'Ngày', 'ngay', hoặc 'date')")

        df = df.rename(columns={dcol: DATE_COL})
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")
        df = df.dropna(subset=[DATE_COL])

        if df.empty:
            raise ValueError(f"File {f.name} không có dòng dữ liệu hợp lệ nào sau khi phân tích ngày")

        present_targets = [c for c in TARGET_COLS if c in df.columns]
        if not present_targets:
            raise ValueError(
                f"File {f.name} không chứa bất kỳ cột mặt hàng dầu nào trong danh sách: {TARGET_COLS}"
            )

        # Chuyển đổi các cột mục tiêu sang kiểu số
        for c in present_targets:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Loại bỏ dòng mà tất cả các cột mục tiêu đều là NaN
        df = df.dropna(subset=present_targets, how="all")
        if df.empty:
            raise ValueError(f"File {f.name} không có giá trị số hợp lệ nào cho các cột: {present_targets}")

        # Lấy subset ngày và các cột mục tiêu có trong file, loại trùng lặp trong chính file (giữ dòng sau)
        sub_df = df[[DATE_COL] + present_targets].drop_duplicates(subset=[DATE_COL], keep="last").set_index(DATE_COL)

        # 1. Thêm các ngày mới chưa có trong base_df
        new_dates = sub_df.index.difference(base_df.index)
        if len(new_dates) > 0:
            base_df = pd.concat([base_df, sub_df.loc[new_dates]])

        # 2. Cập nhật các cột mục tiêu có trong file mới (giá trị mới đè giá trị cũ, các cột khác giữ nguyên)
        base_df.update(sub_df)
        flush_print(f"      ↳ Cập nhật {len(sub_df)} dòng từ file {f.name} (các cột: {present_targets})")

    base_df = base_df.reset_index()
    base_df = base_df.sort_values(DATE_COL).reset_index(drop=True)

    # Nội suy các cột số nếu có thiếu
    num_cols = base_df.select_dtypes(include=[np.number]).columns
    base_df[num_cols] = base_df[num_cols].interpolate().bfill().ffill()

    # Ghi nguyên tử ra file tạm rồi os.replace
    tmp_data_path = DATA_PATH.with_suffix(".csv.tmp")
    base_df.to_csv(tmp_data_path, index=False)
    os.replace(tmp_data_path, DATA_PATH)
    flush_print(f"✅ Đã cập nhật thành công dataset gốc ({len(base_df)} dòng).")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    for c in df.columns:
        if c != DATE_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate().bfill().ffill()
    return df

# ═══════════════════════  GUMNet TRAINING  ════════════════════════════════════

def train_gumnet_horizon(df, horizon, device, epochs=None, force_retrain=False, job_id="job"):
    """Train GUMNet for a specific horizon with checkpoint protection."""
    n_epochs = epochs if epochs is not None else GUMNET_EPOCHS
    # Import — luôn đặt đúng thư mục ở đầu sys.path
    gumnet_dir = str(ROOT / "oil_forecast_research_new-main")
    sys.path = [p for p in sys.path if p != gumnet_dir] 
    sys.path.insert(0, gumnet_dir)
    for m in [k for k in list(sys.modules) if k.startswith("src")]:
        del sys.modules[m]
    from src.model.dataset import DataProcessor, PetroleumDataset
    from src.model.model import GUMNet

    feature_cols = [c for c in df.columns if c != DATE_COL]
    
    processor = DataProcessor(seq_len=GUMNET_SEQ_LEN, horizon=horizon)
    X, y = processor.prepare_data(df, TARGET_COLS, feature_cols, is_train=True)
    
    if len(X) == 0:
        flush_print(f"  [SKIP] Not enough data for horizon={horizon}")
        return None

    split = int(len(X) * (1 - VAL_RATIO))
    train_loader = DataLoader(PetroleumDataset(X[:split], y[:split]), batch_size=GUMNET_BATCH, shuffle=True)
    val_loader = DataLoader(PetroleumDataset(X[split:], y[split:]), batch_size=GUMNET_BATCH)

    model = GUMNet(
        seq_len=GUMNET_SEQ_LEN, input_dim=len(feature_cols),
        output_dim=len(TARGET_COLS), horizon=horizon,
        d_feat=64, num_quantiles=3,
    ).to(device)

    ckpt_path = OUT_DIR / f"gumnet_h{horizon}.pt"
    tmp_ckpt = OUT_DIR / f".tmp_{job_id}_gumnet_h{horizon}.pt"
    current_lr = GUMNET_LR

    if not force_retrain and ckpt_path.exists():
        try:
            ckpt = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            current_lr = GUMNET_LR * 0.2
            flush_print(f"   ♻️ Đã nạp trọng số GUMNet h{horizon} sẵn có để tối ưu tiếp (Finetune)...")
        except Exception:
            flush_print(f"   🌱 Khởi tạo mô hình GUMNet h{horizon} để tối ưu hóa mới...")
    else:
        flush_print(f"   🌱 Khởi tạo mô hình GUMNet h{horizon} ngẫu nhiên (Train mới)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=5e-4)
    quantiles = [0.1, 0.5, 0.9]

    def q_loss(pred, target):
        loss = 0.0
        for i, q in enumerate(quantiles):
            err = target - pred[..., i]
            loss += torch.maximum((q - 1) * err, q * err).mean()
        return loss / len(quantiles)

    best_val = float("inf")
    patience = 15
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = q_loss(pred, yb)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                v_losses.append(q_loss(pred, yb).item())

        tl = np.mean(t_losses)
        vl = np.mean(v_losses) if v_losses else float("inf")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            flush_print(f"    Epoch {epoch+1}/{n_epochs}  train={tl:.6f}  val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            wait = 0
            safe_torch_save({
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "seq_len": GUMNET_SEQ_LEN, "horizon": horizon,
                "num_quantiles": 3, "quantiles": quantiles,
                "feature_cols": feature_cols, "target_cols": TARGET_COLS,
                "input_dim": len(feature_cols), "output_dim": len(TARGET_COLS),
                "feature_scaler": processor.feature_scaler,
                "target_scaler": processor.target_scaler,
                "date_col": DATE_COL, "d_feat": 64,
                "job_id": job_id,
                "best_val_loss": float(vl),
            }, tmp_ckpt)
        else:
            wait += 1
            if wait >= patience:
                flush_print(f"    [Early Stopping] Dừng tại epoch {epoch+1} do không cải thiện.")
                break

    # Xác thực checkpoint mới lưu trước khi cập nhật checkpoint chính thức — bao gồm cả kiểm tra
    # job_id để chắc chắn đây đúng là kết quả của lượt chạy này, không phải lẫn từ job khác.
    if tmp_ckpt.exists():
        try:
            verified = load_checkpoint(tmp_ckpt, map_location="cpu")
            required_keys = ["model_state_dict", "horizon", "seq_len", "feature_cols", "target_cols",
                              "input_dim", "output_dim", "feature_scaler", "target_scaler", "date_col"]
            missing_keys = [k for k in required_keys if k not in verified] if isinstance(verified, dict) else required_keys
            if missing_keys:
                raise ValueError(f"Checkpoint thiếu trường bắt buộc: {missing_keys}")
            if verified["horizon"] != horizon:
                raise ValueError(f"Checkpoint có horizon={verified['horizon']} khác với horizon đang huấn luyện={horizon}")
            if not isinstance(verified["model_state_dict"], dict) or len(verified["model_state_dict"]) == 0:
                raise ValueError("model_state_dict rỗng hoặc không hợp lệ")
            if verified.get("job_id") != job_id:
                raise ValueError(f"Checkpoint thuộc job_id={verified.get('job_id')!r}, không khớp job đang chạy={job_id!r}")
            os.replace(tmp_ckpt, ckpt_path)
            flush_print(f"   ✅ [GUMNet h{horizon}] Checkpoint đã được kiểm tra và cập nhật thành công: {ckpt_path.name}")
        except Exception as e:
            if tmp_ckpt.exists():
                try: tmp_ckpt.unlink(missing_ok=True)
                except Exception: pass
            raise RuntimeError(f"Xác thực checkpoint GUMNet h{horizon} thất bại: {e}")
    else:
        raise RuntimeError(f"Không có checkpoint hợp lệ nào được tạo cho GUMNet h{horizon}")

    flush_print(f"    Best Val Loss: {best_val:.6f}")
    return best_val

# ═══════════════════════  HybridTriNet TRAINING  ══════════════════════════════

def prepare_hybrid_data(df, feature_cols_hybrid):
    """Chuẩn bị dữ liệu cho HybridTriNet (thêm cột time features)."""
    df2 = df.copy()
    dt = df2[DATE_COL]
    df2["NgayTrongTuan"] = dt.dt.dayofweek
    df2["ThangTrongNam"] = dt.dt.month
    df2["QuyTrongNam"] = dt.dt.quarter
    df2["Nam"] = dt.dt.year
    df2["NgayLe"] = 0
    df2["SuKienDacBiet"] = 0
    if "GPR" in df2.columns:
        df2["GPRD"] = df2["GPR"]
    else:
        df2["GPRD"] = 0
    df2["Unnamed: 0"] = range(len(df2))
    return df2

def train_hybrid_horizon(df, horizon, device, epochs=None, force_retrain=False, job_id="job"):
    """Train HybridTriNet for a specific horizon with checkpoint protection."""
    n_epochs = epochs if epochs is not None else HYBRID_EPOCHS
    hybrid_dir = str(ROOT / "Hybridtrinet_Oil")
    sys.path = [p for p in sys.path if p != hybrid_dir]
    sys.path.insert(0, hybrid_dir)
    for m in [k for k in list(sys.modules) if k.startswith("src")]:
        del sys.modules[m]
    from src.model.hybrid_trinet import HybridTriNet
    from src.model.training import standardize, build_windows, WindowDS

    # Feature cols cho HybridTriNet
    f_cols = ["Unnamed: 0", "BRT DTD", "BRT KH", "WTI",
              "NgayTrongTuan", "ThangTrongNam", "QuyTrongNam", "Nam",
              "NgayLe", "SuKienDacBiet", "USD_Index", "GPRD",
              "MG95", "MG92", "DO 0.001%", "DO 0.05%"]
    
    df2 = prepare_hybrid_data(df, f_cols)
    
    # Check all cols exist
    missing = [c for c in f_cols if c not in df2.columns]
    if missing:
        flush_print(f"  [SKIP] Missing columns: {missing}")
        return None

    Y_raw = df2[f_cols].values.astype(np.float32)
    Y_std, mu, sd = standardize(Y_raw)

    K = HYBRID_SEQ_LEN
    X_all, Y_all = build_windows(Y_std, K, horizon)

    # target = last 4 cols (MG95, MG92, DO 0.001%, DO 0.05%)
    tgt_idx = [f_cols.index(c) for c in TARGET_COLS]
    Y_tgt = Y_all[:, :, tgt_idx]  # [N, H, 4]

    # Re-standardize target
    y_raw_flat = Y_raw[:, tgt_idx]
    y_mu = np.nanmean(y_raw_flat, axis=0).astype(np.float32)
    y_sd = np.nanstd(y_raw_flat, axis=0).astype(np.float32)
    y_sd = np.where(y_sd < 1e-8, 1.0, y_sd)

    split = int(len(X_all) * (1 - VAL_RATIO))
    tr_ds = WindowDS(X_all[:split], Y_tgt[:split])
    va_ds = WindowDS(X_all[split:], Y_tgt[split:])
    tr_loader = DataLoader(tr_ds, batch_size=HYBRID_BATCH, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=HYBRID_BATCH)

    model = HybridTriNet(
        k=K, H=horizon, D_in=len(f_cols), D_out=len(TARGET_COLS),
        d_feat=96, kan_M=8, kan_depth=2,
        gru_hidden=128, gru_layers=1,
        attn_dmodel=64, attn_heads=4, attn_layers=2,
        patch_len=16, stride=8,
    ).to(device)

    ckpt_path = OUT_DIR / f"hybrid_h{horizon}.pt"
    tmp_ckpt = OUT_DIR / f".tmp_{job_id}_hybrid_h{horizon}.pt"
    current_lr = HYBRID_LR

    if not force_retrain and ckpt_path.exists():
        try:
            state = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=True)
            current_lr = HYBRID_LR * 0.2
            flush_print(f"   ♻️ Đã nạp trọng số Hybrid h{horizon} sẵn có để tối ưu tiếp (Finetune)...")
        except Exception:
            flush_print(f"   🌱 Khởi tạo mô hình Hybrid h{horizon} để tối ưu hóa mới...")
    else:
        flush_print(f"   🌱 Khởi tạo mô hình Hybrid h{horizon} ngẫu nhiên (Train mới)...")

    # Nới lỏng weight_decay để tránh Underfitting (hạ từ 1e-3 về 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=5e-4)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=current_lr * 2,
        total_steps=max(1, n_epochs * len(tr_loader)),
        pct_start=0.15,
    )

    best_val = float("inf")
    best_state = None
    patience, bad = 15, 0

    for epoch in range(n_epochs):
        model.train()
        t_losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out, _ = model(xb)
            out = out.view(yb.shape[0], horizon, len(TARGET_COLS))
            loss = torch.nn.functional.smooth_l1_loss(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out, _ = model(xb)
                out = out.view(yb.shape[0], horizon, len(TARGET_COLS))
                v_losses.append(torch.nn.functional.smooth_l1_loss(out, yb).item())

        tl = np.mean(t_losses)
        vl = np.mean(v_losses) if v_losses else float("inf")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            flush_print(f"    Epoch {epoch+1}/{n_epochs}  train={tl:.6f}  val={vl:.6f}")

        if vl < best_val - 1e-7:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                flush_print(f"    Early stop at epoch {epoch+1}")
                break

    if not best_state:
        raise RuntimeError(f"Không có trọng số hợp lệ được tạo cho HybridTriNet h{horizon}")

    # Checkpoint và metadata (feature_cols.json, x_mu/x_sd/y_mu/y_sd.npy) phải cùng "phiên bản" —
    # trước đây checkpoint được os.replace() TRƯỚC, rồi mới ghi metadata TRỰC TIẾP vào file chính
    # thức (không qua file tạm, không xác thực) — nếu ghi metadata giữa chừng bị lỗi (hết đĩa, mất
    # điện...), sẽ để lại checkpoint MỚI đi cùng metadata CŨ/hỏng, model nạp sai. Giờ: dựng toàn bộ
    # metadata ra file tạm (đặt tên theo job_id) trước, xác thực đọc lại được hết, RỒI mới thay thế
    # nguyên tử từng file — checkpoint xong mới tới metadata — để không có khoảng hở nào cho việc
    # phối 1 checkpoint mới với 1 bộ metadata cũ/hỏng.
    run_dir = OUT_DIR / f"hybrid_h{horizon}_meta"
    run_dir.mkdir(exist_ok=True)
    tmp_meta = {
        "x_mu.npy": run_dir / f".tmp_{job_id}_x_mu.npy",
        "x_sd.npy": run_dir / f".tmp_{job_id}_x_sd.npy",
        "y_mu.npy": run_dir / f".tmp_{job_id}_y_mu.npy",
        "y_sd.npy": run_dir / f".tmp_{job_id}_y_sd.npy",
        "feature_cols.json": run_dir / f".tmp_{job_id}_feature_cols.json",
    }
    tmp_paths = list(tmp_meta.values()) + [tmp_ckpt]

    def _cleanup_tmp():
        for p in tmp_paths:
            try: p.unlink(missing_ok=True)
            except Exception: pass

    try:
        safe_torch_save(best_state, tmp_ckpt)
        verified = load_checkpoint(tmp_ckpt, map_location="cpu")
        if not isinstance(verified, dict) or len(verified) == 0:
            raise ValueError("Trọng số HybridTriNet không hợp lệ hoặc rỗng")

        np.save(tmp_meta["x_mu.npy"], mu)
        np.save(tmp_meta["x_sd.npy"], sd)
        np.save(tmp_meta["y_mu.npy"], y_mu)
        np.save(tmp_meta["y_sd.npy"], y_sd)
        with open(tmp_meta["feature_cols.json"], "w", encoding="utf-8") as f:
            json.dump({"feature_cols": f_cols, "tgt_idx": tgt_idx, "K": K, "H": horizon, "job_id": job_id}, f, indent=2)

        # Xác thực lại toàn bộ metadata tạm đọc được đúng định dạng trước khi thay thế bất cứ gì
        for name in ("x_mu.npy", "x_sd.npy", "y_mu.npy", "y_sd.npy"):
            np.load(tmp_meta[name])
        with open(tmp_meta["feature_cols.json"], encoding="utf-8") as f:
            meta_check = json.load(f)
        if meta_check.get("H") != horizon or meta_check.get("job_id") != job_id:
            raise ValueError("Metadata tạm không khớp horizon/job_id đang huấn luyện")

        # Mọi thứ đã xác thực OK — giờ mới thay thế nguyên tử: checkpoint trước, metadata sau.
        os.replace(tmp_ckpt, ckpt_path)
        for final_name, tmp_p in tmp_meta.items():
            os.replace(tmp_p, run_dir / final_name)
        flush_print(f"   ✅ [HybridTriNet h{horizon}] Checkpoint + metadata đã được kiểm tra và cập nhật đồng bộ: {ckpt_path.name}")
    except Exception as e:
        _cleanup_tmp()
        raise RuntimeError(f"Xác thực checkpoint/metadata HybridTriNet h{horizon} thất bại: {e}")

    flush_print(f"    Saved: {ckpt_path}  meta: {run_dir.name}  (best_val={best_val:.6f})")
    return best_val



# ═══════════════════════  MAIN  ═══════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = acquire_or_takeover_lock(args.job_id, args.models, args.horizons)
    import atexit
    atexit.register(release_own_lock, job_id)
    flush_print(f"🚀 HỆ THỐNG HUẤN LUYỆN ĐÃ SẴN SÀNG (Job ID: {job_id}).")
    
    # 1. Cập nhật dữ liệu nếu được yêu cầu
    if args.update_data:
        update_training_data(args.new_file)

    # 2. Thiết lập thiết bị và dữ liệu
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flush_print(f"🖥️ Thiết bị sử dụng: {device}")
    
    df = read_data()
    flush_print(f"📊 Dữ liệu sẵn sàng: {len(df)} dòng.")
    
    results = {}
    all_val_losses = []

    # 3. Huấn luyện từng mốc Horizon riêng biệt (Multi-Model Mode)
    for hz in args.horizons:
        flush_print(f"\n{'='*40}")
        flush_print(f"🚀 ĐANG HUẤN LUYỆN MỐC: {hz} NGÀY")
        flush_print(f"{'='*40}")

        if args.force_retrain:
            flush_print("   🔁 Chế độ TRAIN LẠI TỪ ĐẦU (checkpoint cũ được bảo toàn cho đến khi xác thực checkpoint mới)")

        if "GUMNet" in args.models:
            flush_print(f"🧠 [GUMNet] Horizon {hz}d...")
            v_loss = train_gumnet_horizon(df, hz, device, epochs=args.epochs, force_retrain=args.force_retrain, job_id=job_id)
            # Trước đây nếu v_loss là None (SKIP do không đủ dữ liệu), job vẫn lặng lẽ bỏ qua mốc
            # này rồi tiếp tục — cuối cùng vẫn in "TẤT CẢ ĐÃ CẬP NHẬT" dù thiếu hẳn 1 mốc, khiến
            # giao diện báo thành công sai. Giờ 1 mốc/model bị SKIP = toàn bộ job thất bại ngay.
            if v_loss is None:
                flush_print(f"❌ Mốc {hz} ngày (GUMNet) bị bỏ qua (không đủ dữ liệu) — dừng toàn bộ job, không báo thành công khi thiếu kết quả.")
                sys.exit(1)
            results[f"GUMNet_h{hz}"] = round(float(v_loss), 6)
            results[f"h{hz}"] = round(float(v_loss), 6)
            all_val_losses.append(float(v_loss))

        if "HybridTriNet" in args.models:
            flush_print(f"🧬 [HybridTriNet] Horizon {hz}d...")
            v_loss = train_hybrid_horizon(df, hz, device, epochs=args.epochs, force_retrain=args.force_retrain, job_id=job_id)
            if v_loss is None:
                flush_print(f"❌ Mốc {hz} ngày (HybridTriNet) bị bỏ qua (thiếu cột dữ liệu) — dừng toàn bộ job, không báo thành công khi thiếu kết quả.")
                sys.exit(1)
            results[f"HybridTriNet_h{hz}"] = round(float(v_loss), 6)
            all_val_losses.append(float(v_loss))
            if f"h{hz}" not in results:
                results[f"h{hz}"] = round(float(v_loss), 6)

    # Xác nhận đã có đủ kết quả cho MỌI cặp (model, horizon) được yêu cầu trước khi báo thành công
    # — phòng trường hợp lạ nào đó lọt qua 2 chỗ sys.exit(1) ở trên (an toàn 2 lớp).
    _expected = {f"{m}_h{h}" for m in args.models for h in args.horizons}
    _missing = sorted(_expected - set(results.keys()))
    if _missing:
        flush_print(f"❌ Thiếu kết quả cho: {', '.join(_missing)} — KHÔNG báo thành công.")
        sys.exit(1)

    flush_print("\n✅ TẤT CẢ CÁC MÔ HÌNH ĐÃ ĐƯỢC CẬP NHẬT!")
    flush_print(f"Checkpoints đã lưu tại: {OUT_DIR}")
    
    # 4. Lưu lại lịch sử phiên huấn luyện vào training_history.json
    try:
        import datetime
        now = datetime.datetime.now()
        session_id = f"TR-{now.strftime('%Y%m%d-%H%M%S')}"
        history_file = OUT_DIR / "training_history.json"
        
        entries = []
        history_read_failed = False
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                history_read_failed = True

        if history_read_failed:
            flush_print(
                "⚠️ Không đọc được training_history.json cũ (có thể đang bị khóa/hỏng) — "
                "BỎ QUA lưu phiên này vào lịch sử để tránh ghi đè mất lịch sử cũ."
            )
        else:
            d_start = df[DATE_COL].min().strftime("%d/%m/%Y") if DATE_COL in df.columns else "01/05/2008"
            d_end = df[DATE_COL].max().strftime("%d/%m/%Y") if DATE_COL in df.columns else "04/09/2026"

            entry = {
                "session_id": session_id,
                "job_id": job_id,
                "timestamp": now.strftime("%d/%m/%Y %H:%M:%S"),
                "mode": "Train lại từ đầu" if args.force_retrain else "Finetune (Cập nhật)",
                "models": args.models,
                "device": str(device).upper(),
                "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (6 vCPUs)",
                "epochs": args.epochs,
                "horizons": [f"{h}d" for h in args.horizons],
                "data_info": {
                    "total_rows": len(df),
                    "date_range": f"{d_start} ➔ {d_end}",
                    "start_date": d_start,
                    "end_date": d_end,
                    "targets": TARGET_COLS
                },
                "results": results,
                "avg_val_loss": round(sum(all_val_losses) / len(all_val_losses), 6) if all_val_losses else None,
                "status": "success",
            }
            entries.insert(0, entry)
            tmp_history_file = history_file.with_suffix(".json.tmp")
            with open(tmp_history_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_history_file, history_file)

            flush_print(f"📝 Đã lưu thông tin phiên huấn luyện: {session_id}")
    except Exception as ex:
        flush_print(f"ℹ️ Không thể lưu training_history.json: {ex}")
