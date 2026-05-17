"""
train_all_horizons.py
Huấn luyện cả GUMNet và HybridTriNet cho 6 mốc horizon: 1, 5, 10, 30, 60, 100 ngày.
Mỗi mốc tạo 1 checkpoint riêng, lưu vào thư mục checkpoints_multi/.
"""

import sys, json, random, warnings
warnings.filterwarnings("ignore") # Tắt các cảnh báo dư thừa để log sạch sẽ
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "oil_forecast_research_new-main" / "data" / "processed" / "clean_data_exo_ver1.csv"
OUT_DIR = ROOT / "checkpoints_multi"
OUT_DIR.mkdir(exist_ok=True)

DATE_COL = "Ngày"
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
MASTER_HORIZON = 60  # Mốc dài nhất
HORIZONS = [1, 5, 10, 30, 60]



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
    parser.add_argument("--update_data", action="store_true", help="Cập nhật dữ liệu vào CSV gốc")
    parser.add_argument("--new_file", type=str, help="Đường dẫn file mới nhất để nạp lẻ")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch huấn luyện")
    parser.add_argument("--models", nargs="+", default=["GUMNet"], help="Danh sách mô hình")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS, help="Danh sách chân trời")
    parser.add_argument("--force_retrain", action="store_true", help="Xóa checkpoint cũ, train lại từ đầu")
    return parser.parse_args()



def update_training_data(specific_file=None):
    """Gộp dữ liệu mới vào clean_data_exo_ver1.csv"""
    try:
        base_df = pd.read_csv(DATA_PATH)
        base_df[DATE_COL] = pd.to_datetime(base_df[DATE_COL], errors="coerce")
        
        if specific_file:
            files = [Path(specific_file)]
            flush_print(f"🎯 Chỉ nạp lẻ file mới: {files[0].name}")
        else:
            flush_print("🔄 Quét toàn bộ thư mục datasets...")
            data_dir = ROOT / "datasets"
            files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"]]
        
        new_data = []
        for i, f in enumerate(files):
            try:
                if f.suffix.lower() == ".csv": df = pd.read_csv(f)
                else: df = pd.read_excel(f)
                
                dcol = None
                for c in df.columns:
                    if str(c).lower() in ["ngày", "ngay", "date"]: dcol = c; break
                
                if dcol:
                    df = df.rename(columns={dcol: DATE_COL})
                    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format='mixed')
                    df = df.dropna(subset=[DATE_COL])
                    cols = [DATE_COL] + [c for c in TARGET_COLS if c in df.columns]
                    new_data.append(df[cols])
            except: continue
        
        if new_data:
            full_new = pd.concat(new_data)
            combined = pd.concat([base_df, full_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
            combined = combined.interpolate().bfill().ffill()
            combined.to_csv(DATA_PATH, index=False)
            flush_print(f"✅ Đã cập nhật dataset. Tổng cộng {len(combined)} dòng.")
        else:
            flush_print("⚠️ Không có dữ liệu mới.")
    except Exception as e:
        flush_print(f"❌ Lỗi cập nhật: {e}")


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
    df = df.interpolate().bfill().ffill()
    return df

# ═══════════════════════  GUMNet TRAINING  ════════════════════════════════════

def train_gumnet_horizon(df, horizon, device, epochs=None):
    """Train GUMNet for a specific horizon."""
    n_epochs = epochs if epochs is not None else GUMNET_EPOCHS
    # Import — luôn đặt đúng thư mục ở đầu sys.path
    gumnet_dir = str(ROOT / "oil_forecast_research_new-main")
    # Xoá path cũ của project khác, đặt path mới lên đầu
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
        print(f"  [SKIP] Not enough data for horizon={horizon}")
        return

    split = int(len(X) * (1 - VAL_RATIO))
    train_loader = DataLoader(PetroleumDataset(X[:split], y[:split]), batch_size=GUMNET_BATCH, shuffle=True)
    val_loader = DataLoader(PetroleumDataset(X[split:], y[split:]), batch_size=GUMNET_BATCH)

    model = GUMNet(
        seq_len=GUMNET_SEQ_LEN, input_dim=len(feature_cols),
        output_dim=len(TARGET_COLS), horizon=horizon,
        d_feat=64, num_quantiles=3,
    ).to(device)


    # Nạp checkpoint cũ của ĐÚNG MỐC này nếu có để Finetune
    ckpt_path = OUT_DIR / f"gumnet_h{horizon}.pt"
    current_lr = GUMNET_LR
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            current_lr = GUMNET_LR * 0.2
            flush_print(f"   ♻️ Đã nạp GUMNet h{horizon} để học tiếp (Finetune)...")
        except:
            flush_print(f"   ⚠️ Không thể nạp checkpoint GUMNet h{horizon}, sẽ học mới.")


    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=5e-4)

    quantiles = [0.1, 0.5, 0.9]


    def q_loss(pred, target):
        loss = 0.0
        for i, q in enumerate(quantiles):
            err = target - pred[..., i]
            loss += torch.maximum((q - 1) * err, q * err).mean()
        return loss / len(quantiles)

    best_val = float("inf")
    ckpt_path = OUT_DIR / f"gumnet_h{horizon}.pt"
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
            print(f"    Epoch {epoch+1}/{n_epochs}  train={tl:.6f}  val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "seq_len": GUMNET_SEQ_LEN, "horizon": horizon,
                "num_quantiles": 3, "quantiles": quantiles,
                "feature_cols": feature_cols, "target_cols": TARGET_COLS,
                "input_dim": len(feature_cols), "output_dim": len(TARGET_COLS),
                "feature_scaler": processor.feature_scaler,
                "target_scaler": processor.target_scaler,
                "date_col": DATE_COL, "d_feat": 64,
            }, ckpt_path)

        else:
            wait += 1
            if wait >= patience:
                print(f"    [Early Stopping] Dừng tại epoch {epoch+1} do không cải thiện.")
                break

    print(f"    Best Val Loss: {best_val:.6f}")

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

def train_hybrid_horizon(df, horizon, device, epochs=None):
    """Train HybridTriNet for a specific horizon."""
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
        print(f"  [SKIP] Missing columns: {missing}")
        return

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


    # Nạp checkpoint cũ của ĐÚNG MỐC này nếu có để Finetune
    ckpt_path = OUT_DIR / f"hybrid_h{horizon}.pt"
    current_lr = HYBRID_LR
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=True)
            current_lr = HYBRID_LR * 0.2
            flush_print(f"   ♻️ Đã nạp Hybrid h{horizon} để học tiếp (Finetune)...")
        except Exception as ex:
            flush_print(f"   ⚠️ Checkpoint h{horizon} không tương thích ({ex}), sẽ train lại từ đầu.")



    # Nới lỏng weight_decay để tránh Underfitting (hạ từ 1e-3 về 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=5e-4)


    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=current_lr * 2,
        total_steps=max(1, n_epochs * len(tr_loader)),
        pct_start=0.15,
    )

    best_val = float("inf")
    best_state = None
    ckpt_path = OUT_DIR / f"hybrid_h{horizon}.pt"

    # Nới lỏng patience để AI học sâu hơn (nâng lên 15)
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
                print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state:
        torch.save(best_state, ckpt_path)

    # Save metadata — mỗi horizon có thư mục meta riêng
    run_dir = OUT_DIR / f"hybrid_h{horizon}_meta"
    run_dir.mkdir(exist_ok=True)
    np.save(run_dir / "x_mu.npy", mu)
    np.save(run_dir / "x_sd.npy", sd)
    np.save(run_dir / "y_mu.npy", y_mu)
    np.save(run_dir / "y_sd.npy", y_sd)
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": f_cols, "tgt_idx": tgt_idx, "K": K, "H": horizon}, f, indent=2)

    print(f"    Saved: {ckpt_path}  meta: {run_dir.name}  (best_val={best_val:.6f})")



# ═══════════════════════  MAIN  ═══════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    flush_print("🚀 HỆ THỐNG HUẤN LUYỆN ĐÃ SẴN SÀNG.")
    
    # 1. Cập nhật dữ liệu nếu được yêu cầu
    if args.update_data:
        update_training_data(args.new_file)

        
    # 2. Thiết lập thiết bị và dữ liệu
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flush_print(f"🖥️ Thiết bị sử dụng: {device}")
    
    df = read_data()
    flush_print(f"📊 Dữ liệu sẵn sàng: {len(df)} dòng.")
    
    # 3. Huấn luyện từng mốc Horizon riêng biệt (Multi-Model Mode)
    for hz in args.horizons:
        flush_print(f"\n{'='*40}")
        flush_print(f"🚀 ĐANG HUẤN LUYỆN MỐC: {hz} NGÀY")
        flush_print(f"{'='*40}")

        # Xóa checkpoint cũ nếu bật chế độ Train lại từ đầu
        if args.force_retrain:
            for pattern in [f"gumnet_h{hz}.pt", f"hybrid_h{hz}.pt"]:
                old = OUT_DIR / pattern
                if old.exists():
                    old.unlink()
                    flush_print(f"   🗑️ Đã xóa checkpoint cũ: {pattern}")
            flush_print(f"   🔁 Bắt đầu TRAIN LạI Từ ĐẦU (không dùng checkpoint cũ)")
        

        if "GUMNet" in args.models:
            flush_print(f"🧠 [GUMNet] Horizon {hz}d...")
            train_gumnet_horizon(df, hz, device, epochs=args.epochs)
            
        if "HybridTriNet" in args.models:
            flush_print(f"🧬 [HybridTriNet] Horizon {hz}d...")
            train_hybrid_horizon(df, hz, device, epochs=args.epochs)
    
    flush_print("\n✅ TẤT CẢ CÁC MÔ HÌNH ĐÃ ĐƯỢC CẬP NHẬT!")


    print(f"Checkpoints đã lưu tại: {OUT_DIR}")
