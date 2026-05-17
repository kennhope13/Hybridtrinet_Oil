"""
Multi-Model Oil Price Forecast – Evaluation Hub (Robust Version)
"""

import sys, json, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# --- Config DLL (Application Control Policy) ---
import os
os.environ["STREAMLIT_PYARROW_ENABLED"] = "false" # Tắt PyArrow để tránh lỗi chặn DLL

import streamlit as st
import torch

# st.sidebar.info("🚀 Phiên bản: 1.0.6 - HTML Render Fix")

# Hàm hiển thị DataFrame an toàn để tránh lỗi DLL Blocked (Application Control Policy)
def safe_dataframe(df, **kwargs):
    try:
        # 1. Thử hiển thị bằng dataframe (đẹp nhất, cần pyarrow)
        st.dataframe(df, **kwargs)
    except Exception:
        try:
            # 2. Nếu lỗi, thử hiển thị bằng HTML (100% an toàn, không cần DLL, không cần tabulate)
            html = df.to_html(classes='table table-striped', justify='center', border=0)
            st.write(html, unsafe_allow_html=True)
        except Exception:
            # 3. GIẢI PHÁP CUỐI CÙNG: Hiển thị bằng bảng cơ bản nhất
            st.write(df)

# === CONFIG ===
ROOT = Path(__file__).resolve().parent
BUILTIN_CSV = ROOT / "oil_forecast_research_new-main" / "data" / "processed" / "clean_data_exo_ver1.csv"
CKPT_DIR = ROOT / "checkpoints_multi"

TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
DATE_COL = "Ngày"
HORIZONS = [1, 5, 10, 15, 20, 30, 60]
CUTOFF_DATE = pd.Timestamp("2025-09-20")

MODEL_DEFS = {
    "GUMNet": {
        "proj_dir": ROOT / "oil_forecast_research_new-main",
        "mod": "src.model.model", "cls": "GUMNet", "kind": "quantile",
    },
}

# === DATA HELPERS ===

def load_df(path):
    path = Path(path)
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='utf-8')
            if df.columns[0].startswith('Ng'): # Xử lý lỗi font chữ ở đầu file csv
                 df = df.rename(columns={df.columns[0]: DATE_COL})
        
        # Làm sạch tên cột và xóa khoảng trắng
        df.columns = [str(c).strip() for c in df.columns]
        
        # Tìm cột ngày linh hoạt (chấp nhận Ngay, Ngay, Date, Day...)
        potential_date_cols = [c for c in df.columns if any(x in c.lower() for x in ["ng", "date", "time"])]
        if potential_date_cols:
            actual_col = potential_date_cols[0]
            if actual_col != DATE_COL:
                df = df.rename(columns={actual_col: DATE_COL})
        
        if DATE_COL in df.columns:
            # Ép kiểu ngày tháng và normalize (loại bỏ giờ phút giây)
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format='mixed')
            
            df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
            df[DATE_COL] = df[DATE_COL].dt.normalize()
        
        # Ép kiểu số cho các cột còn lại
        for c in df.columns:
            if c != DATE_COL: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.interpolate().bfill().ffill()
        return df
    except Exception as e:
        st.sidebar.error(f"⚠️ Lỗi đọc file {path.name}: {e}")
        return pd.DataFrame()

def generate_time_features(df):
    if DATE_COL not in df.columns: return df
    dt = df[DATE_COL]
    for col, val in [("NgayTrongTuan", dt.dt.dayofweek), ("ThangTrongNam", dt.dt.month),
                      ("QuyTrongNam", dt.dt.quarter), ("Nam", dt.dt.year)]:
        if col not in df.columns: df[col] = val
    for col in ["NgayLe", "SuKienDacBiet"]:
        if col not in df.columns: df[col] = 0
    if "GPRD" not in df.columns: df["GPRD"] = df.get("GPR", 0)
    if "Unnamed: 0" not in df.columns: df["Unnamed: 0"] = range(len(df))
    return df

def enrich_with_exo(df, base_df):
    df = generate_time_features(df)
    missing = [c for c in base_df.columns if c not in df.columns and c != DATE_COL]
    if not missing: return df
    
    # Merge bằng ngày đã normalize
    merged = pd.merge(df, base_df[[DATE_COL] + missing], on=DATE_COL, how="left")
    merged[missing] = merged[missing].ffill().bfill()
    for c in missing:
        if merged[c].isna().any(): merged[c] = merged[c].fillna(base_df[c].iloc[-1])
    return merged

# === MODEL LOADING ===

def _swap_src(proj_dir):
    d = str(proj_dir)
    sys.path = [p for p in sys.path if p != d]
    sys.path.insert(0, d)
    for m in [k for k in list(sys.modules) if k.startswith("src")]: del sys.modules[m]

@st.cache_resource
def load_model(name, horizon):
    """Nạp mô hình chuyên biệt cho từng mốc (đọc đúng cấu trúc file thực tế)."""
    try:
        conf = MODEL_DEFS[name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Force device for debugging if needed (uncomment below if user wants to force)
        # device = "cuda"
        _swap_src(conf["proj_dir"])
        mod = importlib.import_module(conf["mod"])
        importlib.reload(mod)
        cls = getattr(mod, conf["cls"])

        if name == "GUMNet":
            # GUMNet lưu toàn bộ meta trong file .pt
            ckpt_path = CKPT_DIR / f"gumnet_h{horizon}.pt"
            if not ckpt_path.exists():
                return None, None, device
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = cls(
                seq_len=ckpt["seq_len"], input_dim=ckpt["input_dim"],
                output_dim=ckpt["output_dim"], horizon=ckpt["horizon"],
                d_feat=ckpt.get("d_feat", 64), num_quantiles=ckpt["num_quantiles"],
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            meta = {
                "feature_cols": [c.strip() for c in ckpt["feature_cols"]],
                "target_cols":  [c.strip() for c in ckpt["target_cols"]],
                "seq_len": ckpt["seq_len"], "horizon": ckpt["horizon"], "kind": "quantile",
                "feature_scaler": ckpt["feature_scaler"],
                "target_scaler":  ckpt["target_scaler"],
            }

        else:
            # HybridTriNet: file .pt + thư mục meta riêng
            ckpt_path = CKPT_DIR / f"hybrid_h{horizon}.pt"
            meta_dir  = CKPT_DIR / f"hybrid_h{horizon}_meta"
            if not ckpt_path.exists() or not meta_dir.exists():
                return None, None, device

            with open(meta_dir / "feature_cols.json") as f:
                fj = json.load(f)

            f_cols = [c.strip() for c in fj.get("feature_cols", TARGET_COLS)]
            K = fj.get("K", 64)
            # Dùng H từ metadata (H mà model được train), không dùng horizon yêu cầu
            H_model = fj.get("H", horizon)

            model = cls(
                k=K, H=H_model, D_in=len(f_cols), D_out=len(TARGET_COLS),
                d_feat=96, kan_M=8, kan_depth=2,
                gru_hidden=128, gru_layers=1,
                attn_dmodel=64, attn_heads=4, attn_layers=2,
                patch_len=16, stride=8,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            meta = {
                "feature_cols": f_cols, "target_cols": TARGET_COLS,
                "seq_len": K, "horizon": H_model, "kind": "point",
                "x_mu": np.load(meta_dir / "x_mu.npy"),
                "x_sd": np.load(meta_dir / "x_sd.npy"),
                "y_mu": np.load(meta_dir / "y_mu.npy"),
                "y_sd": np.load(meta_dir / "y_sd.npy"),
            }


        model.eval()
        return model, meta, device
    except Exception as e:
        return None, str(e), "cpu"


def predict_from_df(model, meta, df, device):
    k      = meta["seq_len"]
    f_cols = meta["feature_cols"]
    t_cols = meta["target_cols"]
    n_tgt  = len(t_cols)

    # Chuẩn bị đầu vào
    available = [c for c in f_cols if c in df.columns]
    X = df[available].values
    if len(available) < len(f_cols):
        pad = np.zeros((len(X), len(f_cols) - len(available)))
        X = np.hstack([X, pad])

    if "feature_scaler" in meta:
        X = meta["feature_scaler"].transform(X)
    else:
        X = (X - meta["x_mu"]) / (meta["x_sd"] + 1e-8)

    x_in = torch.tensor(X[-k:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out, _ = model(x_in)

    if meta["kind"] == "quantile":
        # GUMNet: out shape (1, H, n_quantiles, n_tgt) hoặc (1, H, n_tgt, n_quantiles)
        raw = out.cpu().numpy()[0]  # (H, ...)
        if raw.ndim == 3:           # (H, n_tgt, n_quantiles)
            p50 = meta["target_scaler"].inverse_transform(raw[:, :, 1])
        else:                       # fallback
            p50 = meta["target_scaler"].inverse_transform(raw[..., 1])
    else:
        # HybridTriNet: out shape (1, H*n_tgt) hoặc (1, H, n_tgt)
        raw = out.cpu().numpy()[0]
        if raw.ndim == 1:
            total = raw.size
            H = total // n_tgt
            raw = raw.reshape(H, n_tgt)
        # y_mu/y_sd là của toàn bộ feature, chỉ lấy n_tgt cột cuối (target cols)
        y_mu = np.array(meta["y_mu"]).reshape(-1)[-n_tgt:]
        y_sd = np.array(meta["y_sd"]).reshape(-1)[-n_tgt:]
        p50 = raw * (y_sd + 1e-8) + y_mu


    # Sinh ngày (bỏ qua cuối tuần)
    h_out = p50.shape[0]
    last  = df[DATE_COL].iloc[-1]
    dates, d = [], last
    while len(dates) < h_out:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)

    result = pd.DataFrame(p50[:len(dates)], columns=t_cols)
    result.insert(0, DATE_COL, [pd.Timestamp(dt).normalize() for dt in dates[:len(result)]])
    return result


# === SIMULATION ENGINE ===

def run_upload_simulation(base_path, upload_files, start_date):
    base_full = load_df(base_path)
    base = base_full[base_full[DATE_COL] < start_date].copy()
    all_records = []
    
    with open(ROOT / "sim_log.txt", "w", encoding="utf-8") as logf:
        logf.write(f"Simulation started. Files: {len(upload_files)}\n")
        
        status_text = st.empty()
        task_idx = 0
        total_tasks = len(upload_files) * len(MODEL_DEFS)

        
        for idx, fpath in enumerate(upload_files):
            df_upload = load_df(fpath)
            if df_upload.empty: continue
            
            # Đồng bộ tên cột target
            for tc in TARGET_COLS:
                for c in df_upload.columns:
                    if tc.replace(" ", "").lower() == str(c).replace(" ", "").lower():
                        df_upload = df_upload.rename(columns={c: tc})
            
            avail_tgt = [c for c in TARGET_COLS if c in df_upload.columns]
            if not avail_tgt: continue

            base_dates = set(base[DATE_COL].dt.strftime('%Y-%m-%d'))
            new_rows = df_upload[~df_upload[DATE_COL].dt.strftime('%Y-%m-%d').isin(base_dates)].copy()
            new_rows = new_rows[new_rows[DATE_COL] >= start_date]
            
            if not new_rows.empty:
                base_for_pred = pd.concat([base_full[base_full[DATE_COL] < base[DATE_COL].min()], base], ignore_index=True)
                base_for_pred = base_for_pred.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).tail(500)
                base_enriched = enrich_with_exo(base_for_pred, base_full)

                for mname in MODEL_DEFS:
                    task_idx += 1
                    try:
                        match_data = []
                        # CHẾ ĐỘ SIÊU NHẸ: 3 điểm kiểm tra mỗi file
                        indices = np.unique(np.linspace(0, len(new_rows) - 1, 3, dtype=int))
                        indices = [i for i in indices if 0 <= i < len(new_rows)]
                        
                        import gc
                        for step_i, idx_in_new in enumerate(indices):
                            status_text.text(f"⏳ {mname} | File {idx+1} | Điểm {step_i+1}/{len(indices)}")
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            
                            history_df = pd.concat([base_enriched, new_rows.iloc[:idx_in_new]], ignore_index=True)
                            actual_pool = new_rows.iloc[idx_in_new:idx_in_new+1].copy()
                            if actual_pool.empty: continue
                            
                            for h in HORIZONS:
                                # Nạp mô hình chuyên biệt cho đúng mốc h
                                model_h, meta_h, device_h = load_model(mname, h)
                                if not model_h: continue
                                _swap_src(MODEL_DEFS[mname]["proj_dir"])
                                
                                missing = [c for c in meta_h["feature_cols"] if c not in history_df.columns]
                                hist_filled = history_df.copy()
                                if missing:
                                    hist_filled = generate_time_features(hist_filled)
                                    for mc in missing:
                                        if mc in base_full.columns:
                                            hist_filled[mc] = base_full.set_index(DATE_COL).reindex(hist_filled[DATE_COL])[mc].values
                                hist_filled = hist_filled.ffill().bfill().fillna(0)
                                
                                pred_df = predict_from_df(model_h, meta_h, hist_filled, device_h)
                                if pred_df.empty: continue
                                
                                idx_h = min(h - 1, len(pred_df) - 1)
                                p_row = pred_df.iloc[idx_h]
                                
                                diff = (actual_pool[DATE_COL] - p_row[DATE_COL]).dt.days.abs()
                                if not diff.empty and diff.min() == 0:
                                    a_row = actual_pool.loc[diff.idxmin()]
                                    for tgt in avail_tgt:
                                        if tgt in p_row and tgt in a_row.index and pd.notna(a_row[tgt]):
                                            match_data.append({
                                                "Model": mname, "Horizon": f"{h}d",
                                                "Upload": f"#{idx+1} {fpath.name}",
                                                DATE_COL: a_row[DATE_COL], "Target": tgt,
                                                "Dự báo": round(float(p_row[tgt]), 2),
                                                "Thực tế": round(float(a_row[tgt]), 2),
                                                "Sai lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])), 2),
                                                "% Lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])) / (abs(float(a_row[tgt])) + 1e-8) * 100, 2),
                                            })
                        
                        if match_data:
                            res_df = pd.DataFrame(match_data).drop_duplicates(subset=["Model", "Horizon", DATE_COL, "Target"])
                            all_records.extend(res_df.to_dict("records"))
                    except Exception as e: 
                        logf.write(f"EXCEPTION: {mname}: {e}\n")
                        continue
                
                base = pd.concat([base, new_rows], ignore_index=True)
                base = base.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

        status_text.empty()

    return pd.DataFrame(all_records)

def show_live_forecasts(base_full, file_paths, sel_models):
    """Tính toán và hiển thị dự báo đa mốc thời gian dựa trên dữ liệu mới nhất."""
    if not file_paths:
        st.warning("⚠️ Chưa có file upload để lấy dữ liệu mới nhất.")
        return

    # Lấy ngày lớn nhất từ các file upload
    upload_dfs = []
    for fp in file_paths:
        upload_dfs.append(load_df(fp))
    
    upload_max_date = pd.concat(upload_dfs)[DATE_COL].max() if upload_dfs else base_full[DATE_COL].max()

    # Tổng hợp dữ liệu mới nhất
    latest_df = base_full.copy()
    for fdf in upload_dfs:
        latest_df = pd.concat([latest_df, fdf])
        
    latest_df = latest_df.drop_duplicates(DATE_COL).sort_values(DATE_COL).reset_index(drop=True)
    
    # CẮT DỮ LIỆU ĐẾN NGÀY UPLOAD CUỐI CÙNG (Để dự báo từ mốc file upload)
    latest_df = latest_df[latest_df[DATE_COL] <= upload_max_date]
    
    history = latest_df.tail(500)
    last_date = history[DATE_COL].iloc[-1]
    
    # Tìm xem ngày này thuộc file nào để báo cho người dùng
    source_file = "Dataset gốc"
    for f in file_info:
        if f["max_date"] == last_date:
            source_file = f["name"]
            break
            
    st.markdown(f"### 🔮 Bảng dự báo (Từ mốc: **{last_date.strftime('%d/%m/%Y')}**)")
    st.caption(f"📌 Nguồn dữ liệu mốc: **{source_file}**")

    
    tabs = st.tabs(sel_models)
    for idx, mname in enumerate(sel_models):
        with tabs[idx]:
            all_preds = []
            future_points = [] # Dùng cho biểu đồ xu hướng
            detailed_preds = {} # Lưu dự báo chi tiết
            
            # Lấy giá hiện tại làm mốc 0
            last_prices = history.iloc[-1]
            future_points = []
            for tgt in TARGET_COLS:
                if tgt in last_prices:
                    future_points.append({DATE_COL: last_date, "Target": tgt, "Giá": float(last_prices[tgt]), "Loại": "Hiện tại"})

            # Chuẩn bị dữ liệu lịch sử một lần duy nhất
            history_enriched = generate_time_features(history.copy())

            # Chạy dự báo cho từng mốc với model chuyên biệt
            all_preds = []
            pred_cache = {}  # Lưu kết quả để vẽ biểu đồ

            for h in HORIZONS:
                try:
                    model_h, meta_h, device_h = load_model(mname, h)
                    if not model_h:
                        continue
                    _swap_src(MODEL_DEFS[mname]["proj_dir"])

                    # Bổ sung cột còn thiếu
                    hist_h = history_enriched.copy()
                    missing = [c for c in meta_h["feature_cols"] if c not in hist_h.columns]
                    for mc in missing:
                        if mc in base_full.columns:
                            hist_h[mc] = base_full.set_index(DATE_COL).reindex(hist_h[DATE_COL])[mc].values
                    hist_h = hist_h.ffill().bfill().fillna(0)

                    pred_df = predict_from_df(model_h, meta_h, hist_h, device_h)
                    if pred_df.empty:
                        continue

                    # Lấy dòng cuối cùng là dự báo tại mốc h ngày
                    p_row = pred_df.iloc[-1]
                    f_date = last_date + pd.Timedelta(days=h)
                    if DATE_COL in pred_df.columns:
                        f_date = p_row[DATE_COL]

                    row_ui = {"Ngày dự đoán": f_date.strftime('%d/%m/%Y'), "Mốc": f"+{h} ngày"}
                    for tgt in TARGET_COLS:
                        if tgt in p_row:
                            val = float(p_row[tgt])
                            row_ui[tgt] = f"{val:,.2f}"
                            future_points.append({DATE_COL: f_date, "Target": tgt, "Giá": val, "Loại": "Dự báo"})

                    all_preds.append(row_ui)

                    pred_cache[h] = (pred_df, f_date)

                except Exception as e:
                    # Quay lại dùng caption để không làm rối giao diện nếu mốc đó chưa có model
                    st.caption(f"ℹ️ Mốc {h}d: {e}")
                    continue



            # Hiển thị bảng tóm tắt
            if all_preds:
                st.markdown("#### 📋 Tóm tắt dự báo các mốc")
                safe_dataframe(pd.DataFrame(all_preds).set_index("Ngày dự đoán"))

            # Vẽ biểu đồ lộ trình cho từng mặt hàng
            if pred_cache:
                st.markdown(f"**📈 Biểu đồ so sánh các chân trời dự báo ({mname})**")
                for tgt in TARGET_COLS:
                    fig = go.Figure()
                    last_val = float(last_prices[tgt]) if tgt in last_prices else 0

                    for h, (pred_df, f_date) in pred_cache.items():
                        if tgt not in pred_df.columns:
                            continue
                        vals = pred_df[tgt].values[:h]
                        if DATE_COL in pred_df.columns:
                            dates = list(pred_df[DATE_COL].values[:h])
                        else:
                            dates = [last_date + pd.Timedelta(days=d) for d in range(1, h + 1)]

                        all_dates = [last_date] + list(dates)
                        all_vals  = [last_val]  + list(vals)

                        dash  = "solid" if h == max(HORIZONS) else "dot"
                        width = 2.5    if h == max(HORIZONS) else 1.5
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(all_dates), y=all_vals,
                            name=f"Dự báo {h}d", mode="lines",
                            line=dict(dash=dash, width=width)
                        ))


                    fig.update_layout(
                        title=f"So sánh lộ trình dự báo: {tgt}",
                        template="plotly_dark", height=350, hovermode="x unified",
                        xaxis=dict(type='date', tickformat='%d/%m/%Y')
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{mname}_{tgt}")


    st.markdown("---")



# === UI ===

st.set_page_config(page_title="Oil Forecast Hub", layout="wide", page_icon="🛢️")
st.markdown("<style>.block-container { padding-top: 1rem; } h1 { background: linear-gradient(135deg, #00d4aa, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }</style>", unsafe_allow_html=True)
st.title("🛢️ Oil Forecast – Automated Evaluation Hub")


# Lưu base_full gốc để dùng cho live forecast
base_full_orig = load_df(BUILTIN_CSV)

CACHE_FILE = ROOT / "simulation_cache.pkl"

def get_dir_fingerprint():
    data_dir = ROOT / "datasets"
    files = list(data_dir.glob("*"))
    if not files: return "empty"
    return f"{len(files)}_{max(f.stat().st_mtime for f in files)}"

fingerprint = get_dir_fingerprint()

@st.cache_data
def get_sorted_files(fp):
    data_dir = ROOT / "datasets"
    files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"]]
    info = []
    for f in files:
        df = load_df(f)
        if not df.empty: info.append({"path": str(f), "max_date": df[DATE_COL].max(), "name": f.name, "rows": len(df)})
    info.sort(key=lambda x: x["max_date"])
    return info

file_info = get_sorted_files(fingerprint)
file_paths = [Path(i["path"]) for i in file_info]

# Load cache
if CACHE_FILE.exists():
    try:
        cache = pd.read_pickle(CACHE_FILE)
        if cache.get("fp") == fingerprint: combined = cache.get("df")
        else: combined = None
    except: combined = None
else: combined = None

if combined is None:
    # Không tự động chạy simulation ở đây nữa để tránh làm chậm việc upload/finetune
    combined = pd.DataFrame()


# Sidebar
st.sidebar.header("⚙️ Cấu hình")

# --- GPU Status ---
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
device_color = "green" if torch.cuda.is_available() else "orange"
st.sidebar.markdown(f"**Thiết bị:** :{device_color}[{device_name}]")
if torch.cuda.is_available():
    st.sidebar.caption(f"CUDA Version: {torch.version.cuda}")
# ------------------

st.sidebar.markdown("**Mô hình sử dụng:** :blue[GUMNet]")
sel_models = ["GUMNet"]

st.sidebar.markdown("---")
# Đã ẩn các nút Cache và Lịch sử theo yêu cầu của bạn


if not combined.empty:
    if "Mặt hàng" in combined.columns:
        combined = combined.rename(columns={"Mặt hàng": "Target"})
    # Chỉ hiển thị dữ liệu từ ngày bắt đầu simulation (Sep 2025) để tránh hiện lịch sử quá cũ
    df_view = combined[(combined["Model"].isin(sel_models)) & (combined[DATE_COL] >= CUTOFF_DATE)]
else: 
    df_view = pd.DataFrame()


# Tabs
t1, t2, t3, t4, t5 = st.tabs(["⬆️ Upload & Dự báo", "🏆 Tổng kết", "📋 Lịch sử upload", "📈 Biểu đồ", "🗃️ Bảng chi tiết"])

with t1:
    # Kiểm tra xem đã có checkpoint chưa
    ckpts_exist = any((CKPT_DIR / f"gumnet_h{h}.pt").exists() for h in HORIZONS)
    
    if ckpts_exist:
        # Hiển thị dự báo tương lai dựa trên mô hình đang có
        show_live_forecasts(base_full_orig, file_paths, sel_models)
    else:
        st.info("ℹ️ **Chưa có mô hình nào được huấn luyện.**\n\n"
                "Hãy chạy `CHAY_UNG_DUNG.bat` để tự động huấn luyện lần đầu,\n"
                "hoặc upload file dữ liệu và nhấn **Tự động huấn luyện** bên dưới.")


    st.subheader("⬆️ Upload file mới")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        auto_ft = st.checkbox("🔄 Tự động huấn luyện sau khi tải file", value=True)
    with c2:
        sel_hz_auto = st.multiselect("Chọn mốc", HORIZONS, default=HORIZONS)

    with c3:
        train_mode = st.radio("Chế độ", ["⚡ Finetune", "🔁 Train lại từ đầu"], index=0,
                              help="Finetune: học tiếp từ checkpoint cũ (nhanh).\nTrain lại từ đầu: xóa checkpoint cũ, học toàn bộ dữ liệu (chính xác hơn nhưng lâu hơn).")
    force_retrain = (train_mode == "🔁 Train lại từ đầu")

    up = st.file_uploader("Chọn file Excel/CSV", type=["xlsx", "xls", "csv"])
    if up:
        tmp = ROOT / "datasets" / up.name
        with open(tmp, "wb") as f: f.write(up.getbuffer())
        st.success(f"✅ Đã lưu: {up.name}")
        df_new = load_df(tmp)
        if not df_new.empty:
            m_date = df_new[DATE_COL].max()
            st.session_state["last_up_date"] = m_date
            st.info(f"📅 Dữ liệu trong file mới nhất đến: {m_date.strftime('%d/%m/%Y')}")

        # Chỉ train nếu file này chưa được train lần này (tránh lặp vô tận)
        # Nếu chọn "Train lại từ đầu" thì luôn cho phép train
        already_trained = (st.session_state.get("last_trained_file") == up.name) and not force_retrain
            
        if auto_ft and not already_trained:
            if not sel_models:
                st.error("❌ Vui lòng chọn mô hình ở Sidebar!")
            elif not sel_hz_auto:
                st.error("❌ Vui lòng chọn ít nhất một mốc thời gian!")
            else:
                mode_label = "🔁 TRAIN LẠI TỪ ĐẦU" if force_retrain else "⚡ Finetune"
                st.warning(f"⏳ {mode_label} — Mốc: {', '.join([str(x)+'d' for x in sel_hz_auto])}...")
                log_area = st.empty()
                import subprocess
                cmd = [sys.executable, "train_all_horizons.py", "--update_data", "--new_file", str(tmp),
                       "--epochs", "50", "--models"] + sel_models + ["--horizons"] + [str(x) for x in sel_hz_auto]
                if force_retrain:
                    cmd.append("--force_retrain")
                try:

                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
                    log_text = ""
                    for line in process.stdout:
                        log_text += line
                        log_area.code("\n".join(log_text.splitlines()[-15:]))
                    process.wait()
                    if process.returncode == 0:
                        st.session_state["last_trained_file"] = up.name
                        st.cache_resource.clear()
                        st.cache_data.clear()
                        log_area.empty() # Ẩn Log sau khi xong
                        st.success(f"✅ HOÀN TẤT! Đã huấn luyện thành công các mốc: {', '.join([str(x)+'d' for x in sel_hz_auto])}")


                        # Hiển thị dự báo ngay sau khi train xong
                        st.markdown("---")
                        last_upload_date = df_new[DATE_COL].max()
                        st.markdown(f"### 🔮 Dự báo chi tiết sau huấn luyện (từ {last_upload_date.strftime('%d/%m/%Y')})")

                        # Chuẩn bị lịch sử từ file upload
                        hist_for_pred = pd.concat([base_full_orig, df_new]).drop_duplicates(DATE_COL).sort_values(DATE_COL).tail(500)
                        hist_for_pred = generate_time_features(hist_for_pred)

                        for mname in sel_models:
                            st.markdown(f"**🤖 {mname}**")
                            # Tạo các tab cho từng mốc dự báo
                            tabs_h = st.tabs([f"Mốc {h} ngày" for h in sel_hz_auto])
                            
                            for idx_h, h_display in enumerate(sel_hz_auto):
                                with tabs_h[idx_h]:
                                    try:
                                        m_h, meta_h, dev_h = load_model(mname, h_display)
                                        if not m_h:
                                            st.caption(f"⚠️ Chưa có model {mname} h{h_display}. Hãy train trước.")
                                            continue
                                        _swap_src(MODEL_DEFS[mname]["proj_dir"])
                                        hist_h = hist_for_pred.copy()
                                        missing = [c for c in meta_h["feature_cols"] if c not in hist_h.columns]
                                        for mc in missing:
                                            if mc in base_full_orig.columns:
                                                hist_h[mc] = base_full_orig.set_index(DATE_COL).reindex(hist_h[DATE_COL])[mc].values
                                        hist_h = hist_h.ffill().bfill().fillna(0)

                                        pred_df = predict_from_df(m_h, meta_h, hist_h, dev_h)
                                        if pred_df.empty:
                                            st.caption("⚠️ Model không trả về kết quả.")
                                            continue

                                        # Hiển thị từng ngày trong lộ trình dự báo
                                        rows = []
                                        for i, row in pred_df.iterrows():
                                            if DATE_COL in pred_df.columns:
                                                ngay = pd.Timestamp(row[DATE_COL]).strftime('%d/%m/%Y (%a)')
                                            else:
                                                ngay = (last_upload_date + pd.Timedelta(days=i+1)).strftime('%d/%m/%Y (%a)')
                                            
                                            r = {"📅 Ngày": ngay}
                                            for tgt in TARGET_COLS:
                                                if tgt in row:
                                                    r[tgt] = f"{float(row[tgt]):,.2f}"
                                            rows.append(r)

                                        if rows:
                                            safe_dataframe(pd.DataFrame(rows).set_index("📅 Ngày"))

                                    except Exception as ex:
                                        st.error(f"Lỗi dự báo {mname} mốc {h_display} ngày: {ex}")


                        st.markdown("---")
                        if st.button("🔄 Tải lại Dashboard", key="reload_after_train"):
                            st.rerun()

                    else:
                        st.error("❌ Có lỗi xảy ra.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        elif already_trained:
            st.info("ℹ️ File này đã được học. Upload file mới để cập nhật thêm.")





with t2:
    # Nếu chưa có kết quả simulation, cho phép chạy tại đây
    if df_view.empty and file_paths:
        st.info("📊 Bạn vừa upload dữ liệu mới hoặc đổi mô hình. Nhấn nút bên dưới để cập nhật bảng đánh giá độ chính xác.")
        if st.button("📈 Chạy Phân tích & Đánh giá (MAE/MAPE)"):
            with st.spinner("⏳ Hệ thống đang chạy mô phỏng dự báo quá khứ để tính độ lệch..."):
                combined = run_upload_simulation(str(BUILTIN_CSV), file_paths, CUTOFF_DATE)
                pd.to_pickle({"fp": fingerprint, "df": combined}, CACHE_FILE)
                st.rerun()

    if not df_view.empty:
        h_order = [f"{h}d" for h in HORIZONS]
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 MAPE (%) theo Mốc")
            mape_h = df_view.groupby(["Model", "Horizon"])["% Lệch"].mean().unstack().round(2)
            mape_h = mape_h[[c for c in h_order if c in mape_h.columns]]
            safe_dataframe(mape_h.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)
        with col_b:
            st.subheader("🛢️ MAPE (%) theo Mặt hàng")
            mape_t = df_view.groupby(["Model", "Target"])["% Lệch"].mean().unstack().round(2)
            safe_dataframe(mape_t.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

        st.subheader("📋 Chi tiết sai lệch MAE")
        mae_piv = df_view.groupby(["Model", "Horizon"])["Sai lệch"].mean().unstack().round(2)
        mae_piv = mae_piv[[c for c in h_order if c in mae_piv.columns]]
        st.dataframe(mae_piv, use_container_width=True)
        
        # HỆ THỐNG CẢNH BÁO THÔNG MINH
        avg_mape = df_view["% Lệch"].mean()

        if avg_mape > 10.0:
            st.error(f"### ⚠️ CẢNH BÁO ĐỘ CHÍNH XÁC (MAPE: {avg_mape:.2f}%)")
            st.markdown(f"""
            Mức độ sai lệch hiện tại đã vượt ngưỡng an toàn (10%). Mô hình có dấu hiệu bị lỗi thời so với dữ liệu mới.
            
            **💡 Khuyến nghị:** Bạn nên thực hiện **Finetune (Huấn luyện lại)** mô hình với các dữ liệu thực tế vừa upload để cập nhật quy luật giá mới nhất.
            """)
        elif avg_mape > 7.0:
            st.warning(f"### 🔔 Nhắc nhở: Độ chính xác đang giảm nhẹ (MAPE: {avg_mape:.2f}%)")
        else:
            st.success(f"### ✅ Mô hình hoạt động tốt (MAPE: {avg_mape:.2f}%)")

        with st.expander("🛠️ Hướng dẫn & Nút Finetune (Huấn luyện lại)"):
            st.markdown("""
            Để cập nhật 'bộ não' cho AI bằng các dữ liệu mới bạn vừa upload, hãy nhấn nút bên dưới.
            
            **⚠️ Lưu ý:** 
            - Quá trình này sẽ huấn luyện lại **12 mô hình** (6 chân trời x 2 kiến trúc).
            - Thời gian xử lý: **5 - 15 phút**. Vui lòng không tắt ứng dụng trong lúc chạy.
            """)
            
            n_epochs = st.number_input("Số vòng lặp (Epochs) huấn luyện thêm", min_value=1, max_value=200, value=50)
            sel_hz_manual = st.multiselect("Chọn các mốc thời gian cần cập nhật", HORIZONS, default=[1, 5, 10, 15, 20, 30])
            
            if st.button("🚀 Bắt đầu Finetune Ngay"):
                if not sel_models:
                    st.error("❌ Vui lòng chọn ít nhất một mô hình ở thanh bên trái!")
                elif not sel_hz_manual:
                    st.error("❌ Vui lòng chọn ít nhất một mốc thời gian!")
                else:
                    log_area = st.empty()
                    log_text = f"🛠️ Đang chuẩn bị môi trường cho {', '.join(sel_models)} (Mốc: {', '.join([str(x)+'d' for x in sel_hz_manual])})...\n"
                    log_area.code(log_text)
                    
                    import subprocess
                    cmd = [sys.executable, "train_all_horizons.py", "--update_data", "--epochs", str(n_epochs), "--models"] + sel_models + ["--horizons"] + [str(x) for x in sel_hz_manual]
                    
                    try:
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
                        
                        for line in process.stdout:
                            log_text += line
                            # Chỉ lấy 20 dòng cuối để UI không bị quá dài
                            display_text = "\n".join(log_text.splitlines()[-20:])
                            log_area.code(display_text)
                        
                        process.wait()
                        if process.returncode == 0:
                            st.success("✅ HUẤN LUYỆN LẠI THÀNH CÔNG! Hãy nhấn 'Xóa Cache Simulation' để cập nhật kết quả.")
                        else:
                            st.error(f"❌ Có lỗi xảy ra trong quá trình huấn luyện (Code: {process.returncode})")
                    except Exception as e:
                        st.error(f"❌ Không thể khởi chạy tiến trình huấn luyện: {e}")

    else: st.info("Chưa có dữ liệu so sánh.")

with t3:
    if not df_view.empty:
        sel_up = st.selectbox("Chọn đợt upload", df_view["Upload"].unique())
        sub = df_view[df_view["Upload"] == sel_up]
        cols = [c for c in sub.columns if c != "Ngày thứ"]
        safe_dataframe(sub[cols].style.format({"Dự báo":"{:.2f}","Thực tế":"{:.2f}","Sai lệch":"{:.2f}","% Lệch":"{:.2f}%"}), use_container_width=True)
    else: st.info("Trống.")

with t4:
    if not df_view.empty:
        sh = st.selectbox("Chọn horizon", [f"{h}d" for h in HORIZONS])
        sub = df_view[df_view["Horizon"] == sh]
        for tgt in TARGET_COLS:
            tsub = sub[sub["Target"] == tgt]
            if tsub.empty: continue
            fig = go.Figure()
            for m in sel_models:
                ms = tsub[tsub["Model"] == m].sort_values(DATE_COL)
                if not ms.empty: fig.add_trace(go.Scatter(x=ms[DATE_COL], y=ms["Dự báo"], name=m, mode="lines+markers", line=dict(dash="dash")))
            act = tsub.drop_duplicates(DATE_COL).sort_values(DATE_COL)
            fig.add_trace(go.Scatter(x=act[DATE_COL], y=act["Thực tế"], name="Thực tế", mode="lines+markers", line=dict(color="#00d4aa", width=3)))
            fig.update_layout(title=f"{tgt} - {sh}", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

with t5:
    if not df_view.empty:
        cols = [c for c in df_view.columns if c != "Ngày thứ"]
        safe_dataframe(df_view[cols], use_container_width=True)
