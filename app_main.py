"""
Multi-Model Oil Price Forecast – Evaluation Hub (Robust Version)
"""

import sys, json, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# ─── Cấu hình để vượt qua lỗi bảo mật DLL trên một số máy (Application Control Policy) ───
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

# ═══════════════════════════  CONFIG  ═════════════════════════════════════════
ROOT = Path(__file__).resolve().parent
BUILTIN_CSV = ROOT / "oil_forecast_research_new-main" / "data" / "processed" / "clean_data_exo_ver1.csv"
CKPT_DIR = ROOT / "checkpoints_multi"

TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
DATE_COL = "Ngày"
HORIZONS = [1, 5, 10, 30, 60, 100]
CUTOFF_DATE = pd.Timestamp("2025-09-20")

MODEL_DEFS = {
    "GUMNet": {
        "proj_dir": ROOT / "oil_forecast_research_new-main",
        "mod": "src.model.model", "cls": "GUMNet", "kind": "quantile",
    },
    "HybridTriNet": {
        "proj_dir": ROOT / "Hybridtrinet_Oil",
        "mod": "src.model.hybrid_trinet", "cls": "HybridTriNet", "kind": "point",
    },
}

# ═══════════════════════════  DATA HELPERS  ═══════════════════════════════════

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

# ═══════════════════════════  MODEL LOADING  ══════════════════════════════════

def _swap_src(proj_dir):
    d = str(proj_dir)
    sys.path = [p for p in sys.path if p != d]
    sys.path.insert(0, d)
    for m in [k for k in list(sys.modules) if k.startswith("src")]: del sys.modules[m]

@st.cache_resource
def load_model(model_name, horizon):
    try:
        cfg = MODEL_DEFS[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _swap_src(cfg["proj_dir"])
        mod = importlib.import_module(cfg["mod"])
        cls = getattr(mod, cfg["cls"])

        if model_name == "GUMNet":
            ckpt_path = CKPT_DIR / f"gumnet_h{horizon}.pt"
            if not ckpt_path.exists(): raise FileNotFoundError(f"Missing {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = cls(
                seq_len=ckpt["seq_len"], input_dim=ckpt["input_dim"],
                output_dim=ckpt["output_dim"], horizon=ckpt["horizon"],
                d_feat=ckpt.get("d_feat", 64), num_quantiles=ckpt["num_quantiles"],
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            meta = {
                "feature_cols": [c.strip() for c in ckpt["feature_cols"]],
                "target_cols": [c.strip() for c in ckpt["target_cols"]],
                "seq_len": ckpt["seq_len"], "horizon": ckpt["horizon"], "kind": "quantile",
                "feature_scaler": ckpt["feature_scaler"], "target_scaler": ckpt["target_scaler"],
            }
        else:
            ckpt_path = CKPT_DIR / f"hybrid_h{horizon}.pt"
            meta_dir = CKPT_DIR / f"hybrid_h{horizon}_meta"
            if not ckpt_path.exists(): raise FileNotFoundError(f"Missing {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            with open(meta_dir / "feature_cols.json") as f: fj = json.load(f)
            f_cols = [c.strip() for c in fj["feature_cols"]]
            model = cls(
                k=fj["K"], H=horizon, D_in=len(f_cols), D_out=len(TARGET_COLS),
                d_feat=96, kan_M=8, kan_depth=2, gru_hidden=128, gru_layers=1,
                attn_dmodel=64, attn_heads=4, attn_layers=2, patch_len=16, stride=8,
            ).to(device)
            model.load_state_dict(ckpt)
            meta = {
                "feature_cols": f_cols, "target_cols": TARGET_COLS, "seq_len": fj["K"],
                "horizon": horizon, "kind": "point",
                "x_mu": np.load(meta_dir / "x_mu.npy"), "x_sd": np.load(meta_dir / "x_sd.npy"),
                "y_mu": np.load(meta_dir / "y_mu.npy"), "y_sd": np.load(meta_dir / "y_sd.npy"),
            }
        model.eval()
        return model, meta, device
    except Exception as e:
        # Trả về lỗi để UI hiển thị thay vì nuốt chửng
        return None, str(e), None

def predict_from_df(model, meta, df, device):
    k, f_cols, t_cols = meta["seq_len"], meta["feature_cols"], meta["target_cols"]
    X = df[f_cols].values
    if "feature_scaler" in meta: X = meta["feature_scaler"].transform(X)
    else: X = (X - meta["x_mu"]) / (meta["x_sd"] + 1e-8)
    x_in = torch.tensor(X[-k:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad(): out, _ = model(x_in)
    if meta["kind"] == "quantile":
        p = np.sort(out.cpu().numpy()[0], axis=-1)
        p50 = meta["target_scaler"].inverse_transform(p[..., 1])
    else:
        raw = out.cpu().numpy()[0]
        flat = raw.reshape(-1, len(t_cols))
        p50 = flat * (meta["y_sd"] + 1e-8) + meta["y_mu"]
    h = p50.shape[0]
    last = df[DATE_COL].iloc[-1]
    dates, d = [], last
    while len(dates) < h:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5: dates.append(d)
    result = pd.DataFrame(p50[:len(dates)], columns=t_cols)
    result.insert(0, DATE_COL, [pd.Timestamp(dt).normalize() for dt in dates[:len(result)]])
    return result

# ═══════════════════  SIMULATION ENGINE  ══════════════════════════════════════

def run_upload_simulation(base_path, upload_files, start_date):
    base_full = load_df(base_path)
    base = base_full[base_full[DATE_COL] < start_date].copy()
    all_records = []
    
    with open("d:/Anh_Thuy/sim_log.txt", "w", encoding="utf-8") as logf:
        logf.write(f"Simulation started. Files: {len(upload_files)}\n")
        
        total_tasks = len(upload_files) * len(MODEL_DEFS) * len(HORIZONS)
        task_idx = 0
        prog = st.progress(0)
        
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
            # CHỈ lấy dữ liệu từ ngày cutoff trở đi để tập trung vào 15 file đánh giá
            new_rows = new_rows[new_rows[DATE_COL] >= start_date]
            
            if not new_rows.empty:
                base_for_pred = pd.concat([base_full[base_full[DATE_COL] < base[DATE_COL].min()], base], ignore_index=True)
                base_for_pred = base_for_pred.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).tail(500)
                base_enriched = enrich_with_exo(base_for_pred, base_full)

                for mname in MODEL_DEFS:
                    for h in HORIZONS:
                        task_idx += 1
                        prog.progress(task_idx / total_tasks, text=f"⏳ {mname} h={h} | File {idx+1}/{len(upload_files)}")
                        try:
                            model, meta, device = load_model(mname, h)
                            if not model: 
                                logf.write(f"ERROR: {mname} h={h} failed to load: {meta}\n")
                                continue
                            
                            _swap_src(MODEL_DEFS[mname]["proj_dir"])
                            match_data = []
                            
                            # Tối ưu mật độ dự báo
                            if len(new_rows) < 30:
                                indices = np.arange(len(new_rows))
                            else:
                                # Lấy 5 điểm rải đều + toàn bộ 15 ngày cuối cùng
                                indices = np.unique(np.concatenate([
                                    np.linspace(0, len(new_rows) - 1, 5, dtype=int),
                                    np.arange(len(new_rows) - 15, len(new_rows))
                                ]))
                                indices = [i for i in indices if i >= 0 and i < len(new_rows)]
                            
                            for idx_in_new in indices:
                                history_df = pd.concat([base_enriched, new_rows.iloc[:idx_in_new]], ignore_index=True)
                                missing = [c for c in meta["feature_cols"] if c not in history_df.columns]
                                if missing:
                                    history_df = generate_time_features(history_df)
                                    for mc in missing:
                                        if mc in base_full.columns:
                                            history_df[mc] = base_full.set_index(DATE_COL).reindex(history_df[DATE_COL])[mc].values
                                
                                # Đảm bảo không có NaN
                                history_df = history_df.ffill().bfill().fillna(0)
                                
                                pred = predict_from_df(model, meta, history_df, device)
                                logf.write(f"DEBUG: {mname} h={h} File {idx+1} Point {idx_in_new} -> {len(pred)} pred rows\n")
                                
                                actual_pool = df_upload[df_upload[DATE_COL] > history_df[DATE_COL].iloc[-1]].copy()
                                if actual_pool.empty: 
                                    logf.write(f"DEBUG: actual_pool empty for end={history_df[DATE_COL].iloc[-1]}\n")
                                    continue

                                if pred[meta["target_cols"]].isna().any().any():
                                    logf.write(f"WARNING: NaNs in prediction {mname} h={h} File {idx+1}\n")
                                    continue

                                for _, p_row in pred.iterrows():
                                    diff = (actual_pool[DATE_COL] - p_row[DATE_COL]).dt.days.abs()
                                    if not diff.empty and diff.min() <= 3:
                                        a_row = actual_pool.loc[diff.idxmin()]
                                        day_idx = pred[pred[DATE_COL] == p_row[DATE_COL]].index[0] + 1
                                        logf.write(f"   MATCH: Pred {p_row[DATE_COL].date()} vs Actual {a_row[DATE_COL].date()} (diff={diff.min()})\n")
                                        for tgt in avail_tgt:
                                            if tgt in p_row and tgt in a_row and pd.notna(a_row[tgt]):
                                                match_data.append({
                                                    "Model": mname, "Horizon": f"{h}d",
                                                    "Upload": f"#{idx+1} {fpath.name}", "Ngày thứ": day_idx,
                                                    DATE_COL: a_row[DATE_COL], "Target": tgt,
                                                    "Dự báo": round(float(p_row[tgt]), 2), "Thực tế": round(float(a_row[tgt]), 2),
                                                    "Sai lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])), 2),
                                                    "% Lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])) / (abs(float(a_row[tgt])) + 1e-8) * 100, 2),
                                                })
                            
                            if match_data:
                                logf.write(f"SUCCESS: File {idx+1} {h}d - Found {len(match_data)} matches\n")
                                res_df = pd.DataFrame(match_data).drop_duplicates(subset=["Model", "Horizon", DATE_COL, "Target"])
                                all_records.extend(res_df.to_dict("records"))
                        except Exception as e: 
                            logf.write(f"EXCEPTION: {mname} h={h}: {e}\n")
                            continue
                
                base = pd.concat([base, new_rows], ignore_index=True)
                base = base.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

        prog.empty()
    return pd.DataFrame(all_records)

def show_live_forecasts(base_full, file_paths, sel_models):
    """Tính toán và hiển thị dự báo đa mốc thời gian dựa trên dữ liệu mới nhất."""
    if not file_paths:
        st.warning("⚠️ Chưa có file upload để lấy dữ liệu mới nhất.")
        return

    # Tổng hợp dữ liệu mới nhất
    latest_df = base_full.copy()
    for fp in file_paths:
        fdf = load_df(fp)
        latest_df = pd.concat([latest_df, fdf])
    latest_df = latest_df.drop_duplicates(DATE_COL).sort_values(DATE_COL).reset_index(drop=True)
    
    history = latest_df.tail(500)
    last_date = history[DATE_COL].iloc[-1]
    
    st.markdown(f"### 🔮 Bảng dự báo đa mốc thời gian (Từ mốc: **{last_date.strftime('%d/%m/%Y')}**)")
    
    tabs = st.tabs(sel_models)
    for idx, mname in enumerate(sel_models):
        with tabs[idx]:
            all_preds = []
            future_points = [] # Dùng cho biểu đồ xu hướng
            
            # Lấy giá hiện tại làm mốc 0
            last_prices = history.iloc[-1]
            for tgt in TARGET_COLS:
                if tgt in last_prices:
                    future_points.append({DATE_COL: last_date, "Target": tgt, "Giá": float(last_prices[tgt]), "Loại": "Hiện tại"})

            for h in HORIZONS:
                try:
                    model, meta, device = load_model(mname, h)
                    if model:
                        _swap_src(MODEL_DEFS[mname]["proj_dir"])
                        history_enriched = generate_time_features(history)
                        missing = [c for c in meta["feature_cols"] if c not in history_enriched.columns]
                        for mc in missing:
                            if mc in base_full.columns:
                                history_enriched[mc] = base_full.set_index(DATE_COL).reindex(history_enriched[DATE_COL])[mc].values
                        
                        history_enriched = history_enriched.ffill().bfill().fillna(0)
                        pred = predict_from_df(model, meta, history_enriched, device)
                        
                        if not pred.empty:
                            f_date = last_date + pd.Timedelta(days=h)
                            row = {
                                "Ngày dự đoán": f_date.strftime('%d/%m/%Y'),
                                "Mốc (Horizon)": f"+{h} ngày"
                            }
                            for tgt in TARGET_COLS:
                                val = float(pred.iloc[0][tgt])
                                row[tgt] = f"{val:,.0f}"
                                # Thêm điểm vào biểu đồ (chỉ lấy điểm cuối của horizon)
                                future_points.append({DATE_COL: f_date, "Target": tgt, "Giá": val, "Loại": "Dự báo"})
                            all_preds.append(row)
                except: continue
            
            if all_preds:
                safe_dataframe(pd.DataFrame(all_preds).set_index("Ngày dự đoán"))
                
            # Vẽ biểu đồ so sánh TẤT CẢ lộ trình
            st.markdown(f"**📈 Biểu đồ so sánh các chân trời dự báo ({mname})**")
            for tgt in TARGET_COLS:
                fig = go.Figure()
                # Thêm điểm bắt đầu (giá hiện tại)
                last_val = float(last_prices[tgt])
                
                # Duyệt qua tất cả các horizon để vẽ lộ trình của từng cái
                for h in HORIZONS:
                    try:
                        model_h, meta_h, device_h = load_model(mname, h)
                        if model_h:
                            _swap_src(MODEL_DEFS[mname]["proj_dir"])
                            history_enriched = generate_time_features(history)
                            missing = [c for c in meta_h["feature_cols"] if c not in history_enriched.columns]
                            for mc in missing:
                                if mc in base_full.columns:
                                    history_enriched[mc] = base_full.set_index(DATE_COL).reindex(history_enriched[DATE_COL])[mc].values
                            history_enriched = history_enriched.ffill().bfill().fillna(0)
                            
                            pred_h = predict_from_df(model_h, meta_h, history_enriched, device_h)
                            if not pred_h.empty:
                                dates = [last_date + pd.Timedelta(days=d) for d in range(1, h + 1)]
                                vals = pred_h[tgt].values
                                all_dates = [last_date] + dates
                                all_vals = [last_val] + list(vals)
                                
                                # Dùng nét đứt cho các horizon ngắn, nét liền cho 100d
                                dash = "solid" if h == 100 else "dot"
                                width = 3 if h == 100 else 1.5
                                fig.add_trace(go.Scatter(x=all_dates, y=all_vals, name=f"Dự báo {h}d", mode="lines", line=dict(dash=dash, width=width)))
                    except: continue
                
                fig.update_layout(title=f"So sánh lộ trình dự báo: {tgt}", template="plotly_dark", height=350, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# ═══════════════════════════  UI  ═════════════════════════════════════════════

st.set_page_config(page_title="Oil Forecast Hub", layout="wide", page_icon="🛢️")
st.markdown("<style>.block-container { padding-top: 1rem; } h1 { background: linear-gradient(135deg, #00d4aa, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }</style>", unsafe_allow_html=True)
st.title("🛢️ Oil Forecast – Automated Evaluation Hub")
st.info("💡 Lưu ý: MG97 được sử dụng làm biến tham chiếu đầu vào (Feature) để tăng độ chính xác, hiện tại chưa có mô hình dự báo riêng cho mặt hàng này.")

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
    if file_paths:
        combined = run_upload_simulation(str(BUILTIN_CSV), file_paths, CUTOFF_DATE)
        pd.to_pickle({"fp": fingerprint, "df": combined}, CACHE_FILE)
    else: combined = pd.DataFrame()

# Sidebar
st.sidebar.header("⚙️ Cấu hình")
opt = ["Tất cả (So sánh)"] + list(MODEL_DEFS.keys())
sel_opt = st.sidebar.selectbox("Chọn Mô hình hiển thị", opt, index=0)
sel_models = list(MODEL_DEFS.keys()) if sel_opt == "Tất cả (So sánh)" else [sel_opt]

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Xóa Cache Simulation"):
    if CACHE_FILE.exists(): CACHE_FILE.unlink()
    st.cache_data.clear()
    st.rerun()

st.sidebar.header("📁 Lịch sử Upload")
for i, f in enumerate(file_info):
    st.sidebar.text(f"⬆️ {i+1}. {f['name']} ({f['max_date'].strftime('%d/%m')})")

if not combined.empty:
    if "Mặt hàng" in combined.columns:
        combined = combined.rename(columns={"Mặt hàng": "Target"})
    # Chỉ hiển thị dữ liệu từ ngày bắt đầu simulation (Sep 2025) để tránh hiện lịch sử quá cũ
    df_view = combined[(combined["Model"].isin(sel_models)) & (combined[DATE_COL] >= CUTOFF_DATE)]
else: df_view = pd.DataFrame()

# Tabs
t1, t2, t3, t4, t5 = st.tabs(["⬆️ Upload & Dự báo", "🏆 Tổng kết", "📋 Lịch sử upload", "📈 Biểu đồ", "🗃️ Bảng chi tiết"])

with t1:
    # Hiển thị dự báo tương lai dựa trên mô hình đang chọn ở Sidebar
    show_live_forecasts(base_full_orig, file_paths, sel_models)
    
    st.subheader("⬆️ Upload file mới")
    up = st.file_uploader("Chọn file Excel/CSV", type=["xlsx", "xls", "csv"])
    if up:
        tmp = ROOT / "datasets" / up.name
        with open(tmp, "wb") as f: f.write(up.getbuffer())
        st.success(f"✅ Đã lưu: {up.name}. F5 để cập nhật lịch sử.")
        df_new = load_df(tmp)
        if not df_new.empty:
            st.info(f"📅 Dữ liệu mới nhất đến: {df_new[DATE_COL].max().strftime('%d/%m/%Y')}")

with t2:
    if not df_view.empty:
        h_order = [f"{h}d" for h in HORIZONS]
        st.subheader("MAE trung bình")
        mae_piv = df_view.groupby(["Model", "Horizon"])["Sai lệch"].mean().unstack().round(2)
        mae_piv = mae_piv[[c for c in h_order if c in mae_piv.columns]]
        st.dataframe(mae_piv.style.highlight_min(axis=0, color="#00d4aa40"), use_container_width=True)
        st.subheader("MAPE (%) trung bình")
        mape_piv = df_view.groupby(["Model", "Horizon"])["% Lệch"].mean().unstack().round(2)
        mape_piv = mape_piv[[c for c in h_order if c in mape_piv.columns]]
        safe_dataframe(mape_piv)
        
        # HỆ THỐNG CẢNH BÁO THÔNG MINH
        avg_mape = mape_piv.mean().mean()
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
            
            n_epochs = st.number_input("Số vòng lặp (Epochs) huấn luyện thêm", min_value=1, max_value=200, value=20)
            
            if st.button("🚀 Bắt đầu Finetune Ngay"):
                log_area = st.empty()
                log_text = "🛠️ Đang chuẩn bị môi trường...\n"
                log_area.code(log_text)
                
                import subprocess
                cmd = [sys.executable, "train_all_horizons.py", "--update_data", "--epochs", str(n_epochs)]
                
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
