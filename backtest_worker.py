"""backtest_worker.py — logic đối chiếu/backtest KHÔNG phụ thuộc Streamlit, dùng chung bởi cả
app_main.py (để hiển thị) và run_backtest_job.py (chạy nền, độc lập tiến trình web).

Đây là bản trích xuất Streamlit-free của run_upload_simulation()/load_model()/predict_from_df()
vốn nằm trong app_main.py — giữ nguyên logic tính toán, chỉ đổi:
  - @st.cache_data / @st.cache_resource -> cache thường bằng dict trong tiến trình
  - st.empty()/status_text.text(...) -> gọi callback log_fn(msg) (mặc định print)
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
BUILTIN_CSV = ROOT / "oil_forecast_research_new-main" / "data" / "processed" / "clean_data_exo_ver1.csv"
CKPT_DIR = ROOT / "checkpoints_multi"

TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
DATE_COL = "Ngày"
HORIZONS = [1, 5, 10, 15, 20, 30, 60]

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

_DF_CACHE = {}  # (path_str, mtime) -> DataFrame, thay cho @st.cache_data trong tiến trình nền
_MODEL_CACHE = {}  # (name, horizon) -> (model, meta, device), thay cho @st.cache_resource


def load_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime
    key = (str(path), mtime)
    if key in _DF_CACHE:
        return _DF_CACHE[key]
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding="utf-8")
            if df.columns[0].startswith("Ng"):
                df = df.rename(columns={df.columns[0]: DATE_COL})
        df.columns = [str(c).strip() for c in df.columns]
        potential_date_cols = [c for c in df.columns if any(x in c.lower() for x in ["ng", "date", "time"])]
        if potential_date_cols:
            actual_col = potential_date_cols[0]
            if actual_col != DATE_COL:
                df = df.rename(columns={actual_col: DATE_COL})
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")
            df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
            df[DATE_COL] = df[DATE_COL].dt.normalize()
        for c in df.columns:
            if c != DATE_COL:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.interpolate().bfill().ffill()
    except Exception:
        df = pd.DataFrame()
    _DF_CACHE[key] = df
    return df


def generate_time_features(df):
    if DATE_COL not in df.columns:
        return df
    dt = df[DATE_COL]
    for col, val in [("NgayTrongTuan", dt.dt.dayofweek), ("ThangTrongNam", dt.dt.month),
                      ("QuyTrongNam", dt.dt.quarter), ("Nam", dt.dt.year)]:
        if col not in df.columns:
            df[col] = val
    for col in ["NgayLe", "SuKienDacBiet"]:
        if col not in df.columns:
            df[col] = 0
    if "GPRD" not in df.columns:
        df["GPRD"] = df.get("GPR", 0)
    if "Unnamed: 0" not in df.columns:
        df["Unnamed: 0"] = range(len(df))
    return df


def enrich_with_exo(df, base_df):
    df = generate_time_features(df)
    missing = [c for c in base_df.columns if c not in df.columns and c != DATE_COL]
    if not missing:
        return df
    merged = pd.merge(df, base_df[[DATE_COL] + missing], on=DATE_COL, how="left")
    merged[missing] = merged[missing].ffill().bfill()
    for c in missing:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(base_df[c].iloc[-1])
    return merged


def _swap_src(proj_dir):
    d = str(proj_dir)
    sys.path = [p for p in sys.path if p != d]
    sys.path.insert(0, d)
    for m in [k for k in list(sys.modules) if k.startswith("src")]:
        del sys.modules[m]


def load_model(name, horizon):
    key = (name, horizon)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        conf = MODEL_DEFS[name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _swap_src(conf["proj_dir"])
        mod = importlib.import_module(conf["mod"])
        importlib.reload(mod)
        cls = getattr(mod, conf["cls"])

        if name == "GUMNet":
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
                "target_cols": [c.strip() for c in ckpt["target_cols"]],
                "seq_len": ckpt["seq_len"], "horizon": ckpt["horizon"], "kind": "quantile",
                "feature_scaler": ckpt["feature_scaler"],
                "target_scaler": ckpt["target_scaler"],
            }
        else:
            ckpt_path = CKPT_DIR / f"hybrid_h{horizon}.pt"
            meta_dir = CKPT_DIR / f"hybrid_h{horizon}_meta"
            if not ckpt_path.exists() or not meta_dir.exists():
                return None, None, device
            with open(meta_dir / "feature_cols.json") as f:
                fj = json.load(f)
            f_cols = [c.strip() for c in fj.get("feature_cols", TARGET_COLS)]
            K = fj.get("K", 64)
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
        result = (model, meta, device)
    except Exception as e:
        result = (None, str(e), "cpu")
    _MODEL_CACHE[key] = result
    return result


def predict_from_df(model, meta, df, device):
    k = meta["seq_len"]
    f_cols = meta["feature_cols"]
    t_cols = meta["target_cols"]
    n_tgt = len(t_cols)

    X = df.reindex(columns=f_cols, fill_value=0.0).values
    if "feature_scaler" in meta:
        X = meta["feature_scaler"].transform(X)
    else:
        X = (X - meta["x_mu"]) / (meta["x_sd"] + 1e-8)

    x_in = torch.tensor(X[-k:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out, _ = model(x_in)

    if meta["kind"] == "quantile":
        raw = out.cpu().numpy()[0]
        if raw.ndim == 3:
            p50 = meta["target_scaler"].inverse_transform(raw[:, :, 1])
        else:
            p50 = meta["target_scaler"].inverse_transform(raw[..., 1])
    else:
        raw = out.cpu().numpy()[0]
        if raw.ndim == 1:
            total = raw.size
            H = total // n_tgt
            raw = raw.reshape(H, n_tgt)
        y_mu = np.array(meta["y_mu"]).reshape(-1)[-n_tgt:]
        y_sd = np.array(meta["y_sd"]).reshape(-1)[-n_tgt:]
        p50 = raw * (y_sd + 1e-8) + y_mu

    h_out = p50.shape[0]
    last = df[DATE_COL].iloc[-1]
    dates, d = [], last
    while len(dates) < h_out:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)

    result = pd.DataFrame(p50[:len(dates)], columns=t_cols)
    result.insert(0, DATE_COL, [pd.Timestamp(dt).normalize() for dt in dates[:len(result)]])
    return result


def run_upload_simulation(base_path, upload_files, start_date, sel_horizons=None, sel_models=None, log_fn=None):
    log_fn = log_fn or (lambda msg: print(msg, flush=True))
    if sel_horizons is None or len(sel_horizons) == 0:
        sel_horizons = HORIZONS
    if sel_models is None or len(sel_models) == 0:
        sel_models = list(MODEL_DEFS.keys())

    base_full = load_df(base_path)
    base = base_full[base_full[DATE_COL] < start_date].copy()
    all_records = []

    actual_dfs = [base_full]
    for fp in upload_files:
        df_tmp = load_df(fp)
        if not df_tmp.empty:
            actual_dfs.append(df_tmp)
    # keep="last": file upload (nằm sau base_full trong actual_dfs) phải thắng dữ liệu gốc
    # khi trùng ngày, để tính năng "xác nhận ghi đè ngày cũ" có tác dụng thật.
    full_actuals = pd.concat(actual_dfs, ignore_index=True).drop_duplicates(subset=[DATE_COL], keep="last").sort_values(DATE_COL)

    for idx, fpath in enumerate(upload_files):
        df_upload = load_df(fpath)
        if df_upload.empty:
            continue

        for tc in TARGET_COLS:
            for c in df_upload.columns:
                if tc.replace(" ", "").lower() == str(c).replace(" ", "").lower():
                    df_upload = df_upload.rename(columns={c: tc})

        avail_tgt = [c for c in TARGET_COLS if c in df_upload.columns]
        if not avail_tgt:
            continue

        base_dates = set(base[DATE_COL].dt.strftime("%Y-%m-%d"))
        new_rows = df_upload[~df_upload[DATE_COL].dt.strftime("%Y-%m-%d").isin(base_dates)].copy()
        new_rows = new_rows[new_rows[DATE_COL] >= start_date]

        if not new_rows.empty:
            base_for_pred = pd.concat([base_full[base_full[DATE_COL] < base[DATE_COL].min()], base], ignore_index=True)
            base_for_pred = base_for_pred.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).tail(500)
            base_enriched = enrich_with_exo(base_for_pred, base_full)

            for mname in sel_models:
                try:
                    match_data = []
                    indices = np.unique(np.linspace(0, len(new_rows) - 1, 3, dtype=int))
                    indices = [i for i in indices if 0 <= i < len(new_rows)]

                    import gc
                    for step_i, idx_in_new in enumerate(indices):
                        log_fn(f"⏳ {mname} | File {idx+1} | Điểm {step_i+1}/{len(indices)}")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        history_df = pd.concat([base_enriched, new_rows.iloc[:idx_in_new]], ignore_index=True)
                        actual_pool = new_rows.iloc[idx_in_new:idx_in_new + 1].copy()
                        if actual_pool.empty:
                            continue

                        for h in sorted(sel_horizons):
                            model_h, meta_h, device_h = load_model(mname, h)
                            if not model_h:
                                continue
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
                            if pred_df.empty:
                                continue

                            idx_h = min(h - 1, len(pred_df) - 1)
                            p_row = pred_df.iloc[idx_h]
                            pred_date = pd.Timestamp(p_row[DATE_COL]).normalize()

                            match_row = full_actuals[full_actuals[DATE_COL] == pred_date]
                            if not match_row.empty:
                                a_row = match_row.iloc[0]
                                for tgt in avail_tgt:
                                    if tgt in p_row and tgt in a_row.index and pd.notna(a_row[tgt]):
                                        match_data.append({
                                            "Model": mname, "Horizon": f"{h}d",
                                            "Upload": f"#{idx+1} {fpath.name}",
                                            DATE_COL: pred_date, "Target": tgt,
                                            "Dự báo": round(float(p_row[tgt]), 2),
                                            "Thực tế": round(float(a_row[tgt]), 2),
                                            "Sai lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])), 2),
                                            "% Lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])) / (abs(float(a_row[tgt])) + 1e-8) * 100, 2),
                                        })

                    if match_data:
                        res_df = pd.DataFrame(match_data).drop_duplicates(subset=["Model", "Horizon", DATE_COL, "Target"])
                        all_records.extend(res_df.to_dict("records"))
                except Exception as e:
                    log_fn(f"EXCEPTION: {mname}: {e}")
                    continue

            base = pd.concat([base, new_rows], ignore_index=True)
            base = base.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    return pd.DataFrame(all_records)
