import io, json
from pathlib import Path
import os, sys
from datetime import datetime
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
from torch.utils.data import DataLoader
from pandas.tseries.offsets import BDay
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

torch.classes.__path__ = []

from src.utils.paths import BASE_DIR, RUN_OUTPUT_DIR
from src.utils.config_loader import load_yaml_config, load_env_secrets

cfg = load_yaml_config()
load_env_secrets()

DATE_COL_CFG = cfg.get("default_date_col", "Ngày")
DEFAULT_H_NEXT = int(cfg.get("default_h_next", 5))

default_clean_rel = cfg.get("default_clean_path", "data/base/du_lieu_noi_suy_clean_updated_end_14-11.xlsx")
DEFAULT_CLEAN_PATH = (BASE_DIR / default_clean_rel).resolve()

FRED_API_KEY_DEFAULT = os.getenv("FRED_API_KEY", "")

from src.dataio import build_merged, _ensure_date, _align_union_columns
from src.features import _coerce_targets_numeric
from src.model.hybrid_trinet import HybridTriNet
from src.model.training import (
    set_seed,
    standardize,
    build_windows,
    WindowDS,
    eval_metrics_orig,
    fit_model,
    roll_autoregressive,
    USE_L1_LOSS,
    SMOOTHL1_BETA,
    WEIGHT_DECAY,
)

st.set_page_config(page_title="Dự đoán", layout="wide")
st.title("DỰ BÁO GIÁ XĂNG DẦU THEO CHUỖI THỜI GIAN")

DATE_COL = "Ngày"
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]

K = 64
H = 14
BATCH_SZ = 128
EPOCHS = 250
LR = 1e-4
VAL_RATIO = 0.10
SEED = 42

fred_api_key = FRED_API_KEY_DEFAULT


def _parse_dates_any(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.normalize()

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        xi = x.round().astype("Int64")

        m_yyyymmdd = xi.between(19000101, 21001231)
        if m_yyyymmdd.any():
            out = out.fillna(pd.to_datetime(xi.where(m_yyyymmdd).astype(str), format="%Y%m%d", errors="coerce"))

        m_excel = x.between(1, 60000)
        if m_excel.any():
            out = out.fillna(pd.to_datetime(x.where(m_excel), unit="D", origin="1899-12-30", errors="coerce"))

        m_sec = x.between(1e9, 2e9)
        if m_sec.any():
            out = out.fillna(pd.to_datetime(x.where(m_sec), unit="s", errors="coerce"))

        m_ms = x.between(1e12, 2e12)
        if m_ms.any():
            out = out.fillna(pd.to_datetime(x.where(m_ms), unit="ms", errors="coerce"))

        m_us = x.between(1e15, 2e15)
        if m_us.any():
            out = out.fillna(pd.to_datetime(x.where(m_us), unit="us", errors="coerce"))

        return out.dt.normalize()

    s2 = s.astype(str).str.strip()

    mask_ymd = s2.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?$")
    if mask_ymd.any():
        out = out.fillna(pd.to_datetime(s2.where(mask_ymd), errors="coerce", dayfirst=False, yearfirst=True))

    mask_yearstart = s2.str.match(r"^\d{4}[-/]")
    fb_year = pd.to_datetime(s2.where(mask_yearstart), errors="coerce", dayfirst=False, yearfirst=True)
    fb_day = pd.to_datetime(s2.where(~mask_yearstart), errors="coerce", dayfirst=True)
    out = out.fillna(fb_year).fillna(fb_day)

    num = pd.to_numeric(s2, errors="coerce")
    if num.notna().any():
        m_excel2 = num.between(1, 60000)
        if m_excel2.any():
            out = out.fillna(pd.to_datetime(num.where(m_excel2), unit="D", origin="1899-12-30", errors="coerce"))

    return out.dt.normalize()


def _read_upload_file(up):
    data = up.getvalue()
    suf = Path(up.name).suffix.lower()
    bio = io.BytesIO(data)
    if suf == ".csv":
        return pd.read_csv(bio)
    if suf in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        return pd.read_excel(bio, engine="openpyxl")
    if suf == ".xls":
        try:
            return pd.read_excel(bio, engine="xlrd")
        except Exception:
            return pd.read_excel(bio)
    raise ValueError(f"Unsupported file type: {suf}")


def _get_upload_last_date(up, date_col: str):
    u = _read_upload_file(up)
    if date_col not in u.columns:
        raise ValueError(f"File upload thiếu cột ngày '{date_col}'")
    u[date_col] = _parse_dates_any(u[date_col])
    last = u[date_col].dropna().max()
    if pd.isna(last):
        raise ValueError("Không đọc được ngày hợp lệ trong file upload")
    return pd.Timestamp(last).normalize()


def _read_actual_full(clean_path_str: str, date_col: str) -> pd.DataFrame:
    base_full = pd.read_excel(clean_path_str, engine="openpyxl")
    if date_col not in base_full.columns:
        raise ValueError(f"File gốc thiếu cột ngày '{date_col}'")
    base_full[date_col] = _parse_dates_any(base_full[date_col])
    base_full = base_full.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return base_full


def _apply_fill_mode(df: pd.DataFrame, date_col: str, fill_mode: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = _parse_dates_any(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    cols = [c for c in df.columns if c != date_col]
    if fill_mode == "none":
        return df
    if fill_mode == "ffill":
        df.loc[:, cols] = df.loc[:, cols].ffill()
        return df
    if fill_mode == "ffill+bfill":
        df.loc[:, cols] = df.loc[:, cols].ffill().bfill()
        return df
    if fill_mode == "drop rows with any NaN":
        return df.dropna()
    return df


def _interpolate_external(df: pd.DataFrame, date_col: str, cols=("USD_Index", "GPRD")) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = _parse_dates_any(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tmp = df.set_index(date_col)
    try:
        tmp[list(cols)] = tmp[list(cols)].interpolate(method="time", limit_direction="both")
    except Exception:
        tmp[list(cols)] = tmp[list(cols)].interpolate(limit_direction="both")
    tmp[list(cols)] = tmp[list(cols)].ffill().bfill()
    return tmp.reset_index()


def merge_keep_nonnull(base_df: pd.DataFrame, new_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    b = base_df.copy()
    n = new_df.copy()

    b[date_col] = _parse_dates_any(b[date_col])
    n[date_col] = _parse_dates_any(n[date_col])

    b = b.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    n = n.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    b["_src"] = 0
    n["_src"] = 1

    all_df = pd.concat([b, n], ignore_index=True)
    all_df = all_df.dropna(subset=[date_col]).sort_values([date_col, "_src"]).reset_index(drop=True)

    def _take_last_after_ffill(g: pd.DataFrame) -> pd.DataFrame:
        g2 = g.sort_values("_src").ffill()
        return g2.tail(1)

    out = (
        all_df.groupby(date_col, as_index=False, group_keys=False)
        .apply(_take_last_after_ffill)
        .drop(columns=["_src"], errors="ignore")
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    return out


def _fallback_merge_upload_only(up_pp, actual_full: pd.DataFrame, date_col: str, fill_mode: str):
    u = _read_upload_file(up_pp)
    if date_col not in u.columns:
        df0 = actual_full.copy()
        df0 = _apply_fill_mode(df0, date_col, fill_mode)
        df0 = _interpolate_external(df0, date_col)
        return df0

    u = u.copy()
    u[date_col] = _parse_dates_any(u[date_col])
    u = u.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    base = actual_full.copy()
    base[date_col] = _parse_dates_any(base[date_col])
    base = base.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    base2, add2, _ = _align_union_columns(base, u, date_col)
    df_updated = merge_keep_nonnull(base2, add2, date_col)
    df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
    df_updated = _interpolate_external(df_updated, date_col)
    return df_updated


def save_forecast_history(out: pd.DataFrame, last_date: pd.Timestamp, h_next: int, date_col: str):
    forecast_dir = RUN_OUTPUT_DIR / "forecast_history"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    for f in forecast_dir.glob(f"forecast_until_{pd.Timestamp(last_date).strftime('%Y%m%d')}_H{int(h_next)}_*.csv"):
        try:
            f.unlink()
        except Exception:
            pass

    out = out.copy()
    out[date_col] = _parse_dates_any(out[date_col])
    out["train_last_date"] = pd.Timestamp(last_date).normalize()
    out["generated_at"] = pd.Timestamp.now()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = forecast_dir / f"forecast_until_{pd.Timestamp(last_date).strftime('%Y%m%d')}_H{int(h_next)}_{ts}.csv"
    out.to_csv(fname, index=False)
    return fname


def _plot_compare_chart(merged: pd.DataFrame, date_col: str, target_cols):
    tabs_targets = [c for c in target_cols if f"{c}_pred" in merged.columns and f"{c}_actual" in merged.columns]
    if not tabs_targets:
        return False

    tabs = st.tabs(tabs_targets)
    for target, tab in zip(tabs_targets, tabs):
        with tab:
            sub = merged[[date_col, f"{target}_pred", f"{target}_actual"]].copy()
            sub[date_col] = _parse_dates_any(sub[date_col])
            sub[f"{target}_pred"] = pd.to_numeric(sub[f"{target}_pred"], errors="coerce")
            sub[f"{target}_actual"] = pd.to_numeric(sub[f"{target}_actual"], errors="coerce")
            sub = sub.dropna(subset=[date_col])

            long = sub.melt(
                id_vars=[date_col],
                value_vars=[f"{target}_pred", f"{target}_actual"],
                var_name="loại",
                value_name="giá trị",
            )
            long["loại"] = long["loại"].map({f"{target}_pred": "Dự báo", f"{target}_actual": "Thực tế"})

            ch = (
                alt.Chart(long)
                .mark_line(point=alt.OverlayMarkDef(size=80, filled=True))
                .encode(
                    x=alt.X(f"{date_col}:T", title="Ngày"),
                    y=alt.Y("giá trị:Q", title="Giá"),
                    color=alt.Color("loại:N", title=""),
                    tooltip=[date_col, "loại", "giá trị"],
                )
                .properties(height=360)
            )
            st.altair_chart(ch, width="stretch")
    return True


def compare_pred_vs_actual_quiet(pred_df: pd.DataFrame, actual_df: pd.DataFrame, date_col: str, target_cols):
    if pred_df is None or len(pred_df) == 0:
        return False

    p = pred_df.copy()
    if date_col not in p.columns:
        return False
    p[date_col] = _parse_dates_any(p[date_col])
    p = p.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    a = actual_df.copy()
    if date_col not in a.columns:
        return False
    a[date_col] = _parse_dates_any(a[date_col])
    a = a.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    max_act = a[date_col].max()
    if pd.notna(max_act):
        p = p[p[date_col] <= max_act].copy()

    keep_a = [date_col] + [c for c in target_cols if c in a.columns]
    keep_p = [date_col] + [c for c in target_cols if c in p.columns]
    a = a[keep_a].copy()
    p = p[keep_p].copy()

    merged = p.merge(a, on=date_col, suffixes=("_pred", "_actual"))
    if merged.empty:
        return False

    st.subheader("Biểu đồ so sánh (Dự báo vs Thực tế)")
    return _plot_compare_chart(merged, date_col, target_cols)


def compare_best_history_vs_actual_quiet(actual_df: pd.DataFrame, date_col: str, target_cols):
    forecast_dir = RUN_OUTPUT_DIR / "forecast_history"
    if not forecast_dir.exists():
        return False

    files = list(forecast_dir.glob("forecast_until_*.csv"))
    if not files:
        return False

    act = actual_df.copy()
    if date_col not in act.columns:
        return False
    act[date_col] = _parse_dates_any(act[date_col])
    act = act.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    act_keep = [date_col] + [c for c in target_cols if c in act.columns]
    act = act[act_keep].copy()
    act_dates = set(act[date_col].tolist())

    best_key = None
    best_file = None
    best_pred = None

    for f in files:
        try:
            tmp = pd.read_csv(f)
        except Exception:
            continue
        if date_col not in tmp.columns:
            continue
        tmp[date_col] = _parse_dates_any(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        have = set(tmp[date_col].tolist())
        overlap = len(have & act_dates)
        if overlap <= 0:
            continue
        key = (overlap, f.stat().st_mtime)
        if best_key is None or key > best_key:
            best_key = key
            best_file = f
            best_pred = tmp

    if best_pred is None:
        return False

    pred = best_pred.copy()
    pred_keep = [date_col] + [c for c in target_cols if c in pred.columns]
    pred = pred[pred_keep].copy()

    merged = pred.merge(act, on=date_col, suffixes=("_pred", "_actual"))
    if merged.empty:
        return False

    st.subheader("Biểu đồ so sánh (Dự báo vs Thực tế)")
    st.caption(f"Lấy từ forecast_history: {best_file.name} | overlap={merged[date_col].nunique()} ngày")
    return _plot_compare_chart(merged, date_col, target_cols)


def smart_compare(pred_df, actual_df, date_col, target_cols):
    ok = compare_pred_vs_actual_quiet(pred_df, actual_df, date_col, target_cols)
    if ok:
        return
    ok2 = compare_best_history_vs_actual_quiet(actual_df, date_col, target_cols)
    if ok2:
        return
    st.info("Hiện chưa có ngày trùng giữa dự báo và dữ liệu thực tế để so sánh.")


def run_forecast(df: pd.DataFrame, date_col: str, h_next: int, save_history: bool):
    set_seed(SEED)
    device_train = "cuda" if torch.cuda.is_available() else "cpu"

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        st.error(f"Thiếu các cột target trong dữ liệu: {missing}")
        return None, False

    df = df.copy()
    df[date_col] = _parse_dates_any(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    try:
        Y = _coerce_targets_numeric(df, TARGET_COLS)
    except Exception as e:
        st.error(f"Lỗi chuyển target sang số: {e}")
        return None, False

    T, D = Y.shape
    if T <= K + H:
        st.error(f"Dữ liệu quá ngắn (T={T}) so với K={K}, H={H}.")
        return None, False

    val_len = max(int(T * VAL_RATIO), K + H + 1)
    train_len = T - val_len
    Y_tr, Y_val = Y[:train_len], Y[train_len - K:]

    Ytr_std, mu, sd = standardize(Y_tr)
    Yval_std = (Y_val - mu) / (sd + 1e-8)

    Xtr, Ytrw = build_windows(Ytr_std, K, H)
    Xva, Yvaw = build_windows(Yval_std, K, H)

    tr_ld = DataLoader(WindowDS(Xtr, Ytrw), batch_size=BATCH_SZ, shuffle=True)
    va_ld = DataLoader(WindowDS(Xva, Yvaw), batch_size=BATCH_SZ, shuffle=False)

    RUN = RUN_OUTPUT_DIR
    RUN.mkdir(parents=True, exist_ok=True)
    ckpt_path = RUN / "hybrid_trinet_streamlit.pt"

    model = HybridTriNet(
        k=K,
        D=D,
        H=H,
        d_feat=96,
        kan_M=8,
        kan_depth=2,
        kan_drop=0.1,
        gru_hidden=128,
        gru_layers=1,
        gru_drop=0.1,
        attn_dmodel=48,
        attn_heads=3,
        attn_layers=2,
        attn_drop=0.05,
        patch_len=16,
        stride=8,
    ).to(device_train)

    progress_bar = st.progress(0.0)

    def _status_cb(ep, epochs, tr, va, mae, lr_val):
        progress_bar.progress(ep / epochs)

    t0 = time.time()
    with st.spinner("Đang dự đoán..."):
        best = fit_model(
            model,
            tr_ld,
            va_ld,
            H,
            D,
            EPOCHS,
            LR,
            mu,
            sd,
            device=device_train,
            name="HybridTriNet",
            status_cb=_status_cb,
        )
    t1 = time.time()
    st.write(f"Thời gian train: {t1 - t0:.2f} giây")
    progress_bar.progress(1.0)

    metrics_val = eval_metrics_orig(model, va_ld, H, D, mu, sd, device=device_train)

    np.save(RUN / "mu.npy", mu)
    np.save(RUN / "sd.npy", sd)

    diag = {
        "targets": TARGET_COLS,
        "val_loss_standardized": float(best["val"]),
        "val_mae_orig": float(metrics_val["mae"]),
        "val_mse_orig": float(metrics_val["mse"]),
        "val_rmse_orig": float(metrics_val["rmse"]),
        "val_r2_orig": float(metrics_val["r2"]),
        "k": K,
        "H": H,
        "val_ratio": VAL_RATIO,
        "loss": "L1" if USE_L1_LOSS else f"SmoothL1(beta={SMOOTHL1_BETA})",
        "ema_decay": 0.999,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best["ep"],
        "init_from_ckpt": False,
    }
    (RUN / "diagnostics_streamlit.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(model.state_dict(), ckpt_path)

    Y_std_full = (Y - mu) / (sd + 1e-8)
    seed = Y_std_full[-K:]
    F_std = roll_autoregressive(model, seed_std=seed, H_total=int(h_next), H=H, device=device_train)
    F = F_std * sd + mu

    last_date = pd.Timestamp(df[date_col].max()).normalize()
    st.caption(f"Ngày cuối dùng để dự báo: {last_date.date()}")
    idx = pd.bdate_range(last_date + BDay(1), periods=int(h_next))

    out = pd.DataFrame(F, index=idx, columns=TARGET_COLS)
    out_to_save = out.reset_index().rename(columns={"index": date_col})[[date_col] + TARGET_COLS].copy()

    st.subheader(f"Dự đoán {int(h_next)} ngày tiếp theo")
    st.dataframe(out_to_save, width="stretch")

    saved_ok = False
    if save_history:
        fname_hist = save_forecast_history(out_to_save, last_date, int(h_next), date_col)
        st.caption(f"Đã lưu forecast_history: {fname_hist.name}")
        saved_ok = True

    return out_to_save, saved_ok


date_col = st.text_input("Cột ngày", DATE_COL_CFG)

with st.sidebar:
    st.header("Cấu hình")
    clean_path = st.text_input("Đường dẫn dữ liệu gốc", str(DEFAULT_CLEAN_PATH))
    fill_mode = st.selectbox("Xử lý NaN sau khi gộp dữ liệu", ["none", "ffill", "ffill+bfill", "drop rows with any NaN"], index=1)
    h_next = st.number_input("Số ngày dự đoán", 1, 365, DEFAULT_H_NEXT, 1)

if "df_merged" not in st.session_state:
    st.session_state.df_merged = None

base_info_box = st.empty()
clean_path_str = clean_path.strip()

if clean_path_str:
    p = Path(clean_path_str)
    if not p.exists():
        base_info_box.warning(f"Không tìm thấy file gốc: {clean_path_str}")
    else:
        try:
            base0 = pd.read_excel(p, engine="openpyxl")
            if date_col not in base0.columns:
                base_info_box.error(f"Thiếu cột ngày '{date_col}' trong file gốc.")
            else:
                base0 = _ensure_date(base0, date_col)
                base0[date_col] = _parse_dates_any(base0[date_col])
                base_last_display = base0[date_col].dropna().max()
                if pd.isna(base_last_display):
                    base_info_box.warning("Không xác định được ngày cuối trong dữ liệu gốc.")
                else:
                    base_last_display = pd.Timestamp(base_last_display).normalize()
                    base_info_box.info(f"Ngày cuối cùng trong dữ liệu gốc: **{base_last_display.date()}**.")
                if st.session_state.df_merged is None:
                    st.session_state.df_merged = base0.copy()
        except Exception as e:
            base_info_box.error(f"Lỗi đọc file gốc: {e}")

st.subheader("Tải lên file price_petroleum")
up_pp = st.file_uploader("price_petroleum (.csv/.xlsx/.xls)", type=["csv", "xlsx", "xls"], key="up_pp")

if up_pp is not None:
    if not clean_path_str:
        st.error("Thiếu đường dẫn clean.xlsx")
        st.stop()

    try:
        upload_last_date = _get_upload_last_date(up_pp, date_col)
        st.info(f"Ngày cuối trong file upload: **{upload_last_date.date()}**")
    except Exception as e:
        st.error(f"Lỗi đọc ngày cuối file upload: {e}")
        st.stop()

    try:
        actual_full_before = _read_actual_full(clean_path_str, date_col)
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu thực tế (clean.xlsx): {e}")
        st.stop()

    base_last_before = actual_full_before[date_col].dropna().max()
    base_last_before = pd.Timestamp(base_last_before).normalize() if pd.notna(base_last_before) else pd.NaT
    upload_last_date = pd.Timestamp(upload_last_date).normalize()

    is_new_upload = bool(pd.notna(base_last_before) and upload_last_date > base_last_before)

    if not is_new_upload:
        df_updated = actual_full_before.copy()
        df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
        df_updated = _interpolate_external(df_updated, date_col)
        should_update_clean = False
        actual_full = actual_full_before.copy()
    else:
        df_updated = None
        build_ok = True
        try:
            if fred_api_key:
                df_new, base_clean, info = build_merged(up_pp, clean_path_str, date_col, fill_mode, fred_api_key)
                if df_new is None or len(df_new) == 0:
                    df_updated = actual_full_before.copy()
                else:
                    base2, add2, _ = _align_union_columns(actual_full_before, df_new, date_col)
                    df_updated = merge_keep_nonnull(base2, add2, date_col)
            else:
                build_ok = False
        except Exception:
            build_ok = False

        if not build_ok or df_updated is None:
            df_updated = _fallback_merge_upload_only(up_pp, actual_full_before, date_col, fill_mode)

        df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
        df_updated = _interpolate_external(df_updated, date_col)

        df_updated[date_col] = _parse_dates_any(df_updated[date_col])
        new_last_after = df_updated[date_col].dropna().max()
        new_last_after = pd.Timestamp(new_last_after).normalize() if pd.notna(new_last_after) else pd.NaT

        should_update_clean = bool(
            pd.notna(base_last_before)
            and pd.notna(new_last_after)
            and (upload_last_date > base_last_before)
            and (new_last_after > base_last_before)
        )

        if should_update_clean:
            try:
                df_updated.to_excel(clean_path_str, index=False, engine="openpyxl")
                st.success(f"Đã lưu dữ liệu gộp vào: {clean_path_str}")
                actual_full = _read_actual_full(clean_path_str, date_col)
            except Exception as e:
                st.error(f"Lỗi lưu file gộp: {e}")
                actual_full = actual_full_before.copy()
        else:
            actual_full = actual_full_before.copy()

    st.session_state.df_merged = df_updated

    df_use = df_updated.copy()
    df_use[date_col] = _parse_dates_any(df_use[date_col])
    df_use = df_use.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df_use = df_use[df_use[date_col] <= upload_last_date].copy()

    st.caption(f"Dự báo sẽ bắt đầu từ: {(upload_last_date + BDay(1)).date()}")

    st.markdown("---")
    pred_df, saved_ok = run_forecast(df_use, date_col, int(h_next), save_history=should_update_clean)
    st.markdown("---")

    smart_compare(pred_df, actual_full, date_col, TARGET_COLS)
