from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader
from pandas.tseries.offsets import BDay
import altair as alt

from .config import TARGET_COLS, K, H, VAL_RATIO, get_defaults
from .style import inject_css
from .ui import page_header, section_header, soft_divider
from .data_helpers import (
    _read_actual_full,
    _parse_dates_any,
    _read_upload_file,
    _get_upload_last_date,
    _apply_fill_mode,
    _interpolate_external,
    merge_keep_nonnull,
)
from .plots import plot_candlestick_preview
from .train_focus5 import fit_model_better
from .autoregressive import roll_autoregressive_safe, ensure_F_shape
from .calibration import fit_calibration_from_history, apply_calibration
from .history_eval import load_actual_root, compute_metrics

from src.dataio import build_merged, _ensure_date, _align_union_columns
from src.features import _coerce_targets_numeric
from src.model.hybrid_trinet import HybridTriNet
from src.model.training import set_seed, standardize, build_windows, WindowDS
from src.utils.paths import RUN_OUTPUT_DIR


START0 = pd.Timestamp("2025-09-19").normalize()


def _ts_seed_base() -> int:
    return 20260320


def _history_dir_for_h(h: int) -> Path:
    base_dir = RUN_OUTPUT_DIR / "forecast_history"
    base_dir.mkdir(parents=True, exist_ok=True)
    h = int(h)
    return (base_dir / str(h)) if h in (30, 60, 100) else base_dir


# ============================================================
# History reader (robust) for forecast_until_*.csv (wide files)
# ============================================================
def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ["date", "Date", "DATE", "Ngày", "NGAY", "ngay", "time", "Time", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) > 0 else None


def _read_history_wide_file_to_long(
    csv_path: Path, targets: List[str], date_col_hint: Optional[str] = None
) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    dc = date_col_hint if (date_col_hint and date_col_hint in df.columns) else _detect_date_col(df)
    if dc is None or dc not in df.columns:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    df = df.copy()
    df = df.rename(columns={dc: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.normalize()

    keep_targets = [t for t in targets if t in df.columns]
    if not keep_targets:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    meta_cols = [c for c in ["train_last_date", "generated_at", "run_seed_base"] if c in df.columns]

    long_df = df.melt(id_vars=["date"] + meta_cols, value_vars=keep_targets, var_name="target", value_name="yhat")
    long_df["source_file"] = csv_path.name

    if "train_last_date" in long_df.columns:
        long_df["train_last_date"] = pd.to_datetime(long_df["train_last_date"], errors="coerce", dayfirst=True).dt.normalize()
    else:
        long_df["train_last_date"] = pd.NaT

    if "generated_at" in long_df.columns:
        long_df["generated_at"] = pd.to_datetime(long_df["generated_at"], errors="coerce", dayfirst=True)
    else:
        long_df["generated_at"] = pd.NaT

    if "run_seed_base" in long_df.columns:
        long_df["run_seed_base"] = pd.to_numeric(long_df["run_seed_base"], errors="coerce")
    else:
        long_df["run_seed_base"] = np.nan

    long_df["yhat"] = pd.to_numeric(long_df["yhat"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "target"]).reset_index(drop=True)
    return long_df


def _load_history_long_from_dir(hist_dir: Path, targets: List[str], date_col_hint: Optional[str] = None) -> pd.DataFrame:
    if hist_dir is None or (not hist_dir.exists()):
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    files = sorted(hist_dir.glob("forecast_until_*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    parts = []
    for fp in files:
        parts.append(_read_history_wide_file_to_long(fp, targets=targets, date_col_hint=date_col_hint))

    if not parts:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    out = pd.concat(parts, ignore_index=True)

    # sort key for "latest_asof"
    out["_asof_sort"] = out["train_last_date"].copy()
    out["_asof_sort"] = out["_asof_sort"].where(out["_asof_sort"].notna(), out["generated_at"])
    out["_asof_sort"] = out["_asof_sort"].fillna(pd.Timestamp("1900-01-01"))
    return out


def latest_asof_per_date_target(hist_long: pd.DataFrame) -> pd.DataFrame:
    """
    Lấy forecast mới nhất cho mỗi (date,target) dựa trên:
    train_last_date -> generated_at -> (fallback 1900)
    Đồng thời lọc date > train_last_date (forecast hợp lệ).
    """
    if hist_long is None or hist_long.empty:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    df = hist_long.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.normalize()
    df = df.dropna(subset=["date", "target"]).copy()

    # lọc hợp lệ: date phải > train_last_date nếu có
    if "train_last_date" in df.columns:
        tld = pd.to_datetime(df["train_last_date"], errors="coerce", dayfirst=True).dt.normalize()
        df = df[df["date"] > tld].copy()

    if "_asof_sort" not in df.columns:
        df["_asof_sort"] = pd.to_datetime(df.get("train_last_date"), errors="coerce", dayfirst=True).dt.normalize()
        df["_asof_sort"] = df["_asof_sort"].where(
            df["_asof_sort"].notna(),
            pd.to_datetime(df.get("generated_at"), errors="coerce", dayfirst=True),
        )
        df["_asof_sort"] = df["_asof_sort"].fillna(pd.Timestamp("1900-01-01"))

    df = (
        df.sort_values(["date", "target", "_asof_sort"])
          .drop_duplicates(["date", "target"], keep="last")
          .drop(columns=["_asof_sort"], errors="ignore")
    )
    return df


def _latest_pred_long_for_dir(hist_dir: Path, targets: List[str], date_col_hint: Optional[str] = None) -> pd.DataFrame:
    hl = _load_history_long_from_dir(hist_dir, targets=targets, date_col_hint=date_col_hint)
    if hl is None or hl.empty:
        return pd.DataFrame(columns=["date", "target", "yhat"])
    hl = latest_asof_per_date_target(hl)
    return hl[["date", "target", "yhat"]].copy()


def _count_history_files(hist_dir: Path) -> int:
    if hist_dir is None or (not hist_dir.exists()):
        return 0
    return len(list(hist_dir.glob("forecast_until_*.csv")))


def _filter_valid_forecast(hist_long: pd.DataFrame) -> pd.DataFrame:
    """
    Lọc các dòng forecast hợp lệ:
    - date không null, target không null
    - yhat numeric
    - nếu có train_last_date: chỉ giữ date > train_last_date
    """
    if hist_long is None or hist_long.empty:
        return pd.DataFrame(columns=["date", "target", "yhat", "train_last_date", "generated_at", "run_seed_base", "source_file"])

    df = hist_long.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.normalize()
    df = df.dropna(subset=["date", "target"]).copy()
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
    df = df.dropna(subset=["yhat"]).copy()

    if "train_last_date" in df.columns:
        tld = pd.to_datetime(df["train_last_date"], errors="coerce", dayfirst=True).dt.normalize()
        df = df[(tld.isna()) | (df["date"] > tld)].copy()

    return df.reset_index(drop=True)


def _agg_mean_by_date_target(hist_long: pd.DataFrame) -> pd.DataFrame:
    """
    Gộp forecast trùng (date,target) bằng trung bình yhat.
    Giữ thêm n_forecasts + train_last_date_max + generated_at_max để xem.
    """
    if hist_long is None or hist_long.empty:
        return pd.DataFrame(columns=["date", "target", "yhat", "n_forecasts", "train_last_date", "generated_at"])

    df = _filter_valid_forecast(hist_long)
    if df.empty:
        return pd.DataFrame(columns=["date", "target", "yhat", "n_forecasts", "train_last_date", "generated_at"])

    if "train_last_date" in df.columns:
        df["train_last_date"] = pd.to_datetime(df["train_last_date"], errors="coerce", dayfirst=True).dt.normalize()
    else:
        df["train_last_date"] = pd.NaT

    if "generated_at" in df.columns:
        df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", dayfirst=True)
    else:
        df["generated_at"] = pd.NaT

    g = (
        df.groupby(["date", "target"], as_index=False)
          .agg(
              yhat=("yhat", "mean"),
              n_forecasts=("yhat", "size"),
              train_last_date=("train_last_date", "max"),
              generated_at=("generated_at", "max"),
          )
    )
    return g


# ============================================================
# Backfill theo chu kỳ H: mỗi H ngày làm việc chạy 1 lần
# ============================================================
def backfill_by_horizon_period(
    df_full: pd.DataFrame,
    date_col: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizons: List[int],
    train_cfg: Dict,
    min_len: int = None,
) -> Tuple[int, int]:
    """
    Với mỗi H:
      - asof = start_date, start_date+H BDay, start_date+2H BDay, ... đến end_date (thêm end_date nếu chưa có)
      - mỗi asof: cắt df_full tới asof rồi run_forecast(h_next=H, save_history=True)
    """
    if df_full is None or df_full.empty:
        st.error("Không có df_full để backfill.")
        return 0, 0

    df_full = df_full.copy()
    df_full[date_col] = _parse_dates_any(df_full[date_col])
    df_full = df_full.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if min_len is None:
        min_len = int(K + H + 5)

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    horizons = sorted({int(hh) for hh in horizons if int(hh) > 0})
    if not horizons:
        st.warning("Chưa chọn horizon để backfill.")
        return 0, 0

    asof_map = {}
    total = 0
    for hh in horizons:
        asofs = list(pd.bdate_range(start_date, end_date, freq=BDay(int(hh))))
        if len(asofs) == 0:
            asofs = [end_date]
        if asofs[-1] != end_date:
            asofs.append(end_date)
        asof_map[hh] = asofs
        total += len(asofs)

    done = 0
    prog = st.progress(0.0)
    status = st.empty()

    for hh in horizons:
        for asof in asof_map[hh]:
            done += 1
            prog.progress(min(1.0, done / max(1, total)))

            sub = df_full[df_full[date_col] <= asof].copy()
            if len(sub) < min_len:
                continue

            status.caption(f"Backfill theo chu kỳ H={hh} | asof={asof.date()} | rows={len(sub)}")
            run_forecast(
                sub,
                date_col=date_col,
                h_next=int(hh),
                save_history=True,
                retrain=False,
                train_cfg=train_cfg,
                actual_full=sub,
            )

    prog.progress(1.0)
    status.empty()
    return done, total


# ============================================================
# Forecast core
# ============================================================
def save_forecast_history(
    out: pd.DataFrame, last_date: pd.Timestamp, h_next: int, date_col: str, run_seed_base: int
):
    h_next_i = int(h_next)
    forecast_dir = _history_dir_for_h(h_next_i)

    out = out.copy()
    out[date_col] = _parse_dates_any(out[date_col])
    out["train_last_date"] = pd.Timestamp(last_date).normalize()
    out["generated_at"] = pd.Timestamp.now()
    out["run_seed_base"] = int(run_seed_base)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = forecast_dir / f"forecast_until_{pd.Timestamp(last_date).strftime('%Y%m%d')}_H{h_next_i}_{ts}.csv"
    out.to_csv(fname, index=False, date_format="%d/%m/%Y %I:%M:%S %p")
    return fname


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    h_next: int,
    save_history: bool,
    retrain: bool,
    train_cfg: Dict,
    actual_full: Optional[pd.DataFrame] = None,
):
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
    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    sd = np.asarray(sd, dtype=np.float32).reshape(-1)

    if mu.size < D or sd.size < D:
        st.error(f"mu/sd không khớp số target D={D}. mu={mu.shape}, sd={sd.shape}")
        return None, False

    Yval_std = (Y_val - mu[:D]) / (sd[:D] + 1e-8)

    Xtr, Ytrw = build_windows(Ytr_std, K, H)
    Xva, Yvaw = build_windows(Yval_std, K, H)

    tr_ld = DataLoader(WindowDS(Xtr, Ytrw), batch_size=int(train_cfg["batch"]), shuffle=True)
    va_ld = DataLoader(WindowDS(Xva, Yvaw), batch_size=int(train_cfg["batch"]), shuffle=False)

    RUN = RUN_OUTPUT_DIR
    RUN.mkdir(parents=True, exist_ok=True)

    ens_n = int(train_cfg.get("ensemble_n", 1))
    ens_n = max(1, min(ens_n, 7))

    run_seed_base = _ts_seed_base()
    seeds = [run_seed_base + i * 17 for i in range(ens_n)]

    Y_std_full = (Y - mu[:D]) / (sd[:D] + 1e-8)
    seed_std = Y_std_full[-K:]   # seed cuối cùng để rollout

    preds_std_list = []

    pb = st.progress(0.0)

    mu_path = RUN / "mu.npy"
    sd_path = RUN / "sd.npy"

    try:
        np.save(mu_path, mu[:D])
        np.save(sd_path, sd[:D])
    except Exception:
        pass

    mu_use_global, sd_use_global = mu[:D], sd[:D]

    for si, seed_i in enumerate(seeds):
        set_seed(int(seed_i))
        ckpt_path = RUN / f"hybrid_trinet_seed{si}.pt"

        model = HybridTriNet(
            k=K,
            D=D,
            H=H,
            d_feat=96,
            kan_M=8,
            kan_depth=2,
            kan_drop=0.10,
            gru_hidden=128,
            gru_layers=1,
            gru_drop=0.10,
            attn_dmodel=48,
            attn_heads=3,
            attn_layers=2,
            attn_drop=0.05,
            patch_len=16,
            stride=8,
        ).to(device_train)

        loaded = False
        if (not retrain) and ckpt_path.exists() and mu_path.exists() and sd_path.exists():
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device_train))
                _mu = np.load(mu_path).reshape(-1)
                _sd = np.load(sd_path).reshape(-1)
                if _mu.size == D and _sd.size == D:
                    loaded = True
                    mu_use_global, sd_use_global = _mu, _sd
            except Exception:
                loaded = False

        if not loaded:
            def _status_cb(ep, epochs, tr_loss, val_mae, lr_val):
                base = si / max(1, len(seeds))
                pb.progress(min(0.99, base + (ep / max(1, epochs)) / max(1, len(seeds))))

            with st.spinner("Đang dự đoán"):
                model, _best_val = fit_model_better(
                    model=model,
                    tr_loader=tr_ld,
                    va_loader=va_ld,
                    mu=mu_use_global,
                    sd=sd_use_global,
                    epochs=int(train_cfg["epochs"]),
                    lr=float(train_cfg["lr"]),
                    loss_name=str(train_cfg["loss"]),
                    weight_decay=float(train_cfg["wd"]),
                    grad_clip=float(train_cfg["clip"]),
                    patience=int(train_cfg["patience"]),
                    use_amp=bool(train_cfg["amp"]),
                    status_cb=_status_cb,
                    device=device_train,
                    focus_h=5,
                    focus_w=float(train_cfg.get("focus_w", 3.0)),
                )

            try:
                torch.save(model.state_dict(), ckpt_path)
            except Exception:
                pass

        # =========================
        # BLOCK-5 ROLLOUT
        # =========================
        # 5 ngày  -> 1 block
        # 30 ngày -> 6 block
        # 60 ngày -> 12 block
        # 100 ngày -> 20 block
        F_std = roll_autoregressive_safe(
            model,
            seed_std=seed_std,
            H_total=int(h_next),
            H=H,
            device=device_train,
            step_size=5,
        )

        F_std = ensure_F_shape(F_std, int(h_next), D)
        preds_std_list.append(F_std)

    pb.progress(1.0)

    F_std_ens = np.mean(np.stack(preds_std_list, axis=0), axis=0)
    F = F_std_ens * sd_use_global.reshape(1, D) + mu_use_global.reshape(1, D)

    last_date = pd.Timestamp(df[date_col].max()).normalize()
    idx = pd.bdate_range(last_date + BDay(1), periods=int(h_next))

    out = pd.DataFrame(F, index=idx, columns=TARGET_COLS)
    out_to_save = out.reset_index().rename(columns={"index": date_col})[[date_col] + TARGET_COLS].copy()
    
    # calibration (optional)
    calib = {}
    history_dir = _history_dir_for_h(int(h_next))
    hist_files = list(history_dir.glob("forecast_until_*.csv")) if history_dir.exists() else []

    if hist_files and (actual_full is not None):
        try:
            calib = fit_calibration_from_history(
                history_dir=history_dir,
                actual_df=actual_full,
                date_col=date_col,
                target_cols=TARGET_COLS,
                min_points=int(train_cfg.get("calib_min_points", 30)),
                recency_halflife_days=int(train_cfg.get("calib_halflife", 180)),
            )
        except Exception:
            calib = {}

    if calib:
        out_to_save = apply_calibration(out_to_save, calib, TARGET_COLS, keep_raw=True)
        st.caption(
            "Calibration (actual ≈ a*pred + b): "
            + " | ".join([f"{k}: a={v['a']:.4f}, b={v['b']:.4f}, n={v['n']}" for k, v in calib.items()])
        )

    saved_ok = False
    if bool(save_history):
        fname_hist = save_forecast_history(
            out_to_save,
            last_date,
            int(h_next),
            date_col,
            run_seed_base=run_seed_base,
        )
        st.caption(f"Đã lưu forecast_history: {fname_hist.name}")
        saved_ok = True

    return out_to_save, saved_ok


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="Dự đoán giá xăng dầu", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    page_header()

    defs = get_defaults()
    date_col = defs["DATE_COL_CFG"]
    clean_path_str = str(defs["DEFAULT_CLEAN_PATH"])
    DEFAULT_H_NEXT = defs["DEFAULT_H_NEXT"]
    fred_api_key = defs["FRED_API_KEY_DEFAULT"]

    fill_mode = "ffill"

    st.session_state.setdefault("df_merged", None)
    st.session_state.setdefault("pred_df", None)
    st.session_state.setdefault("actual_full", None)
    st.session_state.setdefault("actual_clean", None)
    st.session_state.setdefault("_df_use_for_prev5", None)
    st.session_state.setdefault("run_triggered", False)

    base0 = None
    if clean_path_str and Path(clean_path_str).exists():
        try:
            base0 = pd.read_excel(clean_path_str, engine="openpyxl")
            if date_col in base0.columns:
                base0 = _ensure_date(base0, date_col)
                base0[date_col] = _parse_dates_any(base0[date_col])
                base0 = base0.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        except Exception:
            base0 = None

    # =========================
    # Candlestick preview
    # =========================
    with st.container(border=True):
        section_header("chart-candle", "Biểu đồ nến (dữ liệu)")

        df_plot = st.session_state.get("df_merged")
        if df_plot is None:
            df_plot = base0

        c1, c2, c3 = st.columns([0.40, 0.30, 0.30], vertical_alignment="bottom")
        with c1:
            series = st.selectbox("Chọn series", TARGET_COLS, index=0, key="candle_series")
        with c2:
            last_n = st.number_input("Số ngày hiển thị", 30, 2000, 260, 10, key="candle_lastn")
        with c3:
            show_vol = st.checkbox("Hiển thị volume (proxy)", value=True, key="candle_vol")

        if df_plot is None:
            st.info("Chưa có dữ liệu để vẽ.")
        else:
            plot_candlestick_preview(
                df_plot,
                date_col,
                series,
                last_n=int(last_n),
                show_volume=bool(show_vol),
                title=f"{series} - Candlestick (daily)",
            )

    soft_divider()

    # =========================
    # Forecast controls
    # =========================
    with st.container(border=True):
        section_header("rocket", "Thiết lập dự đoán")

        with st.expander("Cấu hình", expanded=False):
            cc = st.columns(4)
            train_cfg = {
                "batch": cc[0].number_input("Batch", 16, 512, 128, 16),
                "epochs": cc[1].number_input("Epochs", 10, 500, 160, 10),
                "lr": cc[2].number_input("LR", 1e-6, 5e-3, 1e-4, 1e-5, format="%.6f"),
                "loss": cc[3].selectbox("Loss", ["huber", "mae", "mse"], index=0),
                "patience": 30,
                "ensemble_n": 1,
                "wd": 1e-4,
                "clip": 1.0,
                "amp": True,
                "focus_w": 3.0,
                "calib_min_points": 30,
                "calib_halflife": 180,
            }

        cc1, cc2, cc3, cc4 = st.columns([0.40, 0.20, 0.20, 0.20], vertical_alignment="bottom")
        with cc1:
            st.file_uploader("Tải lên price_petroleum (.csv/.xlsx/.xls)", type=["csv", "xlsx", "xls"], key="up_pp_main")
        with cc2:
            st.selectbox(
                "Mốc dự đoán",
                [5, 30, 60, 100],
                index=0,
                key="h_next_main",
                format_func=lambda x: f"{x} ngày",
            )
        with cc3:
            st.checkbox(
                "Lưu forecast_history",
                value=True,
                help="Bật để lưu forecast_until_*.csv (để sau này khi có dữ liệu thực tế overlap thì dashboard sẽ tính metrics).",
                key="save_history_main",
            )
        with cc4:
            st.button("Chạy dự đoán", use_container_width=True, key="run_btn_main", type="primary")

    if st.session_state.get("run_btn_main", False):
        st.session_state.run_triggered = True

    # =========================
    # Run single forecast
    # =========================
    if st.session_state.run_triggered:
        st.session_state.run_triggered = False

        p = Path(clean_path_str)
        if not p.exists():
            st.error(f"Không tìm thấy file gốc: {clean_path_str}")
        else:
            try:
                actual_full_before = _read_actual_full(clean_path_str, date_col)
            except Exception as e:
                st.error(f"Lỗi đọc dữ liệu thực tế (clean.xlsx): {e}")
                actual_full_before = None

            if actual_full_before is not None:
                base_last_before = actual_full_before[date_col].dropna().max()
                base_last_before = pd.Timestamp(base_last_before).normalize() if pd.notna(base_last_before) else pd.NaT

                if st.session_state.get("up_pp_main", None) is None:
                    df_updated = actual_full_before.copy()
                    df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
                    try:
                        df_updated = _interpolate_external(df_updated, date_col)
                    except Exception as e:
                        st.warning(f"Interpolate external lỗi: {e}")
                    upload_last_date = base_last_before
                    actual_full = actual_full_before.copy()
                else:
                    up_pp_obj = st.session_state.get("up_pp_main")
                    try:
                        upload_last_date = _get_upload_last_date(up_pp_obj, date_col)
                    except Exception as e:
                        st.error(f"Lỗi đọc ngày cuối file upload: {e}")
                        upload_last_date = None

                    if upload_last_date is None:
                        df_updated = actual_full_before.copy()
                        df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
                        try:
                            df_updated = _interpolate_external(df_updated, date_col)
                        except Exception as e:
                            st.warning(f"Interpolate external lỗi: {e}")
                        actual_full = actual_full_before.copy()
                    else:
                        upload_last_date = pd.Timestamp(upload_last_date).normalize()
                        is_new_upload = bool(pd.notna(base_last_before) and upload_last_date > base_last_before)

                        if not is_new_upload:
                            df_updated = actual_full_before.copy()
                            df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
                            try:
                                df_updated = _interpolate_external(df_updated, date_col)
                            except Exception as e:
                                st.warning(f"Interpolate external lỗi: {e}")
                            actual_full = actual_full_before.copy()
                        else:
                            df_updated = None
                            build_ok = True
                            try:
                                if fred_api_key:
                                    df_new, base_clean, info = build_merged(up_pp_obj, clean_path_str, date_col, fill_mode, fred_api_key)
                                    if df_new is None or len(df_new) == 0:
                                        df_updated = actual_full_before.copy()
                                    else:
                                        base2, add2, _ = _align_union_columns(actual_full_before, df_new, date_col)
                                        df_updated = merge_keep_nonnull(base2, add2, date_col)
                                else:
                                    build_ok = False
                            except Exception:
                                build_ok = False

                            if (not build_ok) or (df_updated is None):
                                u = _read_upload_file(up_pp_obj)
                                u[date_col] = _parse_dates_any(u[date_col])
                                u = u.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
                                base2, add2, _ = _align_union_columns(actual_full_before, u, date_col)
                                df_updated = merge_keep_nonnull(base2, add2, date_col)

                            df_updated = _apply_fill_mode(df_updated, date_col, fill_mode)
                            try:
                                df_updated = _interpolate_external(df_updated, date_col)
                            except Exception as e:
                                st.warning(f"Interpolate external lỗi: {e}")

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
                                    actual_full = _read_actual_full(clean_path_str, date_col)
                                    st.success(f"Đã cập nhật dữ liệu gộp vào: {clean_path_str}")
                                except Exception as e:
                                    st.error(f"Lỗi lưu file gộp: {e}")
                                    actual_full = actual_full_before.copy()
                            else:
                                actual_full = actual_full_before.copy()

                st.session_state.df_merged = df_updated

                df_use = df_updated.copy()
                df_use[date_col] = _parse_dates_any(df_use[date_col])
                df_use = df_use.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
                if st.session_state.get("up_pp_main", None) is not None and upload_last_date is not None:
                    df_use = df_use[df_use[date_col] <= upload_last_date].copy()

                h_next = int(st.session_state.get("h_next_main", DEFAULT_H_NEXT))
                save_history_flag = bool(st.session_state.get("save_history_main", True))

                st.session_state.actual_full = df_use.copy()
                st.session_state.actual_clean = actual_full.copy() if actual_full is not None else None

                pred_df, _ = run_forecast(
                    df_use,
                    date_col,
                    h_next,
                    save_history=save_history_flag,
                    retrain=True,
                    train_cfg=train_cfg,
                    actual_full=df_use,
                )

                st.session_state.pred_df = pred_df
                st.session_state._df_use_for_prev5 = df_use

    soft_divider()
    with st.container(border=True):
        section_header("table", "Kết quả dự đoán")

        pred_df = st.session_state.get("pred_df", None)
        if pred_df is None or (isinstance(pred_df, pd.DataFrame) and pred_df.empty):
            st.info("Chưa có kết quả dự đoán. Hãy upload file và bấm 'Chạy dự đoán'.")
        else:
            show_cols = [date_col] + TARGET_COLS
            pred_view = pred_df[show_cols].copy() if all(c in pred_df.columns for c in show_cols) else pred_df.copy()

            st.dataframe(pred_view, use_container_width=True, height=280, hide_index=True)
            st.download_button(
                "Tải forecast.csv",
                data=pred_view.to_csv(index=False, date_format="%d/%m/%Y %I:%M:%S %p").encode("utf-8"),
                file_name="forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # =========================
    # Compare with actual + Backfill theo chu kỳ H
    # =========================
    with st.container(border=True):
        section_header("clipboard-check", "So sánh dự đoán với thực tế")

        root_xlsx = None
        candidates = [
            Path.cwd() / "base" / "root.xlsx",
            Path(clean_path_str).with_name("root.xlsx"),
            Path(clean_path_str).resolve().parent / "root.xlsx",
            Path.cwd() / "root.xlsx",
        ]
        for pth in candidates:
            if pth.exists():
                root_xlsx = pth
                break

        if root_xlsx is None:
            st.error("Không tìm thấy root.xlsx. Hãy đặt file tại base/root.xlsx hoặc cùng thư mục với clean.xlsx.")
            return

        # -------- Evaluate/plot --------
        h_eval = st.selectbox("Kịch bản hiển thị (H)", [5, 30, 60, 100], index=0, key="hist_h_eval")
        hist_dir = _history_dir_for_h(int(h_eval))
        if not hist_dir.exists():
            st.info("Chưa có thư mục forecast_history cho H này.")
            return

        # đọc history bằng loader robust
        history_long = _load_history_long_from_dir(hist_dir, targets=list(TARGET_COLS), date_col_hint=date_col)
        if history_long is None or history_long.empty:
            st.warning("Chưa có forecast_history (hoặc không đọc được file forecast_until_*.csv).")
            return

        actual_wide = load_actual_root(root_xlsx, targets=TARGET_COLS)
        if actual_wide is None or actual_wide.empty:
            st.warning("Không đọc được dữ liệu thực tế từ root.xlsx.")
            return

        history_use = history_long.copy()
        history_use["date"] = pd.to_datetime(history_use["date"], errors="coerce", dayfirst=True).dt.normalize()
        history_use = history_use[history_use["date"] >= START0].copy()

        actual_wide = actual_wide.copy()
        actual_wide["date"] = pd.to_datetime(actual_wide["date"], errors="coerce", dayfirst=True).dt.normalize()
        actual_wide = actual_wide[actual_wide["date"] >= START0].copy()

        act_targets = [c for c in actual_wide.columns if c != "date"]
        act_long = actual_wide.melt(id_vars=["date"], value_vars=act_targets, var_name="target", value_name="actual")
        act_long["actual"] = pd.to_numeric(act_long["actual"], errors="coerce")
        act_long = act_long.dropna(subset=["date"]).copy()

        # ====== chọn chế độ dùng history ======
        hist_mode = st.selectbox(
            "Cách dùng forecast_history để tính metrics",
            [
                "Trung bình theo (date,target) từ tất cả file",
                "Tính trên mọi dòng (đếm overlap)",
                "Chỉ dùng forecast mới nhất (latest)",
                "Chọn forecast tốt nhất theo actual (ORACLE - upper bound)",
            ],
            index=0,
            key="hist_mode_eval",
            help=(
                "• Trung bình: nếu 1 ngày được dự báo nhiều lần, lấy mean(pred) rồi mới so với actual.\n"
                "• Đếm overlap: mỗi lần dự báo là 1 điểm -> ngày trùng sẽ được tính nhiều lần.\n"
                "• Latest: giữ đúng 1 forecast mới nhất cho mỗi (date,target).\n"
                "• ORACLE: chọn dự báo có sai số nhỏ nhất so với actual cho mỗi (date,target) (chỉ để tham khảo trần trên)."
            ),
        )

        # ====== build hist theo mode ======
        if hist_mode.startswith("Chọn forecast tốt nhất"):
            # dùng latest để hiển thị full timeline (cả future), rồi overwrite bằng oracle cho ngày có actual
            hist_latest_full = latest_asof_per_date_target(history_use)
            hist_candidates = _filter_valid_forecast(history_use)  # tất cả dòng hợp lệ làm ứng viên

            # merge candidates với actual để tính lỗi
            cand_cmp = hist_candidates.rename(columns={"yhat": "pred"}).merge(
                act_long[["date", "target", "actual"]],
                on=["date", "target"],
                how="inner",
            )
            cand_cmp["pred"] = pd.to_numeric(cand_cmp["pred"], errors="coerce")
            cand_cmp["actual"] = pd.to_numeric(cand_cmp["actual"], errors="coerce")
            cand_cmp = cand_cmp.dropna(subset=["pred", "actual"]).copy()

            if cand_cmp.empty:
                # fallback: nếu không có overlap thì vẫn dùng latest
                hist = hist_latest_full.copy()
                hist["_picked_mode"] = "LATEST_FALLBACK"
            else:
                cand_cmp["ae"] = (cand_cmp["pred"] - cand_cmp["actual"]).abs()
                idx = cand_cmp.groupby(["date", "target"])["ae"].idxmin()
                best = cand_cmp.loc[idx].drop(columns=["ae"]).copy()

                # best đang có pred, actual; đổi lại yhat
                keep_cols = ["date", "target", "pred"]
                meta_cols = [c for c in ["train_last_date", "generated_at", "run_seed_base", "source_file"] if c in best.columns]
                best2 = best[keep_cols + meta_cols].rename(columns={"pred": "yhat"}).copy()

                # overwrite latest_full theo key (date,target)
                hist_latest_full = hist_latest_full.copy()
                hist_latest_full["_key"] = hist_latest_full["date"].astype(str) + "||" + hist_latest_full["target"].astype(str)
                best2["_key"] = best2["date"].astype(str) + "||" + best2["target"].astype(str)

                best_map = best2.set_index("_key")
                hist_map = hist_latest_full.set_index("_key")

                # update yhat + metadata nếu có
                for col in ["yhat"] + meta_cols:
                    if col in best_map.columns and col in hist_map.columns:
                        hist_map.loc[best_map.index.intersection(hist_map.index), col] = best_map.loc[
                            best_map.index.intersection(hist_map.index), col
                        ]

                hist = hist_map.reset_index(drop=False).drop(columns=["_key"], errors="ignore")
                hist["_picked_mode"] = "ORACLE_WHERE_POSSIBLE"

        elif hist_mode.startswith("Trung bình"):
            hist = _agg_mean_by_date_target(history_use)
            hist["_picked_mode"] = "MEAN_DATE_TARGET"
        elif hist_mode.startswith("Tính trên mọi dòng"):
            hist = _filter_valid_forecast(history_use)
            hist["_picked_mode"] = "ALL_ROWS_OVERLAP"
        else:
            hist = latest_asof_per_date_target(history_use)
            hist["_picked_mode"] = "LATEST_ONLY"

        if hist is None or hist.empty:
            st.warning("History sau khi lọc/gộp đang rỗng.")
            return

        # merge lịch sử dự đoán với actual
        cmp_base = hist.rename(columns={"yhat": "pred"}).merge(
            act_long[["date", "target", "actual"]],
            on=["date", "target"],
            how="left",
        )
        cmp_base["pred"] = pd.to_numeric(cmp_base["pred"], errors="coerce")
        cmp_base["actual"] = pd.to_numeric(cmp_base["actual"], errors="coerce")
        cmp_base = cmp_base[cmp_base["target"].isin(list(TARGET_COLS))].copy()
        cmp_base = cmp_base[cmp_base["date"] >= START0].copy()
        cmp_base = cmp_base.sort_values(["target", "date"]).reset_index(drop=True)

        cmp_eval = cmp_base.dropna(subset=["actual", "pred"]).copy()

        section_header("calculator", "Số liệu")
        if cmp_eval.empty:
            st.warning("Chưa có overlap để tính metrics")
        else:
            met = compute_metrics(cmp_eval[["date", "target", "actual", "pred"]])
            st.dataframe(met, use_container_width=True, hide_index=True)

            st.caption(f"Matched points: {len(cmp_eval):,} | Mode={hist.get('_picked_mode', 'N/A').iloc[0] if '_picked_mode' in hist.columns and len(hist)>0 else 'N/A'}")

        # ====== Bảng so sánh theo target ======
        section_header("table", "Bảng so sánh theo target")

        show_compact = True
        if hist_mode.startswith("Tính trên mọi dòng"):
            show_compact = st.checkbox("Gộp theo (date,target) để xem gọn (table/plot)", value=True, key="cmp_compact")

        cmp_for_table = cmp_base.copy()
        if hist_mode.startswith("Tính trên mọi dòng") and show_compact:
            agg = (
                cmp_base.groupby(["date", "target"], as_index=False)
                        .agg(
                            actual=("actual", "first"),
                            pred=("pred", "mean"),
                            n_forecasts=("pred", "size"),
                        )
            )
            cmp_for_table = agg

        tabs_tbl = st.tabs(list(TARGET_COLS))
        for tab, t in zip(tabs_tbl, list(TARGET_COLS)):
            with tab:
                dd = cmp_for_table[cmp_for_table["target"] == t].copy().sort_values("date").reset_index(drop=True)
                dd["actual"] = pd.to_numeric(dd["actual"], errors="coerce")
                dd["pred"] = pd.to_numeric(dd["pred"], errors="coerce")

                keep_tbl = [c for c in ["date", "actual", "pred", "n_forecasts", "train_last_date", "generated_at"] if c in dd.columns]
                st.dataframe(dd[keep_tbl], use_container_width=True, height=360, hide_index=True)

        # overlay lines (30/60/100) - latest_asof per date,target
        overlay_hs = st.multiselect(
            "Biểu đồ đường dự đoán (overlay)",
            [30, 60, 100],
            default=[30, 60, 100],
            key="overlay_hs",
        )

        overlay_preds: Dict[int, pd.DataFrame] = {}
        for hh in overlay_hs:
            hh = int(hh)
            hdir = _history_dir_for_h(hh)
            if not hdir.exists():
                continue
            ph = _latest_pred_long_for_dir(hdir, list(TARGET_COLS), date_col_hint=date_col)
            if ph is None or ph.empty:
                continue
            ph = ph.copy()
            ph["date"] = pd.to_datetime(ph["date"], errors="coerce", dayfirst=True).dt.normalize()
            ph = ph[ph["date"] >= START0].copy()
            overlay_preds[hh] = ph.rename(columns={"yhat": f"pred_H{hh}"})[["date", "target", f"pred_H{hh}"]]

        date_candidates = []
        date_candidates.append(pd.to_datetime(act_long["date"], errors="coerce"))
        date_candidates.append(pd.to_datetime(cmp_base["date"], errors="coerce"))
        for dfh in overlay_preds.values():
            date_candidates.append(pd.to_datetime(dfh["date"], errors="coerce"))

        date_candidates = [s.dropna() for s in date_candidates if s is not None and len(s.dropna()) > 0]
        if not date_candidates:
            st.warning("Không có dữ liệu để vẽ.")
            return

        umax = max(s.max() for s in date_candidates)
        ud1 = START0
        ud2 = pd.Timestamp(umax).normalize()

        section_header("chart-line", "Biểu đồ thực tế với Dự đoán")

        # dùng bản gọn cho plot để tránh “răng cưa” khi overlap
        cmp_for_plot = cmp_base.copy()
        if hist_mode.startswith("Tính trên mọi dòng"):
            cmp_for_plot = (
                cmp_base.groupby(["date", "target"], as_index=False)
                        .agg(actual=("actual", "first"), pred=("pred", "mean"))
            )

        tabs = st.tabs(list(TARGET_COLS))
        for tab, t in zip(tabs, list(TARGET_COLS)):
            with tab:
                frames = []

                a = act_long[act_long["target"] == t][["date", "actual"]].copy()
                a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
                a["actual"] = pd.to_numeric(a["actual"], errors="coerce")
                a = a.rename(columns={"actual": "value"})
                a["series"] = "actual"
                frames.append(a)

                b = cmp_for_plot[cmp_for_plot["target"] == t][["date", "pred"]].copy()
                b["date"] = pd.to_datetime(b["date"], errors="coerce", dayfirst=True).dt.normalize()
                b["pred"] = pd.to_numeric(b["pred"], errors="coerce")
                b = b.rename(columns={"pred": "value"})
                b["series"] = f"pred (H={int(h_eval)})"
                frames.append(b)

                for hh, dfh in overlay_preds.items():
                    col = f"pred_H{hh}"
                    c = dfh[dfh["target"] == t][["date", col]].copy()
                    c["date"] = pd.to_datetime(c["date"], errors="coerce", dayfirst=True).dt.normalize()
                    c[col] = pd.to_numeric(c[col], errors="coerce")
                    c = c.rename(columns={col: "value"})
                    c["series"] = f"pred (H={hh})"
                    frames.append(c)

                plot_df = pd.concat(frames, ignore_index=True)
                plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce", dayfirst=True).dt.normalize()
                plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
                plot_df = plot_df.dropna(subset=["date", "value"])
                plot_df = plot_df[(plot_df["date"] >= ud1) & (plot_df["date"] <= ud2)].copy()

                if plot_df.empty:
                    st.info("Không có dữ liệu trong khoảng ngày đã chọn.")
                    continue

                order = ["actual", f"pred (H={int(h_eval)})"] + [f"pred (H={hh})" for hh in sorted(list(overlay_preds.keys()))]
                dash_domain = [s for s in order if s in plot_df["series"].unique().tolist()]
                dash_range = [[1, 0]] + [[6, 3]] * (len(dash_domain) - 1)
                dash_scale = alt.Scale(domain=dash_domain, range=dash_range)

                ch = (
                    alt.Chart(plot_df)
                    .mark_line(point=False, strokeWidth=2)
                    .encode(
                        x=alt.X("date:T", title=None, axis=alt.Axis(labelAngle=-20)),
                        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                        color=alt.Color("series:N", legend=alt.Legend(title=None, orient="top")),
                        strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("value:Q", title="Value", format=".4f"),
                        ],
                    )
                    .properties(title=t, height=340)
                    .interactive()
                    .configure_view(stroke=None)
                    .configure_axis(grid=True)
                )

                # FIX Streamlit altair
                st.altair_chart(ch, use_container_width=True)


if __name__ == "__main__":
    main()