from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
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
    s = int(datetime.now().strftime("%Y%m%d%H%M%S%f"))
    return int(s % 2_147_483_647)


def _history_dir_for_h(h: int) -> Path:
    base_dir = RUN_OUTPUT_DIR / "forecast_history"
    base_dir.mkdir(parents=True, exist_ok=True)

    h = int(h)
    h_dir = base_dir / str(h)
    h_dir.mkdir(parents=True, exist_ok=True)

    return h_dir


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
        df["_asof_sort"] = df["_asof_sort"].where(df["_asof_sort"].notna(), pd.to_datetime(df.get("generated_at"), errors="coerce", dayfirst=True))
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
                retrain=True,   # backtest đúng
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

    # ---- targets (price)
    try:
        Y_price = _coerce_targets_numeric(df, TARGET_COLS).astype(np.float32)
    except Exception as e:
        st.error(f"Lỗi chuyển target sang số: {e}")
        return None, False

    T, D_out = Y_price.shape
    if D_out != len(TARGET_COLS):
        st.error(f"D_out mismatch: got {D_out}, expected {len(TARGET_COLS)}")
        return None, False

    # overwrite targets in df as numeric
    for j, c in enumerate(TARGET_COLS):
        df[c] = Y_price[:, j]

    # ---- infer feature cols (all numeric-ish) and coerce
    def _coerce_numeric_series(s: pd.Series):
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        return pd.to_numeric(s, errors="coerce")

    feature_cols = []
    for c in df.columns:
        if c == date_col:
            continue
        if c in TARGET_COLS:
            continue
        x = _coerce_numeric_series(df[c])
        ok_ratio = float(np.isfinite(x.to_numpy(dtype=float)).mean())
        if ok_ratio >= 0.80:
            df[c] = x
            feature_cols.append(c)

    # always include targets as features
    for c in TARGET_COLS:
        if c not in feature_cols:
            feature_cols.append(c)

    # fill numeric missing
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()

    D_in = len(feature_cols)

    # ---- delta targets
    Y_delta = np.zeros_like(Y_price, dtype=np.float32)
    Y_delta[1:] = Y_price[1:] - Y_price[:-1]

    if T <= K + H + 2:
        st.error(f"Dữ liệu quá ngắn (T={T}) so với K={K}, H={H}.")
        return None, False

    val_len = max(int(T * VAL_RATIO), K + H + 1)
    train_len = T - val_len

    X_raw = df[feature_cols].values.astype(np.float32)

    X_tr = X_raw[:train_len]
    d_tr = Y_delta[:train_len]

    # standardize X and delta
    Xtr_std, x_mu, x_sd = standardize(X_tr)
    dtr_std, y_mu, y_sd = standardize(d_tr)

    X_std_full = (X_raw - x_mu.reshape(1, -1)) / (x_sd.reshape(1, -1) + 1e-8)
    d_std_full = (Y_delta - y_mu.reshape(1, -1)) / (y_sd.reshape(1, -1) + 1e-8)

    def build_windows_xy(Xs: np.ndarray, Ys: np.ndarray, K_: int, H_: int):
        Xs = np.asarray(Xs, dtype=np.float32)
        Ys = np.asarray(Ys, dtype=np.float32)
        if Xs.ndim != 2 or Ys.ndim != 2:
            raise ValueError(f"build_windows_xy expects 2D arrays, got X={Xs.shape}, Y={Ys.shape}")
        if Xs.shape[0] != Ys.shape[0]:
            raise ValueError(f"X/Y length mismatch: {Xs.shape[0]} vs {Ys.shape[0]}")
        Tt = Xs.shape[0]
        N = Tt - int(K_) - int(H_) + 1
        if N <= 0:
            raise ValueError(f"Not enough data for windows: T={Tt}, K={K_}, H={H_} => N={N}")
        Xw = np.empty((N, int(K_), Xs.shape[1]), dtype=np.float32)
        Yw = np.empty((N, int(H_), Ys.shape[1]), dtype=np.float32)
        for i in range(N):
            Xw[i] = Xs[i : i + int(K_)]
            Yw[i] = Ys[i + int(K_) : i + int(K_) + int(H_)]
        return Xw, Yw

    Xtr_w, Ytr_w = build_windows_xy(X_std_full[:train_len], d_std_full[:train_len], K, H)
    Xva_w, Yva_w = build_windows_xy(X_std_full[train_len - K :], d_std_full[train_len - K :], K, H)

    tr_ld = DataLoader(WindowDS(Xtr_w, Ytr_w), batch_size=int(train_cfg["batch"]), shuffle=True)
    va_ld = DataLoader(WindowDS(Xva_w, Yva_w), batch_size=int(train_cfg["batch"]), shuffle=False)

    RUN = RUN_OUTPUT_DIR
    RUN.mkdir(parents=True, exist_ok=True)

    # files for reuse
    feat_path = RUN / "feature_cols.json"
    xmu_path = RUN / "x_mu.npy"
    xsd_path = RUN / "x_sd.npy"
    ymu_path = RUN / "y_mu.npy"
    ysd_path = RUN / "y_sd.npy"
    alpha_path = RUN / "blend_alpha.json"

    ens_n = int(train_cfg.get("ensemble_n", 1))
    ens_n = max(1, min(ens_n, 7))

    run_seed_base = _ts_seed_base()
    seeds = [run_seed_base + i * 17 for i in range(ens_n)]

    tgt_idx = [feature_cols.index(c) for c in TARGET_COLS]

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

    def _apply_calib_vec(v_raw: np.ndarray) -> np.ndarray:
        v = np.asarray(v_raw, dtype=np.float32).copy()
        if not calib:
            return v
        for j, c in enumerate(TARGET_COLS):
            if c not in calib:
                continue
            a = float(calib[c].get("a", 1.0))
            b = float(calib[c].get("b", 0.0))
            v[j] = a * v[j] + b
        return v

    preds_all = []
    pb = st.progress(0.0)

    # save scalers/cols
    try:
        feat_path.write_text(
            json.dumps({"feature_cols": feature_cols, "tgt_idx": tgt_idx, "K": int(K), "H": int(H)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(xmu_path, x_mu.astype(np.float32))
        np.save(xsd_path, x_sd.astype(np.float32))
        np.save(ymu_path, y_mu.astype(np.float32))
        np.save(ysd_path, y_sd.astype(np.float32))
    except Exception:
        pass

    alpha = None
    if alpha_path.exists() and (not retrain):
        try:
            obj = json.loads(alpha_path.read_text(encoding="utf-8"))
            alpha = np.asarray(obj.get("alpha", None), dtype=np.float32)
            if alpha.size != D_out:
                alpha = None
        except Exception:
            alpha = None

    for si, seed_i in enumerate(seeds):
        set_seed(int(seed_i))
        ckpt_path = RUN / f"hybrid_trinet_seed{si}.pt"

        model = HybridTriNet(
            k=K,
            H=H,
            D_in=D_in,
            D_out=D_out,
            d_feat=96,
            kan_M=8,
            kan_depth=2,
            kan_drop=0.10,
            gru_hidden=128,
            gru_layers=1,
            gru_drop=0.10,
            attn_dmodel=64,
            attn_heads=4,
            attn_layers=2,
            attn_drop=0.05,
            patch_len=16,
            stride=8,
        ).to(device_train)
        loaded = False
        if (not retrain) and ckpt_path.exists():
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device_train))
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            def _status_cb(ep, epochs, tr_loss, val_mae, lr_val):
                base = si / max(1, len(seeds))
                pb.progress(min(0.99, base + (ep / max(1, epochs)) / max(1, len(seeds))))

            with st.spinner("Đang dự đoán với mô hình HybridTriNet..."):
                model, _best_val = fit_model_better(
                    model=model,
                    tr_loader=tr_ld,
                    va_loader=va_ld,
                    mu=y_mu,
                    sd=y_sd,
                    epochs=int(train_cfg["epochs"]),
                    lr=float(train_cfg["lr"]),
                    loss_name=str(train_cfg["loss"]),
                    weight_decay=float(train_cfg["wd"]),
                    grad_clip=float(train_cfg["clip"]),
                    patience=int(train_cfg["patience"]),
                    use_amp=bool(train_cfg["amp"]),
                    status_cb=_status_cb,
                    device=device_train,
                    focus_h=int(train_cfg.get("focus_h", 5)),
                    focus_w=float(train_cfg.get("focus_w", 3.0)),
                    use_delta_price_loss=True,
                    tgt_idx=tgt_idx,
                    x_mu=x_mu,
                    x_sd=x_sd,
                    y_mu=y_mu,
                    y_sd=y_sd,
                    alpha_delta=float(train_cfg.get("alpha_delta", 0.20)),
                    beta_price=float(train_cfg.get("beta_price", 1.00)),
                    eps_mape=float(train_cfg.get("eps_mape", 1e-3)),
                )

            try:
                torch.save(model.state_dict(), ckpt_path)
            except Exception:
                pass

        # fit alpha on val (once)
        if alpha is None:
            alpha = np.ones((D_out,), dtype=np.float32)
            P, N, Tt = [], [], []
            model.eval()
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb = xb.to(device_train)
                    yb = yb.to(device_train)
                    out = model(xb)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out = out.view(out.size(0), int(H), int(D_out))

                    pred_d1 = out[:, 0, :].detach().cpu().numpy() * y_sd.reshape(1, -1) + y_mu.reshape(1, -1)
                    true_d1 = yb[:, 0, :].detach().cpu().numpy() * y_sd.reshape(1, -1) + y_mu.reshape(1, -1)

                    last_price = xb[:, -1, tgt_idx].detach().cpu().numpy() * x_sd[tgt_idx].reshape(1, -1) + x_mu[tgt_idx].reshape(1, -1)
                    pred1 = last_price + pred_d1
                    true1 = last_price + true_d1

                    P.append(pred1); N.append(last_price); Tt.append(true1)

            if P:
                P = np.concatenate(P, axis=0)
                N = np.concatenate(N, axis=0)
                Tt = np.concatenate(Tt, axis=0)
                grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
                for j in range(D_out):
                    best_a, best_mae = 1.0, 1e18
                    for a in grid:
                        pred_bl = a * P[:, j] + (1.0 - a) * N[:, j]
                        mae = float(np.mean(np.abs(pred_bl - Tt[:, j])))
                        if mae < best_mae:
                            best_mae, best_a = mae, float(a)
                    alpha[j] = best_a

                try:
                    alpha_path.write_text(json.dumps({"alpha": alpha.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

        # forecast roll (exog-aware)
        hist = df.copy().reset_index(drop=True)
        last_date = pd.Timestamp(hist[date_col].max()).normalize()
        fut_dates = pd.bdate_range(last_date + BDay(1), periods=int(h_next))

        prev_price = hist.iloc[-1][TARGET_COLS].to_numpy(dtype=np.float32)
        pred_rows = []
        done = 0

        def _update_calendar_row(row: pd.Series, d: pd.Timestamp):
            dow = int(pd.Timestamp(d).weekday())
            if "dow" in row.index: row["dow"] = float(dow)
            if "weekday" in row.index: row["weekday"] = float(dow)
            if "month" in row.index: row["month"] = float(d.month)
            if "year" in row.index: row["year"] = float(d.year)
            if "dom" in row.index: row["dom"] = float(d.day)
            if "NgayTrongTuan" in row.index: row["NgayTrongTuan"] = float(dow + 1)
            if "ThangTrongNam" in row.index: row["ThangTrongNam"] = float(d.month)
            if "QuyTrongNam" in row.index: row["QuyTrongNam"] = float(((d.month - 1) // 3) + 1)
            if "Nam" in row.index: row["Nam"] = float(d.year)

        while done < int(h_next):
            x_win_raw = hist[feature_cols].iloc[-K:].to_numpy(dtype=np.float32)
            x_win_std = (x_win_raw - x_mu.reshape(1, -1)) / (x_sd.reshape(1, -1) + 1e-8)

            xb = torch.tensor(x_win_std, dtype=torch.float32, device=device_train).unsqueeze(0)
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.view(out.size(0), int(H), int(D_out))

            out_std = out.squeeze(0).detach().cpu().numpy()
            pred_d_raw = out_std * y_sd.reshape(1, -1) + y_mu.reshape(1, -1)

            take = min(int(H), int(h_next) - done)
            for j in range(take):
                d = pd.Timestamp(fut_dates[done + j]).normalize()
                raw = prev_price + pred_d_raw[j]
                cal = _apply_calib_vec(raw)
                final = alpha * cal + (1.0 - alpha) * prev_price

                row = {date_col: d}
                for k, c in enumerate(TARGET_COLS):
                    row[f"{c}_raw"] = float(raw[k])
                    row[f"{c}_cal"] = float(cal[k])
                    row[c] = float(final[k])
                pred_rows.append(row)

                new_row = hist.iloc[-1].copy()
                new_row[date_col] = d
                _update_calendar_row(new_row, d)
                for sc in ["news_abnormal", "impact_score", "News_Abnormal", "Impact_Score"]:
                    if sc in new_row.index:
                        new_row[sc] = 0.0
                for k, c in enumerate(TARGET_COLS):
                    new_row[c] = float(final[k])
                hist = pd.concat([hist, new_row.to_frame().T], ignore_index=True)

                prev_price = final.astype(np.float32)

            done += take

        out_df = pd.DataFrame(pred_rows)
        preds_all.append(out_df)

        pb.progress(min(1.0, (si + 1) / max(1, len(seeds))))

    pb.progress(1.0)

    # ensemble average on FINAL columns
    if len(preds_all) == 1:
        out_to_save = preds_all[0].copy()
    else:
        base = preds_all[0][[date_col] + [f"{c}_raw" for c in TARGET_COLS] + [f"{c}_cal" for c in TARGET_COLS]].copy()
        finals = [dff[TARGET_COLS].to_numpy(dtype=np.float32) for dff in preds_all]
        F = np.mean(np.stack(finals, axis=0), axis=0)
        out_to_save = base
        for j, c in enumerate(TARGET_COLS):
            out_to_save[c] = F[:, j]

    if calib:
        st.caption(
            "Calibration (actual ≈ a*pred + b): "
            + " | ".join([f"{k}: a={v['a']:.4f}, b={v['b']:.4f}, n={v['n']}" for k, v in calib.items()])
        )
    st.caption("Blend alpha: " + ", ".join([f"{c}={alpha[i]:.2f}" for i, c in enumerate(TARGET_COLS)]))

    saved_ok = False
    if bool(save_history):
        last_date = pd.Timestamp(df[date_col].max()).normalize()
        fname_hist = save_forecast_history(out_to_save, last_date, int(h_next), date_col, run_seed_base=run_seed_base)
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
            default_batch = defs["DEFAULT_BATCH_GPU"] if torch.cuda.is_available() else defs["DEFAULT_BATCH_CPU"]
            loss_options = ["huber", "mae", "mse"]
            loss_index = loss_options.index(defs["DEFAULT_LOSS"]) if defs["DEFAULT_LOSS"] in loss_options else 0
            train_cfg = {
                "batch": cc[0].number_input("Batch", 16, 512, int(default_batch), 16),
                "epochs": cc[1].number_input("Epochs", 10, 500, int(defs["DEFAULT_EPOCHS"]), 10),
                "lr": cc[2].number_input("LR", 1e-6, 5e-3, float(defs["DEFAULT_LR"]), 1e-5, format="%.6f"),
                "loss": cc[3].selectbox("Loss", loss_options, index=loss_index),

                "patience": int(defs["DEFAULT_PATIENCE"]),
                "ensemble_n": int(defs["DEFAULT_ENSEMBLE_N"]),
                "wd": float(defs["DEFAULT_WD"]),
                "clip": float(defs["DEFAULT_CLIP"]),
                "amp": bool(defs["DEFAULT_AMP"]),

                "focus_h": int(defs["DEFAULT_FOCUS_H"]),
                "focus_w": float(defs["DEFAULT_FOCUS_W"]),

                "alpha_delta": float(defs["DEFAULT_ALPHA_DELTA"]),
                "beta_price": float(defs["DEFAULT_BETA_PRICE"]),
                "eps_mape": float(defs["DEFAULT_EPS_MAPE"]),

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
            st.button("Chạy dự đoán", width="stretch", key="run_btn_main", type="primary")

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

        # # -------- Backfill theo chu kỳ H (KHÔNG CẦN UPLOAD) --------
        # with st.expander("Backfill theo chu kỳ H (30/60/100): mỗi H ngày dự đoán 1 lần đến nay (không cần upload)", expanded=False):
        #     try:
        #         df_full_train = _read_actual_full(clean_path_str, date_col)
        #         df_full_train = df_full_train.copy()
        #         df_full_train[date_col] = _parse_dates_any(df_full_train[date_col])
        #         df_full_train = df_full_train.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        #     except Exception as e:
        #         df_full_train = None
        #         st.error(f"Lỗi đọc clean.xlsx để backfill: {e}")

        #     if df_full_train is None or df_full_train.empty:
        #         st.warning("Không có dữ liệu clean.xlsx để backfill.")
        #     else:
        #         dmin = pd.Timestamp(df_full_train[date_col].min()).normalize()
        #         dmax = pd.Timestamp(df_full_train[date_col].max()).normalize()

        #         c1, c2 = st.columns([0.5, 0.5], vertical_alignment="bottom")
        #         with c1:
        #             bf_start = st.date_input(
        #                 "Start asof",
        #                 value=max(START0, dmin).date(),
        #                 min_value=dmin.date(),
        #                 max_value=dmax.date(),
        #                 key="bf_period_start",
        #             )
        #         with c2:
        #             bf_end = st.date_input(
        #                 "End asof (đến nay)",
        #                 value=dmax.date(),
        #                 min_value=dmin.date(),
        #                 max_value=dmax.date(),
        #                 key="bf_period_end",
        #             )

        #         bf_horizons = st.multiselect(
        #             "Chọn H cần backfill (mỗi H ngày chạy 1 lần)",
        #             [30, 60, 100],
        #             default=[30, 100],
        #             key="bf_period_horizons",
        #         )

        #         st.caption(
        #             "Số file history hiện có: "
        #             + " | ".join([f"H={hh}: {_count_history_files(_history_dir_for_h(int(hh)))}" for hh in [30, 60, 100]])
        #         )

        #         with st.expander("Cấu hình train cho backfill (khuyên dùng nhẹ)", expanded=False):
        #             ccc = st.columns(3)
        #             train_cfg_bf = dict(train_cfg)
        #             train_cfg_bf["epochs"] = ccc[0].number_input("Epochs (backfill)", 5, 200, 30, 5, key="bf_epochs")
        #             train_cfg_bf["batch"] = ccc[1].number_input("Batch (backfill)", 16, 512, int(train_cfg_bf["batch"]), 16, key="bf_batch")
        #             train_cfg_bf["lr"] = ccc[2].number_input("LR (backfill)", 1e-6, 5e-3, float(train_cfg_bf["lr"]), 1e-5, format="%.6f", key="bf_lr")

        #         if st.button("Chạy backfill theo chu kỳ H", type="primary", key="bf_period_run"):
        #             done, total = backfill_by_horizon_period(
        #                 df_full=df_full_train,
        #                 date_col=date_col,
        #                 start_date=pd.Timestamp(bf_start).normalize(),
        #                 end_date=pd.Timestamp(bf_end).normalize(),
        #                 horizons=[int(x) for x in bf_horizons],
        #                 train_cfg=train_cfg_bf,
        #             )
        #             st.success(f"Backfill xong: {done}/{total} lượt.")

        # -------- Evaluate/plot --------
        h_eval = st.selectbox("Kịch bản hiển thị (H)", [5, 30, 60, 100], index=0, key="hist_h_eval")
        hist_dir = _history_dir_for_h(int(h_eval))
        if not hist_dir.exists():
            st.info("Chưa có thư mục forecast_history cho H này.")
            return

        # đọc history bằng loader robust (không phụ thuộc format trong history_eval.py)
        history_long = _load_history_long_from_dir(hist_dir, targets=list(TARGET_COLS), date_col_hint=date_col)
        if history_long is None or history_long.empty:
            st.warning("Chưa có lịch sử")
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

        hist = latest_asof_per_date_target(history_use)

        cmp_base = hist.rename(columns={"yhat": "pred"}).merge(
            act_long[["date", "target", "actual"]],
            on=["date", "target"],
            how="left",
        )
        cmp_base["pred"] = pd.to_numeric(cmp_base["pred"], errors="coerce")
        cmp_base = cmp_base[cmp_base["target"].isin(list(TARGET_COLS))].copy()
        cmp_base = cmp_base[cmp_base["date"] >= START0].copy()
        cmp_base = cmp_base.sort_values(["target", "date"]).reset_index(drop=True)

        cmp_eval = cmp_base.dropna(subset=["actual", "pred"]).copy()

        section_header("calculator", "Số liệu ")
        if cmp_eval.empty:
            st.warning("Chưa có overlap để tính metrics")
        else:
            met = compute_metrics(cmp_eval[["date", "target", "actual", "pred"]])
            st.dataframe(met, width="stretch", hide_index=True)

        # show_only_actual = st.checkbox("Chỉ hiển thị dòng có actual", value=False, key="show_only_actual_tbl")

        section_header("table", "Bảng so sánh theo target")
        tabs_tbl = st.tabs(list(TARGET_COLS))
        for tab, t in zip(tabs_tbl, list(TARGET_COLS)):
            with tab:
                dd = cmp_base[cmp_base["target"] == t].copy().sort_values("date").reset_index(drop=True)
                dd["actual"] = pd.to_numeric(dd["actual"], errors="coerce")
                dd["pred"] = pd.to_numeric(dd["pred"], errors="coerce")
                # if show_only_actual:
                #     dd = dd.dropna(subset=["actual"])
                keep_tbl = [c for c in ["date", "actual", "pred", "train_last_date", "generated_at"] if c in dd.columns]
                st.dataframe(dd[keep_tbl], width="stretch", height=360, hide_index=True)

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

                b = cmp_base[cmp_base["target"] == t][["date", "pred"]].copy()
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

                st.altair_chart(ch, width="stretch")


if __name__ == "__main__":
    main()