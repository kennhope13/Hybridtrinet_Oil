# forecast_app/core.py
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader
from pandas.tseries.offsets import BDay

from .config import TARGET_COLS, K, H, VAL_RATIO, SEED, get_defaults
from .style import inject_css
from .ui import page_header, section_header, soft_divider, stat_card, ti
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
from .metrics import pred5_vs_prev5_metrics_table, _wavg, _r2_global_from_stats
from .history_eval import (
    _history_signature,
    _cached_eval_history,
    history_line_chart,
    history_rank_bar,
)

from src.dataio import build_merged, _ensure_date, _align_union_columns
from src.features import _coerce_targets_numeric
from src.model.hybrid_trinet import HybridTriNet
from src.model.training import (
    set_seed,
    standardize,
    build_windows,
    WindowDS,
)
from src.utils.paths import RUN_OUTPUT_DIR


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
    Y_tr, Y_val = Y[:train_len], Y[train_len - K :]

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
    seeds = [SEED + i * 17 for i in range(ens_n)]

    Y_std_full = (Y - mu[:D]) / (sd[:D] + 1e-8)
    seed_std = Y_std_full[-K:]

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
        ckpt_path = RUN / f"hybrid_trinet_seed{seed_i}.pt"

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

            with st.spinner(
                f"Đang dự đoán"
            ):
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

        F_std = roll_autoregressive_safe(model, seed_std=seed_std, H_total=int(h_next), H=H, device=device_train)
        F_std = ensure_F_shape(F_std, int(h_next), D)
        preds_std_list.append(F_std)

    pb.progress(1.0)

    F_std_ens = np.mean(np.stack(preds_std_list, axis=0), axis=0)  # (h_next, D)
    F = F_std_ens * sd_use_global.reshape(1, D) + mu_use_global.reshape(1, D)

    last_date = pd.Timestamp(df[date_col].max()).normalize()
    idx = pd.bdate_range(last_date + BDay(1), periods=int(h_next))

    out = pd.DataFrame(F, index=idx, columns=TARGET_COLS)
    out_to_save = out.reset_index().rename(columns={"index": date_col})[[date_col] + TARGET_COLS].copy()

    calib = {}
    history_dir = RUN_OUTPUT_DIR / "forecast_history"
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
        fname_hist = save_forecast_history(out_to_save, last_date, int(h_next), date_col)
        st.caption(f"Đã lưu forecast_history: {fname_hist.name}")
        saved_ok = True

    return out_to_save, saved_ok


def main():
    st.set_page_config(
        page_title="Dự đoán giá xăng dầu",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
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

    # Load base clean
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

    # ===== SECTION 1 =====
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
                df_plot, date_col, series,
                last_n=int(last_n),
                show_volume=bool(show_vol),
                title=f"{series} - Candlestick (daily)",
            )

    soft_divider()

    # ===== SECTION 2 =====
    with st.container(border=True):
        section_header("rocket", "Thiết lập dự đoán")

        with st.expander("Cấu hình", expanded=False):
            cc = st.columns(6)
            train_cfg = {
                "batch": cc[0].number_input("Batch", 16, 512, 128, 16),
                "epochs": cc[1].number_input("Epochs", 10, 500, 160, 10),
                "lr": cc[2].number_input("LR", 1e-6, 5e-3, 1e-4, 1e-5, format="%.6f"),
                "loss": cc[3].selectbox("Loss", ["huber", "mae", "mse"], index=0),
                "patience": cc[4].number_input("Patience", 5, 120, 30, 5),
                "ensemble_n": cc[5].number_input("Ensemble_n", 1, 7, 1, 1),
                "wd": 1e-4,
                "clip": 1.0,
                "amp": True,
                "focus_w": 3.0,
                "calib_min_points": 30,
                "calib_halflife": 180,
            }

        cc1, cc2, cc3, cc4 = st.columns([0.40, 0.20, 0.20, 0.20], vertical_alignment="bottom")
        with cc1:
            st.file_uploader(
                "Tải lên price_petroleum (.csv/.xlsx/.xls)",
                type=["csv", "xlsx", "xls"],
                key="up_pp_main",
            )
        with cc2:
            st.number_input("Số ngày dự đoán", 1, 365, DEFAULT_H_NEXT, 1, key="h_next_main")
        with cc3:
            st.checkbox(
                "Lưu forecast_history",
                value=True,
                help="Bật để lưu forecast_until_*.csv (để sau này khi có dữ liệu thực tế overlap thì dashboard sẽ tính MAE/MAPE/MSE/RMSE/R2).",
                key="save_history_main",
            )
        with cc4:
            st.button("Chạy dự đoán", use_container_width=True, key="run_btn_main", type="primary")

    if st.session_state.get("run_btn_main", False):
        st.session_state.run_triggered = True

    # ===== RUN =====
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
                    df_updated = _interpolate_external(df_updated, date_col)

                    upload_last_date = base_last_before
                    should_update_clean = False
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
                        df_updated = _interpolate_external(df_updated, date_col)
                        should_update_clean = False
                        actual_full = actual_full_before.copy()
                    else:
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
                                    df_new, base_clean, info = build_merged(
                                        up_pp_obj, clean_path_str, date_col, fill_mode, fred_api_key
                                    )
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
                retrain = True
                save_history_flag = bool(st.session_state.get("save_history_main", True))

                st.session_state.actual_full = df_use.copy()
                st.session_state.actual_clean = actual_full.copy() if actual_full is not None else None

                pred_df, _ = run_forecast(
                    df_use,
                    date_col,
                    h_next,
                    save_history=save_history_flag,
                    retrain=retrain,
                    train_cfg=train_cfg,
                    actual_full=df_use,
                )

                st.session_state.pred_df = pred_df
                st.session_state._df_use_for_prev5 = df_use

    # ===== BELOW hidden until pred exists =====
    pred_df = st.session_state.get("pred_df")
    if pred_df is None or len(pred_df) == 0:
        return

    soft_divider()

    with st.container(border=True):
        section_header("table", "Kết quả dự đoán")
        st.dataframe(pred_df, use_container_width=True, height=280, hide_index=True)
        st.download_button(
            "Tải forecast.csv",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.container(border=True):
        section_header("flask", "Sanity check: 5 ngày dự đoán vs 5 ngày trước đó")
        df_use = st.session_state.get("_df_use_for_prev5")
        metrics_5, tbl_5 = pred5_vs_prev5_metrics_table(
            df_use=df_use,
            pred_df=pred_df,
            date_col=date_col,
            target_cols=TARGET_COLS,
            n=5,
            eps=1e-8,
        )

        if metrics_5 is None:
            st.info("Không đủ dữ liệu để so sánh 5 ngày (cần >=5 dòng lịch sử và >=5 dòng dự đoán).")
        else:
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("n_days", str(metrics_5["n_days"]))
            c2.metric("Macro MAE", f'{metrics_5["macro_mae"]:.4f}')
            c3.metric("Macro MAPE (%)", f'{metrics_5["macro_mape_%"]:.3f}')
            c4.metric("Macro MSE", f'{metrics_5["macro_mse"]:.4f}')
            c5.metric("Macro RMSE", f'{metrics_5["macro_rmse"]:.4f}')
            r2v = metrics_5.get("macro_r2", np.nan)
            c6.metric("Macro R2", f"{r2v:.4f}" if np.isfinite(r2v) else "—")
            do_avg_mae = (metrics_5["DO 0.001%_mae"] + metrics_5["DO 0.05%_mae"]) / 2
            c7.metric("DO avg MAE", f"{do_avg_mae:.4f}")
            st.dataframe(tbl_5, use_container_width=True, height=320, hide_index=True)

    soft_divider()

    with st.container(border=True):
        section_header("clipboard-check", "Đánh giá forecast_history (so với thực tế)")

        actual_full = st.session_state.get("actual_full")
        if actual_full is None:
            try:
                actual_full = _read_actual_full(clean_path_str, date_col)
                st.session_state.actual_full = actual_full
            except Exception as e:
                st.error(f"Không đọc được dữ liệu thực tế từ clean.xlsx: {e}")
                return

        hist_dir = RUN_OUTPUT_DIR / "forecast_history"
        if not hist_dir.exists():
            st.info("Chưa có thư mục forecast_history.")
            return
        files = list(hist_dir.glob("forecast_until_*.csv"))
        if not files:
            st.info("Chưa có file forecast_until_*.csv trong forecast_history.")
            return

        actual_eval = actual_full[[date_col] + [c for c in TARGET_COLS if c in actual_full.columns]].copy()
        actual_eval[date_col] = _parse_dates_any(actual_eval[date_col])
        actual_eval = actual_eval.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        act_last = actual_eval[date_col].dropna().max()
        st.caption(
            f"Actual(last) = {act_last} | History files = {len(files)} | "
            "Lưu ý: Metrics chỉ có khi actual đã có dữ liệu cho các ngày từng dự đoán (overlap_days>0)."
        )

        sig = _history_signature(hist_dir, clean_path_str) + f"|actual_last={act_last}|actual_n={len(actual_eval)}"
        with st.spinner("Đang tính MAE/MAPE/MSE/RMSE/R2 từ forecast_history so với thực tế..."):
            df_hist = _cached_eval_history(sig, actual_eval, date_col, tuple(TARGET_COLS))

        if df_hist is None or df_hist.empty:
            st.info("Không có kết quả để hiển thị.")
            return

        df_show = df_hist.copy()
        df_show["overlap_days"] = pd.to_numeric(df_show.get("overlap_days"), errors="coerce").fillna(0).astype(int)
        df_show["train_last_date"] = pd.to_datetime(df_show.get("train_last_date"), errors="coerce")
        df_show["generated_at"] = pd.to_datetime(df_show.get("generated_at"), errors="coerce")

        valid = df_show[df_show["overlap_days"] > 0].copy()
        if valid.empty:
            st.warning("Chưa có file nào overlap với thực tế (overlap_days=0). Khi bạn cập nhật actual cho các ngày đã dự đoán thì sẽ có metrics.")
            st.dataframe(df_show.sort_values(["train_last_date", "mtime"], na_position="last"),
                         use_container_width=True, height=420, hide_index=True)
            return

        t1, t2 = st.tabs(["Biểu đồ lịch sử", "Bảng chi tiết"])
        with t1:
            cA, cB = st.columns([0.55, 0.45], vertical_alignment="top")
            with cA:
                section_header("chart-line", "Xu hướng Macro MAPE (%) theo thời gian")
                st.altair_chart(history_line_chart(valid, "train_last_date", "macro_mape_%", "Macro MAPE (%)"), use_container_width=True)
            with cB:
                section_header("chart-line", "Xu hướng Macro MAE theo thời gian")
                st.altair_chart(history_line_chart(valid, "train_last_date", "macro_mae", "Macro MAE"), use_container_width=True)

            soft_divider()

            r1, r2 = st.columns([0.5, 0.5], vertical_alignment="top")
            with r1:
                section_header("trophy", "Top tốt nhất (MAPE thấp)")
                ch = history_rank_bar(valid, "macro_mape_%", "Macro MAPE (%)", top_k=8, ascending=True)
                if ch is not None:
                    st.altair_chart(ch, use_container_width=True)
            with r2:
                section_header("alert-triangle", "Top kém nhất (MAPE cao)")
                ch2 = history_rank_bar(valid, "macro_mape_%", "Macro MAPE (%)", top_k=8, ascending=False)
                if ch2 is not None:
                    st.altair_chart(ch2, use_container_width=True)

        with t2:
            st.dataframe(
                df_show.sort_values(["train_last_date", "mtime"], na_position="last"),
                use_container_width=True,
                height=420,
                hide_index=True,
            )

        soft_divider()

        w = valid["overlap_days"]
        avg_macro_mae_w  = _wavg(valid["macro_mae"], w)
        avg_macro_mape_w = _wavg(valid["macro_mape_%"], w)
        avg_macro_mse_w  = _wavg(valid["macro_mse"], w)
        avg_macro_rmse_w = _wavg(valid["macro_rmse"], w)

        avg_macro_r2_global = _r2_global_from_stats(valid, "macro")

        section_header("sum", "Trung bình lịch sử")
        s1, s2, s3, s4, s5, s6 = st.columns(6, vertical_alignment="top")
        with s1:
            stat_card("Số file có overlap", f"{len(valid):,}", icon="database")
        with s2:
            stat_card("Avg Macro MAE (wavg)", f"{avg_macro_mae_w:.4f}" if np.isfinite(avg_macro_mae_w) else "—", icon="ruler")
        with s3:
            stat_card("Avg Macro MAPE (wavg)", f"{avg_macro_mape_w:.3f}%" if np.isfinite(avg_macro_mape_w) else "—", icon="percentage")
        with s4:
            stat_card("Avg Macro MSE (wavg)", f"{avg_macro_mse_w:.4f}" if np.isfinite(avg_macro_mse_w) else "—", icon="calculator")
        with s5:
            stat_card("Avg Macro RMSE (wavg)", f"{avg_macro_rmse_w:.4f}" if np.isfinite(avg_macro_rmse_w) else "—", icon="arrows-diagonal")
        with s6:
            stat_card("Avg Macro R2 (global)", f"{avg_macro_r2_global:.4f}" if np.isfinite(avg_macro_r2_global) else "—", icon="chart-bar")

        soft_divider()
        section_header("target-arrow", "Theo từng sản phẩm")
        cols = st.columns(len(TARGET_COLS), vertical_alignment="top")
        for i, c in enumerate(TARGET_COLS):
            mae_col  = f"{c}_mae"
            mape_col = f"{c}_mape_%"
            mse_col  = f"{c}_mse"
            rmse_col = f"{c}_rmse"

            v_mae  = _wavg(valid[mae_col], w) if mae_col in valid.columns else float("nan")
            v_mape = _wavg(valid[mape_col], w) if mape_col in valid.columns else float("nan")
            v_mse  = _wavg(valid[mse_col], w) if mse_col in valid.columns else float("nan")
            v_rmse = _wavg(valid[rmse_col], w) if rmse_col in valid.columns else float("nan")
            v_r2 = _r2_global_from_stats(valid, c)

            val = "—"
            parts = []
            if np.isfinite(v_mae):  parts.append(f"MAE={v_mae:.4f}")
            if np.isfinite(v_mape): parts.append(f"MAPE={v_mape:.3f}%")
            if np.isfinite(v_mse):  parts.append(f"MSE={v_mse:.4f}")
            if np.isfinite(v_rmse): parts.append(f"RMSE={v_rmse:.4f}")
            if np.isfinite(v_r2):   parts.append(f"R2={v_r2:.4f}")
            if parts:
                val = "<br/>".join(parts)

            with cols[i]:
                st.markdown(
                    f"""
                    <div class="stat-card">
                      <div class="row">
                        <div class="ttl">{c}</div>
                        <div class="badge">{ti("droplet")}</div>
                      </div>
                      <div class="val" style="font-size:20px;">{val}</div>
                      <div class="sub">MAE/MAPE/MSE/RMSE: wavg overlap | R2: global từ stats</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
