import io, json
from pathlib import Path
import math, os
import sys

# ---- Thêm PROJECT ROOT vào sys.path ----
ROOT = Path(__file__).resolve().parents[1]   # D:\HybridTrinet_oil
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch

torch.classes.__path__ = []

import streamlit as st
from torch.utils.data import DataLoader
from pandas.tseries.offsets import BDay
import altair as alt
from src.utils.paths import BASE_DIR, DATA_DIR, RUN_OUTPUT_DIR
from src.utils.config_loader import load_yaml_config, load_env_secrets
import time

cfg = load_yaml_config()
load_env_secrets()

DATE_COL = cfg.get("default_date_col", "Ngày")
K = int(cfg.get("default_k", 64))
H = int(cfg.get("default_h", 14))
DEFAULT_H_NEXT = int(cfg.get("default_h_next", 5))

default_clean_rel = cfg.get("default_clean_path", "data/base/du_lieu_noi_suy_clean_updated_14-11.xlsx")
DEFAULT_CLEAN_PATH = (BASE_DIR / default_clean_rel).resolve()

# Ưu tiên FRED_API_KEY từ env (đã load từ secrets.env)
FRED_API_KEY_DEFAULT = os.getenv("FRED_API_KEY", "")
from src.dataio import (
    build_merged,
    _ensure_date,
    _align_union_columns,
    _to_excel_bytes,
)
from src.features import (
    _coerce_targets_numeric,
    _force_numeric_for_plot,
)
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
from src.utils.paths import BASE_DIR, DATA_DIR, RUN_OUTPUT_DIR

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

if not fred_api_key:
    st.sidebar.error("Chưa cấu hình FRED_API_KEY trong config/secrets.env")
date_col = st.text_input("Cột ngày", DATE_COL)
with st.sidebar:
    st.header("Cấu hình")

   

    clean_path = st.text_input(
        "Đường dẫn dữ liệu gốc",
        str(DEFAULT_CLEAN_PATH),
    )

    fill_mode = st.selectbox(
        "Xử lý NaN sau khi gộp dữ liệu",
        ["none", "ffill", "ffill+bfill", "drop rows with any NaN"],
        index=1,
    )

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
                if base0.empty:
                    base_info_box.warning("File gốc không có bản ghi hợp lệ.")
                else:
                    last_date0 = base0[date_col].max()
                    base_info_box.info(
                        f"Ngày cuối cùng trong dữ liệu gốc: **{last_date0.date()}**. "
                        "Hãy chuẩn bị file price_petroleum mới từ ngày tiếp theo trở đi."
                    )
        except Exception as e:
            base_info_box.error(f"Lỗi đọc file gốc: {e}")


st.subheader("Tải lên file price_petroleum")
up_pp = st.file_uploader(
    "price_petroleum (.csv/.xlsx/.xls)",
    type=["csv", "xlsx", "xls"],
    key="up_pp",
)

if st.button("GỘP DỮ LIỆU"):
    if not (up_pp and clean_path_str and fred_api_key):
        st.error("Thiếu file upload hoặc thiếu đường dẫn clean.xlsx hoặc thiếu FRED API key")
    else:
        try:
            df_new, base_clean, info = build_merged(
                up_pp,
                clean_path_str,
                date_col,
                fill_mode,
                fred_api_key,
            )
        except Exception as e:
            st.error(f"Lỗi merge: {e}")
            df_new = None
            base_clean = None
            info = ""

        if df_new is None or len(df_new) == 0:
            st.warning("Không có dữ liệu mới để thêm.")
        else:
            st.success(f"Gộp thành công — {info}")

            st.subheader("Dữ liệu mới")
            st.dataframe(df_new.tail(10), use_container_width=True)

            # Gộp với base_clean
            base2, add2, _ = _align_union_columns(base_clean, df_new, date_col)
            df_updated = (
                pd.concat([base2, add2], ignore_index=True)
                .drop_duplicates(subset=[date_col], keep="last")
                .sort_values(date_col)
                .reset_index(drop=True)
            )

            preferred_cols = [
                date_col,
                "MG95",
                "DO 0.001%",
                "DO 0.05%",
                "BRT DTD",
                "BRT KH",
                "WTI",
                "NgayTrongTuan",
                "ThangTrongNam",
                "QuyTrongNam",
                "Nam",
                "NgayLe",
                "SuKienDacBiet",
                "USD_Index",
                "MG92",
                "GPRD",
            ]
            order2 = [c for c in preferred_cols if c in df_updated.columns] + [
                c for c in df_updated.columns if c not in preferred_cols
            ]
            df_updated = df_updated[order2]

            st.subheader("Cập nhật dữ liệu")
            st.dataframe(df_updated.tail(15), use_container_width=True)

            st.download_button(
                "Tải xuống",
                data=_to_excel_bytes(df_updated),
                file_name="du_lieu_noi_suy_clean_updated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.session_state.df_merged = df_updated

st.header("Dự đoán dữ liệu đã cập nhật")

set_seed(SEED)
device_train = "cuda" if torch.cuda.is_available() else "cpu"

if st.session_state.df_merged is None:
    st.warning("Chưa gộp dữ liệu.")
else:
    df = st.session_state.df_merged.copy()
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        st.error(f"Thiếu các cột target trong dữ liệu gộp: {missing}")
    else:
        # Chuyển target sang numeric
        try:
            Y = _coerce_targets_numeric(df, TARGET_COLS)
        except Exception as e:
            st.error(f"Lỗi chuyển target sang số: {e}")
            Y = None

        if Y is not None:
            T, D = Y.shape
            if T <= K + H:
                st.error(f"Dữ liệu quá ngắn (T={T}) so với K={K}, H={H}.")
            else:
                # Train/val split
                val_len = max(int(T * VAL_RATIO), K + H + 1)
                train_len = T - val_len
                Y_tr, Y_val = Y[:train_len], Y[train_len - K:]

                # Chuẩn hóa theo train
                Ytr_std, mu, sd = standardize(Y_tr)
                Yval_std = (Y_val - mu) / (sd + 1e-8)

                # Xây windows
                Xtr, Ytrw = build_windows(Ytr_std, K, H)
                Xva, Yvaw = build_windows(Yval_std, K, H)

                tr_ld = DataLoader(WindowDS(Xtr, Ytrw), batch_size=BATCH_SZ, shuffle=True)
                va_ld = DataLoader(WindowDS(Xva, Yvaw), batch_size=BATCH_SZ, shuffle=False)

                if st.button("DỰ ĐOÁN"):
                    t0 = time.time()
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

                    status_box = st.empty()
                    pred_status = st.empty()
                    progress_bar = st.progress(0.0)

                    def _status_cb(ep, epochs, tr, va, mae, lr_val):
                        progress_bar.progress(ep / epochs)
                        # status_box.markdown("Đang huấn luyện...")

                    with st.spinner("Đang khởi tạo mô hình..."):
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
                    metrics_val = eval_metrics_orig(
                        model,
                        va_ld,
                        H,
                        D,
                        mu,
                        sd,
                        device=device_train,
                    )

                    progress_bar.progress(1.0)

                    RUN = RUN_OUTPUT_DIR
                    RUN.mkdir(parents=True, exist_ok=True)

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
                    }
                    diag_path = RUN / "diagnostics_streamlit.json"
                    diag_path.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")

                    ckpt_path = RUN / "hybrid_trinet_streamlit.pt"
                    torch.save(model.state_dict(), ckpt_path)

                    pred_status.write("Đang dự đoán tương lai...")
                    Y_std_full = (Y - mu) / (sd + 1e-8)
                    seed = Y_std_full[-K:]
                    F_std = roll_autoregressive(
                        model,
                        seed_std=seed,
                        H_total=int(h_next),
                        H=H,
                        device=device_train,
                    )
                    F = F_std * sd + mu
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    last_date = df[date_col].dropna().iloc[-1]
                    idx = pd.bdate_range(last_date + BDay(1), periods=int(h_next))

                    out = pd.DataFrame(F, index=idx, columns=TARGET_COLS)
                    out.to_csv(RUN / "forecast.csv", index=True)
                    pred_status.write("Dự đoán xong.")

                    cols = TARGET_COLS  # ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]

                    hist_last = Y[-1]  # shape (D,)
                    hist_last_df = pd.DataFrame([hist_last], index=[last_date], columns=cols)

                    full = pd.concat([hist_last_df, out], axis=0)

                    diff_full = full[cols].diff()

                    diff = diff_full.iloc[1:].copy()

                    diff.columns = [
                        "MG95_change",
                        "MG92_change",
                        "DO 0.001%_change",
                        "DO 0.05%_change",
                    ]

                    out_display = pd.concat([out, diff], axis=1)


                    st.subheader(f"Dự đoán {int(h_next)} tiếp theo")
                    st.dataframe(out_display, use_container_width=True)

                    plot_cols = TARGET_COLS
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

                    hist_df = df[[date_col] + plot_cols].copy()
                    hist_df = hist_df[
                        (hist_df[date_col] >= pd.Timestamp("2025-01-01")) &
                        (hist_df[date_col] <= pd.Timestamp("2025-12-31"))
                    ]

                    if not hist_df.empty:
                        hist_df.loc[:, plot_cols] = _force_numeric_for_plot(hist_df[plot_cols], plot_cols)
                        hist_long = hist_df.melt(
                            id_vars=[date_col],
                            var_name="series",
                            value_name="value",
                        )
                        hist_long["type"] = "history"
                    else:
                        hist_long = pd.DataFrame(columns=[date_col, "series", "value", "type"])

                    fcast = (
                        out.reset_index()
                        .rename(columns={"index": date_col})
                        [[date_col] + plot_cols]
                        .copy()
                    )
                    fcast_long = fcast.melt(
                        id_vars=[date_col],
                        var_name="series",
                        value_name="value",
                    )
                    fcast_long["type"] = "forecast"

                    viz = pd.concat([hist_long, fcast_long], ignore_index=True)
                    viz["value"] = pd.to_numeric(viz["value"], errors="coerce")

                    cutoff = df[date_col].max()
                    rule_df = pd.DataFrame({date_col: [cutoff]})

                    ch_lines = alt.Chart(viz).mark_line().encode(
                        x=alt.X(f"{date_col}:T", title="Ngày"),
                        y=alt.Y("value:Q", title="Giá trị"),
                        color=alt.Color("series:N", title="Series"),
                        tooltip=[date_col, "series", "type", "value"],
                    )
                    ch_rule = alt.Chart(rule_df).mark_rule(
                        strokeDash=[4, 4]
                    ).encode(
                        x=alt.X(f"{date_col}:T")
                    )

                    st.subheader("Lịch sử và dự đoán")
                    st.altair_chart((ch_lines + ch_rule).properties(height=420), use_container_width=True)
