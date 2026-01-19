# app_forecast.py
# Streamlit scientific UI + HybridTriNet forecast
# - Loss ưu tiên 5 ngày đầu (focus5)
# - Fit calibration a,b từ forecast_history (overlap với actual)
# - Áp calibration vào forecast mới (+ lưu raw)
#
# Run: streamlit run app_forecast.py
#
# ✅ FIX CHÍNH:
# (1) Thêm checkbox "Lưu forecast_history" và mặc định TRUE
# (2) Không còn phụ thuộc should_update_clean để quyết định lưu history
# (3) Giữ st.session_state.actual_full = df_use (actual dùng để eval history), không overwrite lại
#
# ✅ THÊM METRICS:
# - MSE, RMSE, R2 cho:
#   (a) Sanity check: 5 ngày dự đoán vs 5 ngày trước đó
#   (b) Forecast_history vs actual
#
# ✅ FIX R² (GLOBAL, KHÔNG MEAN THEO FILE):
# - Khi eval forecast_history, lưu thêm thống kê: SSE / Σy / Σy² / n (macro & per-target)
# - Ở phần tổng hợp lịch sử, tính R² global bằng:
#   R2 = 1 - ΣSSE / ( Σy² - (Σy)² / n )

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pandas.tseries.offsets import BDay
import altair as alt

# =========================
# Path / import project
# =========================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

torch.classes.__path__ = []  # tránh lỗi torch classes trong streamlit

from src.utils.paths import BASE_DIR, RUN_OUTPUT_DIR
from src.utils.config_loader import load_yaml_config, load_env_secrets

cfg = load_yaml_config()
load_env_secrets()

DATE_COL_CFG = cfg.get("default_date_col", "Ngày")
DEFAULT_H_NEXT = int(cfg.get("default_h_next", 5))
default_clean_rel = cfg.get(
    "default_clean_path", "data/base/du_lieu_noi_suy_clean_updated_end_14-11.xlsx"
)
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
)

# =========================
# Streamlit config
# =========================
st.set_page_config(
    page_title="Dự đoán giá xăng dầu",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# Scientific UI Style
# =========================
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">

<style>
:root{
  --bg: #F5FAFF;
  --card: #FFFFFF;
  --text: #0B1220;
  --muted: #5B6B82;
  --border: #E3EDF8;
  --accent: #2563EB;
  --accent2: #14B8A6;
  --shadow: 0 10px 26px rgba(2, 6, 23, 0.06);
}

html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 1260px; padding-top: 1.6rem; padding-bottom: 2.0rem; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
section[data-testid="stSidebar"] {display: none !important;}

/* Containers (border=True) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: 18px !important;
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 0.15rem 0.15rem 0.40rem 0.15rem;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px !important;
  padding: 0.62rem 1.0rem !important;
  font-weight: 800 !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
  transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 12px 26px rgba(2, 6, 23, 0.08);
  border-color: rgba(37, 99, 235, 0.28) !important;
}
button[kind="primary"]{
  background: linear-gradient(135deg, rgba(37,99,235,1), rgba(20,184,166,0.95)) !important;
  color: white !important;
  border: 1px solid rgba(37,99,235,0.22) !important;
}
button[kind="primary"]:hover{
  border-color: rgba(20,184,166,0.45) !important;
}

/* Dataframe */
.stDataFrame{
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid var(--border);
}

/* Metrics */
[data-testid="stMetric"]{
  border-radius: 14px;
  padding: 12px 14px;
  background: #FFFFFF;
  border: 1px solid var(--border);
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
  border-radius: 12px !important;
  padding: 10px 14px !important;
  font-weight: 900 !important;
}
.stTabs [aria-selected="true"]{
  border: 1px solid rgba(37,99,235,0.18) !important;
  background: rgba(37,99,235,0.06) !important;
}

/* Hero header */
.hero{
  border-radius: 20px;
  border: 1px solid var(--border);
  background:
    radial-gradient(900px circle at 16% 12%, rgba(37,99,235,0.13), transparent 55%),
    radial-gradient(900px circle at 86% 10%, rgba(20,184,166,0.12), transparent 55%),
    linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.86));
  box-shadow: var(--shadow);
  padding: 26px 18px 18px 18px;
  margin-bottom: 18px;
}

.page-header{ text-align: center; }
.page-title{
  font-size: 42px;
  font-weight: 900;
  line-height: 1.18;
  margin: 0;
  letter-spacing: 0.2px;
}
.page-title::after{
  content:"";
  display:block;
  height: 5px;
  width: 180px;
  margin: 12px auto 0 auto;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(20,184,166,0.95));
}
.page-subtitle{
  font-size: 15px;
  color: var(--muted);
  margin: 10px 0 0 0;
}

.hr-soft{
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, rgba(2,6,23,0.12), transparent);
  border: 0;
  margin: 1.0rem 0 1.0rem 0;
}

/* Section header */
.section-h{
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0.10rem 0 0.8rem 0;
}
.section-ico{
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display:flex;
  align-items:center;
  justify-content:center;
  border: 1px solid var(--border);
  background: #FFFFFF;
}
.section-ico i{ font-size: 18px; color: var(--accent2); }
.section-txt{
  font-size: 20px;
  font-weight: 900;
  color: var(--text);
}

/* Nice stat cards */
.stat-card{
  border-radius: 16px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(37,99,235,0.06), rgba(255,255,255,1));
  box-shadow: 0 8px 18px rgba(2,6,23,0.05);
  padding: 14px 14px 12px 14px;
}
.stat-card .row{
  display:flex; align-items:center; justify-content:space-between;
}
.stat-card .ttl{
  font-size: 13px;
  font-weight: 800;
  color: var(--muted);
}
.stat-card .val{
  font-size: 30px;
  font-weight: 900;
  color: var(--text);
  margin-top: 6px;
  line-height: 1.1;
}
.stat-card .sub{
  font-size: 12px;
  color: var(--muted);
  margin-top: 6px;
}
.stat-card .badge{
  width: 34px; height: 34px;
  border-radius: 12px;
  border: 1px solid rgba(37,99,235,0.18);
  background: rgba(20,184,166,0.08);
  display:flex; align-items:center; justify-content:center;
}
.stat-card .badge i{ font-size: 18px; color: var(--accent); }

details summary { font-weight: 900 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Constants
# =========================
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]

K = 64
H = 14

VAL_RATIO = 0.10
SEED = 42

# =========================
# UI helpers
# =========================
def ti(name: str) -> str:
    return f'<i class="ti ti-{name}"></i>'


def soft_divider():
    st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)


def page_header():
    st.markdown(
        """
        <div class="hero">
          <div class="page-header">
            <div class="page-title">DỰ ĐOÁN GIÁ XĂNG DẦU</div>
            <div class="page-subtitle">
              Nền tảng dự đoán chuỗi thời gian sử dụng mô hình HybridTriNet (KAN + GRU + Attention)
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon_name: str, text: str):
    st.markdown(
        f"""
        <div class="section-h">
          <div class="section-ico">{ti(icon_name)}</div>
          <div class="section-txt">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, icon: str = "activity", subtitle: str = ""):
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="row">
            <div class="ttl">{title}</div>
            <div class="badge">{ti(icon)}</div>
          </div>
          <div class="val">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Data helpers
# =========================
def _parse_dates_any(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.normalize()

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        xi = x.round().astype("Int64")

        m_yyyymmdd = xi.between(19000101, 21001231)
        if m_yyyymmdd.any():
            out = out.fillna(
                pd.to_datetime(
                    xi.where(m_yyyymmdd).astype(str),
                    format="%Y%m%d",
                    errors="coerce",
                )
            )

        m_excel = x.between(1, 60000)
        if m_excel.any():
            out = out.fillna(
                pd.to_datetime(
                    x.where(m_excel),
                    unit="D",
                    origin="1899-12-30",
                    errors="coerce",
                )
            )

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

    mask_ymd = s2.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?$"
    )
    if mask_ymd.any():
        out = out.fillna(
            pd.to_datetime(s2.where(mask_ymd), errors="coerce", dayfirst=False, yearfirst=True)
        )

    mask_yearstart = s2.str.match(r"^\d{4}[-/]")
    fb_year = pd.to_datetime(s2.where(mask_yearstart), errors="coerce", dayfirst=False, yearfirst=True)
    fb_day = pd.to_datetime(s2.where(~mask_yearstart), errors="coerce", dayfirst=True)
    out = out.fillna(fb_year).fillna(fb_day)

    num = pd.to_numeric(s2, errors="coerce")
    if num.notna().any():
        m_excel2 = num.between(1, 60000)
        if m_excel2.any():
            out = out.fillna(
                pd.to_datetime(num.where(m_excel2), unit="D", origin="1899-12-30", errors="coerce")
            )

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


def _read_actual_full(clean_path_str: str, date_col: str) -> pd.DataFrame:
    base_full = pd.read_excel(clean_path_str, engine="openpyxl")
    if date_col not in base_full.columns:
        raise ValueError(f"File gốc thiếu cột ngày '{date_col}'")
    base_full[date_col] = _parse_dates_any(base_full[date_col])
    base_full = base_full.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return base_full


@st.cache_data(show_spinner=False)
def _cached_read_actual(clean_path_str: str, date_col: str, file_mtime: float, file_size: int) -> pd.DataFrame:
    return _read_actual_full(clean_path_str, date_col)

# =========================
# Candlestick (pseudo from close)
# =========================
def _build_pseudo_ohlc_from_close(df: pd.DataFrame, date_col: str, price_col: str, last_n: int = 260):
    tmp = df[[date_col, price_col]].copy()
    tmp[date_col] = _parse_dates_any(tmp[date_col])
    tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)
    if tmp.empty:
        return tmp

    tmp = tmp.tail(int(last_n)).reset_index(drop=True)
    tmp["close"] = tmp[price_col]
    tmp["open"] = tmp["close"].shift(1)
    tmp.loc[tmp.index[0], "open"] = tmp.loc[tmp.index[0], "close"]

    delta = (tmp["close"] - tmp["open"]).abs()
    wick = delta.rolling(10, min_periods=1).mean() * 0.6
    wick = wick.fillna(0.0)

    tmp["high"] = tmp[["open", "close"]].max(axis=1) + wick
    tmp["low"] = tmp[["open", "close"]].min(axis=1) - wick
    tmp["volume"] = delta
    return tmp


def plot_candlestick_preview(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    last_n: int = 260,
    show_volume: bool = True,
    title: Optional[str] = None,
):
    if df is None or df.empty or date_col not in df.columns or price_col not in df.columns:
        st.info("Chưa có dữ liệu để vẽ candlestick.")
        return

    ohlc = _build_pseudo_ohlc_from_close(df, date_col, price_col, last_n=last_n)
    if ohlc is None or ohlc.empty:
        st.info("Không đủ dữ liệu để vẽ.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        st.error("Thiếu Plotly. Cài bằng: pip install plotly")
        return

    if title is None:
        title = f"{price_col} - Candlestick (daily)"

    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    else:
        fig = make_subplots(rows=1, cols=1)

    candle = go.Candlestick(
        x=ohlc[date_col],
        open=ohlc["open"],
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"],
        name=f"{price_col}",
        increasing_line_color="#14B8A6",
        decreasing_line_color="#F43F5E",
        increasing_fillcolor="#14B8A6",
        decreasing_fillcolor="#F43F5E",
        line=dict(width=1),
        whiskerwidth=0.3,
    )
    fig.add_trace(candle, row=1, col=1)

    if show_volume:
        vol = go.Bar(x=ohlc[date_col], y=ohlc["volume"], name="Volume", marker=dict(color="rgba(2,6,23,0.10)"))
        fig.add_trace(vol, row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=620 if show_volume else 520,
        margin=dict(l=10, r=10, t=55, b=10),
        title=dict(text=title, x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(2,6,23,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(2,6,23,0.05)")
    if show_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume (proxy)", row=2, col=1, showgrid=False)

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# =========================
# Forecast history save
# =========================
def save_forecast_history(out: pd.DataFrame, last_date: pd.Timestamp, h_next: int, date_col: str):
    forecast_dir = RUN_OUTPUT_DIR / "forecast_history"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    # giữ 1 file / (train_last_date, H) để tránh trùng
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

# =========================
# Loss ưu tiên 5 ngày đầu (focus5)
# =========================
def _horizon_weights_torch(Hh: int, focus_h: int = 5, focus_w: float = 3.0, device=None, dtype=torch.float32):
    Hh = int(Hh)
    focus_h = int(min(max(1, focus_h), Hh))
    w = torch.ones(Hh, device=device, dtype=dtype)
    w[:focus_h] = float(focus_w)
    w = w / (w.mean() + 1e-12)  # normalize mean=1
    return w.view(1, Hh, 1)


def loss_focus5(
    out: torch.Tensor,
    y: torch.Tensor,
    loss_name: str = "huber",
    focus_h: int = 5,
    focus_w: float = 3.0,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    if out.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expect out,y [B,H,D], got out={tuple(out.shape)} y={tuple(y.shape)}")
    w = _horizon_weights_torch(out.size(1), focus_h, focus_w, device=out.device, dtype=out.dtype)

    ln = str(loss_name).lower()
    if ln == "mae":
        base = (out - y).abs()
    elif ln == "mse":
        base = (out - y).pow(2)
    elif ln == "huber":
        base = F.smooth_l1_loss(out, y, beta=float(huber_beta), reduction="none")
    else:
        raise ValueError("loss_name must be one of: mae | mse | huber")

    return (w * base).mean()


def _batch_xy(batch):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict) and ("x" in batch) and ("y" in batch):
        return batch["x"], batch["y"]
    raise ValueError("Batch format not supported. Expect (x,y) or {'x','y'}")


def _match_out_to_y(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        # maybe [B,D,H]
        if y.ndim == 3 and out.shape[1] == y.shape[2] and out.shape[2] == y.shape[1]:
            return out.transpose(1, 2)
        return out
    if out.ndim == 2 and y.ndim == 3:
        B, HH, DD = y.shape
        if out.shape[1] == HH * DD:
            return out.view(B, HH, DD)
    return out


@torch.no_grad()
def val_mae_real_focus5(model, va_loader, mu, sd, device, focus_h: int = 5, focus_w: float = 3.0):
    model.eval()

    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    sd = np.asarray(sd, dtype=np.float32).reshape(-1)

    all_sum = 0.0
    all_wsum = 0.0

    for batch in va_loader:
        x, y = _batch_xy(batch)
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        out = model(x)
        out = _match_out_to_y(out, y)

        if out.ndim != 3:
            raise ValueError(f"Model output must be [B,H,D], got {tuple(out.shape)}")

        B, Hh, Dd = out.shape
        if mu.size < Dd or sd.size < Dd:
            raise ValueError(f"mu/sd shape mismatch: mu={mu.shape}, sd={sd.shape}, D={Dd}")

        mu_t = torch.tensor(mu[:Dd], device=device, dtype=torch.float32).view(1, 1, Dd)
        sd_t = torch.tensor(sd[:Dd], device=device, dtype=torch.float32).view(1, 1, Dd)

        pr = out * sd_t + mu_t
        gt = y * sd_t + mu_t

        err = (pr - gt).abs()  # [B,H,D]
        w_h = _horizon_weights_torch(Hh, focus_h, focus_w, device=device, dtype=err.dtype)  # [1,H,1]

        all_sum += float((w_h * err).sum().item())
        all_wsum += float((w_h * torch.ones_like(err)).sum().item())

    return all_sum / max(1e-12, all_wsum)


def fit_model_better(
    model,
    tr_loader,
    va_loader,
    mu,
    sd,
    epochs: int,
    lr: float,
    loss_name: str = "huber",
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    patience: int = 25,
    use_amp: bool = True,
    status_cb=None,
    device: str = "cpu",
    focus_h: int = 5,
    focus_w: float = 3.0,
):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    total_steps = max(1, int(epochs) * max(1, len(tr_loader)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(lr), total_steps=total_steps, pct_start=0.15)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_losses = []

        for batch in tr_loader:
            x, y = _batch_xy(batch)
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x)
                out = _match_out_to_y(out, y)
                loss = loss_focus5(out, y, loss_name=loss_name, focus_h=focus_h, focus_w=focus_w, huber_beta=1.0)

            scaler.scale(loss).backward()

            if grad_clip and float(grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(opt)
            scaler.update()
            sched.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        val_mae = val_mae_real_focus5(model, va_loader, mu=mu, sd=sd, device=device, focus_h=focus_h, focus_w=focus_w)
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        lr_now = float(opt.param_groups[0]["lr"])

        if status_cb is not None:
            status_cb(ep, int(epochs), tr_loss, val_mae, lr_now)

        if val_mae < best_val - 1e-7:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val

# =========================
# ✅ AUTOREGRESSIVE SAFE (FIX tuple output)
# =========================
@torch.no_grad()
def roll_autoregressive_safe(model, seed_std: np.ndarray, H_total: int, H: int, device: str):
    """
    Safe autoregressive roll:
    - model may return Tensor or (Tensor, extra...)
    - output may be [B,H,D] or [B,D,H] or [B, H*D]
    Return: numpy (H_total, D)
    """
    model.eval()

    seed_std = np.asarray(seed_std, dtype=np.float32)
    if seed_std.ndim != 2:
        raise ValueError(f"seed_std must be (K,D), got {seed_std.shape}")

    Kk, D = seed_std.shape
    x = torch.tensor(seed_std, dtype=torch.float32, device=device).unsqueeze(0)  # [1,K,D]

    outs = []
    done = 0

    while done < int(H_total):
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        # normalize out shape -> [B,H,D]
        if out.ndim == 2:
            if out.shape[1] == int(H) * int(D):
                out = out.view(out.shape[0], int(H), int(D))
            else:
                raise ValueError(f"Unexpected 2D out shape: {tuple(out.shape)} (expect [B,H*D])")
        elif out.ndim == 3:
            if out.shape[1] == D and out.shape[2] == int(H):
                out = out.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected out.ndim={out.ndim}, shape={tuple(out.shape)}")

        take = min(int(H), int(H_total) - done)
        chunk = out[:, :take, :]  # [1,take,D]
        outs.append(chunk.detach().cpu())

        # append and keep last K
        x = torch.cat([x, chunk], dim=1)
        x = x[:, -Kk:, :]
        done += take

    y = torch.cat(outs, dim=1).squeeze(0)  # [H_total,D]
    return y.numpy()

# =========================
# Calibration a,b từ forecast_history
# =========================
def _wls_fit_ab(pred: np.ndarray, actual: np.ndarray, w: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    m = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(w) & (w > 0)
    pred, actual, w = pred[m], actual[m], w[m]
    n = int(pred.size)
    if n < 8:
        return 1.0, 0.0, np.nan, n

    X = np.column_stack([pred, np.ones_like(pred)])
    WX = X * w[:, None]
    XtWX = X.T @ WX
    XtWy = X.T @ (w * actual)

    XtWX = XtWX + 1e-10 * np.eye(2)
    beta = np.linalg.solve(XtWX, XtWy)
    a, b = float(beta[0]), float(beta[1])

    yhat = a * pred + b
    ybar = np.sum(w * actual) / (np.sum(w) + 1e-12)
    ss_res = np.sum(w * (actual - yhat) ** 2)
    ss_tot = np.sum(w * (actual - ybar) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return a, b, float(r2), n


def fit_calibration_from_history(
    history_dir: Path,
    actual_df: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    min_points: int = 30,
    recency_halflife_days: int = 180,
) -> Dict[str, Dict]:
    history_dir = Path(history_dir)
    files = sorted(history_dir.glob("forecast_until_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {}

    act = actual_df[[date_col] + list(target_cols)].copy()
    act[date_col] = pd.to_datetime(act[date_col], errors="coerce").dt.normalize()
    for c in target_cols:
        act[c] = pd.to_numeric(act[c], errors="coerce")
    act = act.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    now = datetime.now().timestamp()
    hl = float(recency_halflife_days) * 86400.0
    ln2 = np.log(2.0)

    buf = {c: {"pred": [], "act": [], "w": []} for c in target_cols}

    for f in files:
        try:
            pred = pd.read_csv(f)
        except Exception:
            continue
        if date_col not in pred.columns:
            continue

        pred[date_col] = pd.to_datetime(pred[date_col], errors="coerce").dt.normalize()
        pred = pred.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        merged = pred.merge(act, on=date_col, how="inner", suffixes=("_pred", "_actual"))
        if merged.empty:
            continue

        age = max(0.0, now - f.stat().st_mtime)
        w_recency = float(np.exp(-ln2 * age / max(1.0, hl)))
        overlap_days = int(merged[date_col].nunique())
        w_file = float(overlap_days) * w_recency

        for c in target_cols:
            cp = f"{c}_pred"
            ca = f"{c}_actual"
            if cp not in merged.columns or ca not in merged.columns:
                continue
            p = pd.to_numeric(merged[cp], errors="coerce")
            a_ = pd.to_numeric(merged[ca], errors="coerce")
            m = p.notna() & a_.notna()
            if m.sum() == 0:
                continue
            buf[c]["pred"].append(p[m].to_numpy(dtype=np.float64))
            buf[c]["act"].append(a_[m].to_numpy(dtype=np.float64))
            buf[c]["w"].append(np.full(int(m.sum()), w_file, dtype=np.float64))

    calib = {}
    for c in target_cols:
        if len(buf[c]["pred"]) == 0:
            continue
        P = np.concatenate(buf[c]["pred"])
        A = np.concatenate(buf[c]["act"])
        W = np.concatenate(buf[c]["w"])
        if int(P.size) < int(min_points):
            continue
        a, b, r2, n = _wls_fit_ab(P, A, W)
        calib[c] = {"a": a, "b": b, "r2": r2, "n": n}

    return calib


def apply_calibration(pred_df: pd.DataFrame, calib: Dict[str, Dict], target_cols: List[str], keep_raw: bool = True) -> pd.DataFrame:
    out = pred_df.copy()
    for c in target_cols:
        if c not in out.columns:
            continue
        if c not in calib:
            continue
        if keep_raw and (f"{c}_raw" not in out.columns):
            out[f"{c}_raw"] = pd.to_numeric(out[c], errors="coerce")
        a = float(calib[c]["a"])
        b = float(calib[c]["b"])
        out[c] = a * pd.to_numeric(out[c], errors="coerce") + b
    return out

# =========================
# Metrics & history eval (for dashboard)
# =========================
def _nan_mape(true_arr, pred_arr, eps=1e-8) -> float:
    true_arr = np.asarray(true_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)
    return float(np.nanmean(np.abs((pred_arr - true_arr) / (np.abs(true_arr) + eps))) * 100.0)


def _nan_mae(true_arr, pred_arr) -> float:
    true_arr = np.asarray(true_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)
    return float(np.nanmean(np.abs(pred_arr - true_arr)))


def _nan_mse(true_arr, pred_arr) -> float:
    true_arr = np.asarray(true_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)
    return float(np.nanmean((pred_arr - true_arr) ** 2))


def _nan_rmse(true_arr, pred_arr) -> float:
    mse = _nan_mse(true_arr, pred_arr)
    return float(np.sqrt(mse))


# ✅ NEW: thống kê R2 để gộp global chuẩn
def _r2_stats(true_arr, pred_arr):
    """
    Trả về thống kê để có thể gộp R2 nhiều file một cách chuẩn:
    n, sse, sum_y, sum_y2, r2 (tính theo global mean của chính mảng này)
    """
    y = np.asarray(true_arr, dtype=float).reshape(-1)
    yhat = np.asarray(pred_arr, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(yhat)
    if int(m.sum()) < 2:
        return {"n": 0, "sse": np.nan, "sum_y": np.nan, "sum_y2": np.nan, "r2": np.nan}

    y = y[m]
    yhat = yhat[m]
    n = int(y.size)
    sse = float(np.sum((y - yhat) ** 2))
    sum_y = float(np.sum(y))
    sum_y2 = float(np.sum(y ** 2))

    sst = sum_y2 - (sum_y * sum_y) / max(1, n)
    if sst <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - sse / (sst + 1e-12))

    return {"n": n, "sse": sse, "sum_y": sum_y, "sum_y2": sum_y2, "r2": r2}


def _nan_r2(true_arr, pred_arr) -> float:
    return float(_r2_stats(true_arr, pred_arr)["r2"])


# ✅ NEW: gộp R2 global từ các stats đã lưu trong df
def _r2_global_from_stats(df_stats: pd.DataFrame, prefix: str) -> float:
    n = pd.to_numeric(df_stats.get(f"{prefix}__n"), errors="coerce").fillna(0).sum()
    if float(n) < 2:
        return float("nan")

    sse = pd.to_numeric(df_stats.get(f"{prefix}__sse"), errors="coerce").fillna(0.0).sum()
    sum_y = pd.to_numeric(df_stats.get(f"{prefix}__sum_y"), errors="coerce").fillna(0.0).sum()
    sum_y2 = pd.to_numeric(df_stats.get(f"{prefix}__sum_y2"), errors="coerce").fillna(0.0).sum()

    sst = float(sum_y2) - (float(sum_y) * float(sum_y)) / max(1.0, float(n))
    if sst <= 1e-12:
        return float("nan")
    return float(1.0 - float(sse) / (sst + 1e-12))


def pred5_vs_prev5_metrics_table(df_use: pd.DataFrame, pred_df: pd.DataFrame, date_col: str, target_cols, n: int = 5, eps: float = 1e-8):
    if df_use is None or pred_df is None:
        return None, None
    if date_col not in df_use.columns or date_col not in pred_df.columns:
        return None, None

    hist = df_use[[date_col] + list(target_cols)].copy()
    fut = pred_df[[date_col] + list(target_cols)].copy()

    hist[date_col] = _parse_dates_any(hist[date_col])
    fut[date_col] = _parse_dates_any(fut[date_col])

    hist = hist.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    fut = fut.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    n_eff = min(int(n), len(hist), len(fut))
    if n_eff <= 0:
        return None, None

    prev = hist.tail(n_eff).reset_index(drop=True)
    pred = fut.head(n_eff).reset_index(drop=True)

    prev_vals = prev[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    pred_vals = pred[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    y_true_flat = prev_vals.reshape(-1)
    y_pred_flat = pred_vals.reshape(-1)

    metrics = {
        "n_days": int(n_eff),
        "macro_mae": _nan_mae(y_true_flat, y_pred_flat),
        "macro_mape_%": _nan_mape(y_true_flat, y_pred_flat, eps=eps),
        "macro_mse": _nan_mse(y_true_flat, y_pred_flat),
        "macro_rmse": _nan_rmse(y_true_flat, y_pred_flat),
        "macro_r2": _nan_r2(y_true_flat, y_pred_flat),
    }

    for i, c in enumerate(target_cols):
        yt = prev_vals[:, i]
        yp = pred_vals[:, i]
        metrics[f"{c}_mae"] = _nan_mae(yt, yp)
        metrics[f"{c}_mape_%"] = _nan_mape(yt, yp, eps=eps)
        metrics[f"{c}_mse"] = _nan_mse(yt, yp)
        metrics[f"{c}_rmse"] = _nan_rmse(yt, yp)
        metrics[f"{c}_r2"] = _nan_r2(yt, yp)

    out = pd.DataFrame({"prev_date": prev[date_col], "pred_date": pred[date_col]})
    for c in target_cols:
        prev_c = pd.to_numeric(prev[c], errors="coerce")
        pred_c = pd.to_numeric(pred[c], errors="coerce")
        out[f"prev_{c}"] = prev_c
        out[f"pred_{c}"] = pred_c
        out[f"abs_err_{c}"] = (pred_c - prev_c).abs()
        out[f"sq_err_{c}"] = (pred_c - prev_c) ** 2
        out[f"delta_{c}"] = pred_c - prev_c
        out[f"pct_{c}"] = (pred_c - prev_c) / (prev_c.abs() + eps) * 100.0

    return metrics, out


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return float("nan")
    return float((v[m] * w[m]).sum() / w[m].sum())


def _alt_theme_base():
    return dict(
        view=dict(strokeOpacity=0),
        axis=dict(
            gridColor="rgba(2,6,23,0.06)",
            labelColor="#5B6B82",
            titleColor="#0B1220",
            tickColor="rgba(2,6,23,0.10)",
        ),
    )


def history_line_chart(df_valid: pd.DataFrame, xcol: str, ycol: str, title: str):
    d = df_valid.dropna(subset=[xcol, ycol]).sort_values(xcol)
    base = alt.Chart(d).encode(
        x=alt.X(f"{xcol}:T", title="Mốc train_last_date"),
        tooltip=[
            alt.Tooltip("file:N", title="File"),
            alt.Tooltip(f"{xcol}:T", title="Train last"),
            alt.Tooltip("overlap_days:Q", title="Overlap days"),
            alt.Tooltip(f"{ycol}:Q", title=title),
        ],
    )
    line = base.mark_line().encode(
        y=alt.Y(f"{ycol}:Q", title=title),
        color=alt.value("#2563EB" if "MAPE" in title else "#14B8A6"),
    )
    pts = base.mark_circle(size=80, filled=True).encode(
        y=alt.Y(f"{ycol}:Q", title=title),
        color=alt.value("#2563EB" if "MAPE" in title else "#14B8A6"),
        size=alt.Size("overlap_days:Q", legend=None, scale=alt.Scale(range=[40, 260])),
    )
    ch = (line + pts).properties(height=300).interactive()
    return ch.configure(**_alt_theme_base())


def history_rank_bar(df_valid: pd.DataFrame, ycol: str, title: str, top_k: int = 8, ascending=True):
    d = df_valid.dropna(subset=["train_last_date", ycol]).sort_values(ycol, ascending=ascending).head(int(top_k)).copy()
    if d.empty:
        return None
    d["label"] = d["train_last_date"].dt.strftime("%Y-%m-%d")
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort="-x", title="train_last_date"),
            x=alt.X(f"{ycol}:Q", title=title),
            tooltip=["file", "train_last_date", "overlap_days", ycol],
            color=alt.value("#2563EB" if "MAPE" in title else "#14B8A6"),
        )
        .properties(height=260)
        .configure(**_alt_theme_base())
    )


def eval_forecast_history_dir_local(history_dir: Path, actual_df: pd.DataFrame, date_col: str, target_cols: List[str], eps: float = 1e-8) -> pd.DataFrame:
    hdir = Path(history_dir)
    files = sorted(hdir.glob("forecast_until_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        return pd.DataFrame()

    act = actual_df.copy()
    act[date_col] = _parse_dates_any(act[date_col])
    act = act.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    act_keep = [date_col] + [c for c in target_cols if c in act.columns]
    act = act[act_keep].copy()

    rows = []
    for f in files:
        try:
            pred = pd.read_csv(f)
        except Exception:
            continue
        if date_col not in pred.columns:
            continue

        pred[date_col] = _parse_dates_any(pred[date_col])
        pred = pred.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        pred_keep = [date_col] + [c for c in target_cols if c in pred.columns]
        pred2 = pred[pred_keep].copy()

        merged = pred2.merge(act, on=date_col, suffixes=("_pred", "_actual"))
        overlap_days = int(merged[date_col].nunique()) if not merged.empty else 0

        macro_mae = np.nan
        macro_mape = np.nan
        macro_mse = np.nan
        macro_rmse = np.nan
        macro_r2 = np.nan

        # ✅ stats for global R2 aggregation
        macro__n = 0
        macro__sse = np.nan
        macro__sum_y = np.nan
        macro__sum_y2 = np.nan

        per = {}

        # default per-target stats keys (so columns exist consistently)
        per_stats_default = {"n": 0, "sse": np.nan, "sum_y": np.nan, "sum_y2": np.nan, "r2": np.nan}

        if overlap_days > 0:
            all_gt = []
            all_pr = []

            for c in target_cols:
                cp, ca = f"{c}_pred", f"{c}_actual"
                if cp not in merged.columns or ca not in merged.columns:
                    per[c] = {"mae": np.nan, "mape_%": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan, **per_stats_default}
                    continue

                pr = pd.to_numeric(merged[cp], errors="coerce").to_numpy(dtype=float)
                gt = pd.to_numeric(merged[ca], errors="coerce").to_numpy(dtype=float)
                m = np.isfinite(pr) & np.isfinite(gt)

                if int(m.sum()) == 0:
                    per[c] = {"mae": np.nan, "mape_%": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan, **per_stats_default}
                    continue

                pr_m = pr[m]
                gt_m = gt[m]

                stc = _r2_stats(gt_m, pr_m)

                per[c] = {
                    "mae": _nan_mae(gt_m, pr_m),
                    "mape_%": _nan_mape(gt_m, pr_m, eps=eps),
                    "mse": _nan_mse(gt_m, pr_m),
                    "rmse": _nan_rmse(gt_m, pr_m),
                    "r2": stc["r2"],
                    "n": stc["n"],
                    "sse": stc["sse"],
                    "sum_y": stc["sum_y"],
                    "sum_y2": stc["sum_y2"],
                }

                all_gt.append(gt_m)
                all_pr.append(pr_m)

            if all_gt:
                gt_flat = np.concatenate(all_gt)
                pr_flat = np.concatenate(all_pr)

                macro_mae = _nan_mae(gt_flat, pr_flat)
                macro_mape = _nan_mape(gt_flat, pr_flat, eps=eps)
                macro_mse = _nan_mse(gt_flat, pr_flat)
                macro_rmse = float(np.sqrt(macro_mse)) if np.isfinite(macro_mse) else np.nan

                stm = _r2_stats(gt_flat, pr_flat)
                macro_r2 = stm["r2"]

                macro__n = stm["n"]
                macro__sse = stm["sse"]
                macro__sum_y = stm["sum_y"]
                macro__sum_y2 = stm["sum_y2"]

        meta_train_last = None
        if "train_last_date" in pred.columns:
            try:
                meta_train_last = pd.to_datetime(pred["train_last_date"].iloc[0], errors="coerce")
                if pd.notna(meta_train_last):
                    meta_train_last = pd.Timestamp(meta_train_last).normalize()
            except Exception:
                meta_train_last = None

        meta_generated_at = None
        if "generated_at" in pred.columns:
            try:
                meta_generated_at = pd.to_datetime(pred["generated_at"].iloc[0], errors="coerce")
            except Exception:
                meta_generated_at = None

        # ✅ row meta + macro
        row = {
            "file": f.name,
            "mtime": datetime.fromtimestamp(f.stat().st_mtime),
            "train_last_date": meta_train_last,
            "generated_at": meta_generated_at,
            "overlap_days": overlap_days,
            "macro_mae": macro_mae,
            "macro_mape_%": macro_mape,
            "macro_mse": macro_mse,
            "macro_rmse": macro_rmse,
            "macro_r2": macro_r2,
            # ✅ stats for global R2
            "macro__n": macro__n,
            "macro__sse": macro__sse,
            "macro__sum_y": macro__sum_y,
            "macro__sum_y2": macro__sum_y2,
        }

        # ✅ per-target columns + stats
        for c in target_cols:
            row[f"{c}_mae"] = per.get(c, {}).get("mae", np.nan)
            row[f"{c}_mape_%"] = per.get(c, {}).get("mape_%", np.nan)
            row[f"{c}_mse"] = per.get(c, {}).get("mse", np.nan)
            row[f"{c}_rmse"] = per.get(c, {}).get("rmse", np.nan)
            row[f"{c}_r2"] = per.get(c, {}).get("r2", np.nan)

            row[f"{c}__n"] = per.get(c, {}).get("n", 0)
            row[f"{c}__sse"] = per.get(c, {}).get("sse", np.nan)
            row[f"{c}__sum_y"] = per.get(c, {}).get("sum_y", np.nan)
            row[f"{c}__sum_y2"] = per.get(c, {}).get("sum_y2", np.nan)

        rows.append(row)

    dfm = pd.DataFrame(rows)
    if dfm.empty:
        return dfm
    return dfm.sort_values(["train_last_date", "mtime"], na_position="last").reset_index(drop=True)


def _history_signature(forecast_dir: Path, clean_path_str: str) -> str:
    items = []
    p = Path(clean_path_str)
    if p.exists():
        stt = p.stat()
        items.append(("clean", p.name, stt.st_mtime, stt.st_size))
    for f in sorted(forecast_dir.glob("forecast_until_*.csv")):
        try:
            stt = f.stat()
            items.append((f.name, stt.st_mtime, stt.st_size))
        except Exception:
            continue
    return str(items)


@st.cache_data(show_spinner=False)
def _cached_eval_history(sig: str, actual_df: pd.DataFrame, date_col: str, target_cols: tuple) -> pd.DataFrame:
    forecast_dir = RUN_OUTPUT_DIR / "forecast_history"
    return eval_forecast_history_dir_local(
        history_dir=forecast_dir,
        actual_df=actual_df,
        date_col=date_col,
        target_cols=list(target_cols),
    )

# =========================
# Forecast core (train focus5 + calibration)
# =========================
def _ensure_F_shape(F_std: np.ndarray, h_next: int, D: int) -> np.ndarray:
    if isinstance(F_std, torch.Tensor):
        F_std = F_std.detach().cpu().numpy()
    F_std = np.asarray(F_std)

    if F_std.ndim == 1:
        F_std = F_std.reshape(-1, 1)
    elif F_std.ndim == 3:
        F_std = F_std.reshape(-1, F_std.shape[-1])

    if F_std.shape[0] == D and F_std.shape[1] == h_next:
        F_std = F_std.T
    if F_std.shape[0] != h_next or F_std.shape[1] != D:
        raise ValueError(f"Forecast shape mismatch. Expect ({h_next},{D}), got {F_std.shape}")
    return F_std


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
                f"Training {si+1}/{len(seeds)} | loss={train_cfg['loss']} | epochs={train_cfg['epochs']} | focus=5d"
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

        # ✅ forecast std (SAFE)
        F_std = roll_autoregressive_safe(model, seed_std=seed_std, H_total=int(h_next), H=H, device=device_train)
        F_std = _ensure_F_shape(F_std, int(h_next), D)
        preds_std_list.append(F_std)

    pb.progress(1.0)

    F_std_ens = np.mean(np.stack(preds_std_list, axis=0), axis=0)  # (h_next, D)
    F = F_std_ens * sd_use_global.reshape(1, D) + mu_use_global.reshape(1, D)

    last_date = pd.Timestamp(df[date_col].max()).normalize()
    idx = pd.bdate_range(last_date + BDay(1), periods=int(h_next))

    out = pd.DataFrame(F, index=idx, columns=TARGET_COLS)
    out_to_save = out.reset_index().rename(columns={"index": date_col})[[date_col] + TARGET_COLS].copy()

    # ===== Calibration từ forecast_history (gom overlap) =====
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

# =========================
# App
# =========================
def main():
    page_header()

    date_col = DATE_COL_CFG
    clean_path_str = str(DEFAULT_CLEAN_PATH)
    fill_mode = "ffill"
    fred_api_key = FRED_API_KEY_DEFAULT

    st.session_state.setdefault("df_merged", None)
    st.session_state.setdefault("pred_df", None)
    st.session_state.setdefault("actual_full", None)      # ✅ actual dùng để eval history (df_use)
    st.session_state.setdefault("actual_clean", None)     # (optional) bản clean full
    st.session_state.setdefault("_df_use_for_prev5", None)
    st.session_state.setdefault("run_triggered", False)

    # Load base (clean file)
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
    # SECTION 1: Candlestick
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
                df_plot, date_col, series,
                last_n=int(last_n),
                show_volume=bool(show_vol),
                title=f"{series} - Candlestick (daily)",
            )

    soft_divider()

    # =========================
    # SECTION 2: Forecast setup
    # =========================
    with st.container(border=True):
        section_header("rocket", "Thiết lập dự đoán")

        with st.expander("Cấu hình train (focus 5 ngày + calibration)", expanded=False):
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

    # =========================
    # RUN
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

    # =========================
    # ẨN hết các phần dưới cho tới khi có dự đoán
    # =========================
    pred_df = st.session_state.get("pred_df")
    if pred_df is None or len(pred_df) == 0:
        return

    soft_divider()

    # =========================
    # SECTION 3: Forecast results + sanity
    # =========================
    with st.container(border=True):
        section_header("table", "Kết quả dự đoán (đã áp calibration nếu có)")
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

    # =========================
    # SECTION 4: forecast_history evaluation + charts
    # =========================
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

        # macro numeric
        for col in ["macro_mae", "macro_mape_%", "macro_mse", "macro_rmse", "macro_r2",
                    "macro__n", "macro__sse", "macro__sum_y", "macro__sum_y2"]:
            if col in df_show.columns:
                df_show[col] = pd.to_numeric(df_show[col], errors="coerce")
            else:
                df_show[col] = np.nan

        df_show["train_last_date"] = pd.to_datetime(df_show.get("train_last_date"), errors="coerce")
        df_show["generated_at"] = pd.to_datetime(df_show.get("generated_at"), errors="coerce")

        # per-target numeric (+ stats)
        for c in TARGET_COLS:
            for suf in ["mae", "mape_%", "mse", "rmse", "r2"]:
                col = f"{c}_{suf}"
                if col in df_show.columns:
                    df_show[col] = pd.to_numeric(df_show[col], errors="coerce")
                else:
                    df_show[col] = np.nan

            for suf in ["__n", "__sse", "__sum_y", "__sum_y2"]:
                col = f"{c}{suf}"
                if col in df_show.columns:
                    df_show[col] = pd.to_numeric(df_show[col], errors="coerce")
                else:
                    df_show[col] = np.nan

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

        # ✅ FIX: R2 global (không mean theo file, không wavg)
        avg_macro_r2_global = _r2_global_from_stats(valid, "macro")

        section_header("sum", "Trung bình lịch sử (weighted theo số ngày overlap)")
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
        section_header("target-arrow", "Theo từng sản phẩm (weighted)")
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

            # ✅ FIX: R2 theo từng target (global từ stats)
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


if __name__ == "__main__":
    main()
