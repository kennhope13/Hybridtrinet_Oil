"""
Oil Price Forecast App  –  GUMNet
Tính năng:
  1. Dự đoán tương lai (Future Forecast) – n ngày tiếp theo
  2. So sánh với lịch sử (In-sample Backtest) – dự đoán lại trên dữ liệu đã có nhãn thực tế
  3. Metrics: MAE / RMSE / MAPE cho từng target
  4. Quantile bands (p10 – p50 – p90)
  5. Tự động bổ sung cột exogenous còn thiếu từ dataset mặc định
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.model.model import GUMNet


# ─────────────────────── constants ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = BASE_DIR / "checkpoints" / "gumnet_ckpt.pt"
BUILTIN_DATA = BASE_DIR / "data" / "processed" / "clean_data_exo_ver1.csv"
QUANTILE_LABELS = ["p10", "p50", "p90"]
COLORS = {
    "actual":   "#00d4aa",
    "pred":     "#ff6b6b",
    "p10_fill": "rgba(255,107,107,0.12)",
    "p90_fill": "rgba(255,107,107,0.12)",
    "band":     "rgba(255,107,107,0.25)",
}


# ─────────────────────── helpers ─────────────────────────────────────────────
def next_business_days(last_date: pd.Timestamp, n: int):
    days = []
    d = pd.Timestamp(last_date)
    while len(days) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


def read_df(uploaded_file, date_col: str) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = [c for c in df.columns if c != date_col]
    df[num_cols] = df[num_cols].interpolate(method="linear").bfill().ffill()
    return df


def load_builtin_df(date_col: str) -> pd.DataFrame:
    """Load dataset mặc định có sẵn trong dự án."""
    df = pd.read_csv(BUILTIN_DATA)
    # tìm cột ngày (có thể bị encoding lạ)
    date_candidates = [c for c in df.columns if "ng" in c.lower() or "date" in c.lower() or "day" in c.lower()]
    if date_candidates and date_candidates[0] != date_col:
        df = df.rename(columns={date_candidates[0]: date_col})
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = [c for c in df.columns if c != date_col]
    df[num_cols] = df[num_cols].interpolate(method="linear").bfill().ffill()
    return df


def merge_missing_exo(user_df: pd.DataFrame, date_col: str, required_cols: list) -> tuple[pd.DataFrame, list]:
    """
    Nếu user_df thiếu một số cột exogenous, tự động merge từ dataset mặc định theo ngày.
    Trả về (df đã merge, danh sách cột đã được bổ sung tự động).
    """
    missing = [c for c in required_cols if c not in user_df.columns]
    if not missing:
        return user_df, []

    if not BUILTIN_DATA.exists():
        return user_df, []  # không có file mặc định, để lỗi bình thường

    builtin = load_builtin_df(date_col)
    # chỉ lấy cột cần thiết từ builtin
    exo_available = [c for c in missing if c in builtin.columns]
    if not exo_available:
        return user_df, []

    exo_df = builtin[[date_col] + exo_available].copy()
    merged = pd.merge(user_df, exo_df, on=date_col, how="left")

    # forward/backward fill cho các ngày không có trong builtin
    merged[exo_available] = merged[exo_available].ffill().bfill()

    # nếu vẫn NaN (user_df có ngày mới hơn builtin), dùng giá trị cuối của builtin
    for col in exo_available:
        if merged[col].isna().any():
            last_val = builtin[col].iloc[-1]
            merged[col] = merged[col].fillna(last_val)

    still_missing = [c for c in missing if c not in exo_available]
    return merged, exo_available


def load_model(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = GUMNet(
        seq_len=ckpt["seq_len"],
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        horizon=ckpt["horizon"],
        num_quantiles=ckpt["num_quantiles"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, device


def predict_future(model, ckpt, df: pd.DataFrame, device: str):
    """Dự đoán n bước tiếp theo sau dữ liệu cuối cùng."""
    feature_cols  = ckpt["feature_cols"]
    target_cols   = ckpt["target_cols"]
    seq_len       = ckpt["seq_len"]
    horizon       = ckpt["horizon"]
    num_q         = ckpt["num_quantiles"]
    date_col      = ckpt["date_col"]

    X_all = ckpt["feature_scaler"].transform(df[feature_cols].values)
    x_last = torch.tensor(X_all[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled, weights = model(x_last)   # [1, H, O, Q]

    pred_np = pred_scaled.cpu().numpy()[0]      # [H, O, Q]
    pred_np = np.sort(pred_np, axis=-1)         # Đảm bảo p10 <= p50 <= p90
    results = {}
    for qi, ql in enumerate(QUANTILE_LABELS):
        p = ckpt["target_scaler"].inverse_transform(pred_np[:, :, qi])   # [H, O]
        results[ql] = pd.DataFrame(p, columns=target_cols)

    future_dates = next_business_days(pd.to_datetime(df[date_col].iloc[-1]), horizon)
    for ql in QUANTILE_LABELS:
        results[ql].insert(0, date_col, future_dates)

    return results, weights.cpu().numpy()


def backtest(model, ckpt, df: pd.DataFrame, device: str, n_samples: int = 200):
    """
    In-sample backtest: trượt cửa sổ seq_len qua dữ liệu lịch sử,
    lấy bước dự đoán đầu tiên (h=0) của từng vị trí → so sánh với giá thực tế.
    """
    feature_cols  = ckpt["feature_cols"]
    target_cols   = ckpt["target_cols"]
    seq_len       = ckpt["seq_len"]
    date_col      = ckpt["date_col"]

    X_all   = ckpt["feature_scaler"].transform(df[feature_cols].values)
    dates   = df[date_col].values
    actuals = df[target_cols].values  # original scale

    # Lấy n_samples điểm cuối để so sánh (tránh quá chậm)
    start_i = max(seq_len, len(df) - seq_len - n_samples)
    preds_p10, preds_p50, preds_p90 = [], [], []
    actual_list, date_list = [], []

    for i in range(start_i, len(df) - 1):
        x = torch.tensor(X_all[i - seq_len: i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled, _ = model(x)   # [1, H, O, Q]

        p_np = pred_scaled.cpu().numpy()[0, 0, :, :]  # [O, Q] – h=0
        p_np = np.sort(p_np, axis=-1)                 # Đảm bảo p10 <= p50 <= p90

        def inv(arr):
            return ckpt["target_scaler"].inverse_transform(arr.reshape(1, -1)).flatten()

        preds_p10.append(inv(p_np[:, 0]))
        preds_p50.append(inv(p_np[:, 1]))
        preds_p90.append(inv(p_np[:, 2]))
        actual_list.append(actuals[i])
        date_list.append(dates[i])

    bt = pd.DataFrame(date_list, columns=[date_col])
    for j, col in enumerate(target_cols):
        bt[f"{col}_actual"] = [a[j] for a in actual_list]
        bt[f"{col}_p10"]    = [p[j] for p in preds_p10]
        bt[f"{col}_p50"]    = [p[j] for p in preds_p50]
        bt[f"{col}_p90"]    = [p[j] for p in preds_p90]
    return bt


def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def make_comparison_chart(bt_df: pd.DataFrame, col: str, date_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_df[date_col], y=bt_df[f"{col}_p90"],
        mode="lines", line=dict(width=0),
        name="p90", showlegend=False,
        fillcolor=COLORS["p10_fill"],
    ))
    fig.add_trace(go.Scatter(
        x=bt_df[date_col], y=bt_df[f"{col}_p10"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=COLORS["band"],
        name="Dải p10–p90",
    ))
    fig.add_trace(go.Scatter(
        x=bt_df[date_col], y=bt_df[f"{col}_p50"],
        mode="lines", line=dict(color=COLORS["pred"], width=2),
        name="Dự đoán (p50)",
    ))
    fig.add_trace(go.Scatter(
        x=bt_df[date_col], y=bt_df[f"{col}_actual"],
        mode="lines", line=dict(color=COLORS["actual"], width=2),
        name="Thực tế",
    ))
    fig.update_layout(
        title=f"📊 So sánh dự đoán vs thực tế – {col}",
        xaxis_title="Ngày",
        yaxis_title="Giá (VNĐ/lít hoặc USD/thùng)",
        template="plotly_dark",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_future_chart(hist_df: pd.DataFrame, future_results: dict,
                      col: str, date_col: str, n_hist: int) -> go.Figure:
    fig = go.Figure()
    hist_slice = hist_df.tail(n_hist)

    # Lịch sử
    fig.add_trace(go.Scatter(
        x=hist_slice[date_col], y=hist_slice[col],
        mode="lines", line=dict(color=COLORS["actual"], width=2),
        name="Lịch sử thực tế",
    ))

    fut_dates = future_results["p50"][date_col]
    # Nối điểm cuối lịch sử vào dự đoán để đường liền mạch
    last_date  = hist_slice[date_col].iloc[-1]
    last_val   = hist_slice[col].iloc[-1]

    def concat_last(series):
        return pd.concat([pd.Series([last_val]), series], ignore_index=True)

    def concat_date(dates):
        return pd.concat([pd.Series([last_date]), dates], ignore_index=True)

    all_dates = concat_date(fut_dates)

    # Band p10–p90
    fig.add_trace(go.Scatter(
        x=all_dates, y=concat_last(future_results["p90"][col]),
        mode="lines", line=dict(width=0), showlegend=False,
        fillcolor=COLORS["p10_fill"],
    ))
    fig.add_trace(go.Scatter(
        x=all_dates, y=concat_last(future_results["p10"][col]),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=COLORS["band"],
        name="Dải p10–p90",
    ))
    # Dự đoán p50
    fig.add_trace(go.Scatter(
        x=all_dates, y=concat_last(future_results["p50"][col]),
        mode="lines+markers", line=dict(color=COLORS["pred"], width=2.5, dash="dash"),
        marker=dict(size=7),
        name="Dự đoán (p50)",
    ))

    fig.update_layout(
        title=f"🔮 Dự báo tương lai – {col}",
        xaxis_title="Ngày",
        yaxis_title="Giá",
        template="plotly_dark",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═══════════════════════ MAIN APP ═══════════════════════════════════════════
st.set_page_config(
    page_title="Oil Price Forecast | GUMNet",
    page_icon="🛢️",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {

    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
}
.metric-title { color: #8892a4; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #ffffff; font-size: 28px; font-weight: 700; margin: 4px 0 0; }
.metric-sub   { color: #00d4aa; font-size: 12px; margin-top: 2px; }
.section-header {
    background: linear-gradient(90deg, #0f3460, #16213e);
    border-left: 4px solid #00d4aa;
    padding: 10px 18px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px;
    font-weight: 600;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 8px;">
  <h1 style="font-size:2.6rem; font-weight:800; margin:0;
      background: linear-gradient(135deg,#00d4aa,#4facfe);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🛢️ Oil Price Forecast
  </h1>
  <p style="color:#8892a4; margin-top:8px; font-size:1rem;">
    GUMNet – Dự báo giá xăng dầu với so sánh thực tế & quantile bands
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    ckpt_path = st.text_input("📁 Checkpoint path", str(DEFAULT_CKPT))
    st.divider()

    data_source = st.radio(
        "📂 Nguồn dữ liệu",
        ["📤 Upload file của bạn", "📦 Dùng dataset mặc định"],
        index=1,
    )
    uploaded = None
    if data_source == "📤 Upload file của bạn":
        uploaded = st.file_uploader(
            "Upload CSV / XLSX",
            type=["csv", "xlsx", "xls"],
            help="Nếu thiếu cột exogenous (USD_Index, GPR…), app sẽ tự động bổ sung từ dataset mặc định theo ngày.",
        )
        st.caption("💡 Cột exogenous còn thiếu sẽ được **tự động merge** từ dữ liệu gốc của dự án.")
    st.divider()

    # Các tham số cấu hình được ẩn đi và dùng giá trị mặc định để tối giản giao diện
    n_hist   = 90    # Số ngày lịch sử hiển thị
    n_bt     = 200   # Số điểm backtest
    n_future = 20    # Số ngày dự báo hiển thị mặc định
    
    st.divider()
    st.caption("Model: **GUMNet** (CNN + GRU + WaveletKAN)")

# ── Main ─────────────────────────────────────────────────────────────────────
use_builtin = data_source == "📦 Dùng dataset mặc định"

if not use_builtin and not uploaded:
    st.info("👈 Hãy upload file dữ liệu ở sidebar hoặc chọn **'Dùng dataset mặc định'** để bắt đầu phân tích.", icon="📂")

    st.markdown('<div class="section-header">📋 Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Bước 1 – Chọn nguồn dữ liệu**
        - 📦 **Dataset mặc định**: dùng ngay `clean_data_exo_ver1.csv` có sẵn trong dự án
        - 📤 **Upload file**: CSV/XLSX của bạn — nếu thiếu cột exogenous, app tự bổ sung

        **Bước 2 – Xem kết quả**
        - Tab **Dự báo tương lai**: dự đoán n bước tiếp theo
        - Tab **So sánh lịch sử**: kiểm tra độ chính xác in-sample
        """)
    with col2:
        st.markdown("""
        **Kết quả bao gồm**
        - 📈 Biểu đồ dự báo với dải tin cậy p10–p90
        - 📊 Biểu đồ so sánh dự đoán vs thực tế
        - 📉 Metrics: MAE, RMSE, MAPE
        - 📋 Bảng giá trị chi tiết (có thể tải về)
        """)
    st.stop()

# ── Load model & data ─────────────────────────────────────────────────────────
try:
    model, ckpt, device = load_model(ckpt_path)
except Exception as e:
    st.error(f"❌ Không load được model: {e}")
    st.stop()

date_col     = ckpt["date_col"]
target_cols  = ckpt["target_cols"]
feature_cols = ckpt["feature_cols"]

if use_builtin:
    try:
        df = load_builtin_df(date_col)
        st.sidebar.success(f"✅ Đã load dataset mặc định ({len(df):,} ngày)")
    except Exception as e:
        st.error(f"❌ Không đọc được dataset mặc định: {e}")
        st.stop()
else:
    try:
        df = read_df(uploaded, date_col)
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {e}")
        st.stop()

    # Tự động merge cột exogenous còn thiếu từ dataset mặc định
    df, auto_merged = merge_missing_exo(df, date_col, feature_cols)
    if auto_merged:
        st.info(
            f"💡 **Tự động bổ sung {len(auto_merged)} cột exogenous** từ dataset mặc định theo ngày: "
            f"`{'`, `'.join(auto_merged)}`",
            icon="🔗",
        )

    # Kiểm tra cột vẫn còn thiếu sau khi merge
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(
            f"❌ File vẫn thiếu các cột sau (không có trong dataset mặc định): "
            f"`{'`, `'.join(missing)}`"
        )
        st.stop()

# ── Summary metrics row ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Tổng quan dữ liệu</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Tổng số ngày</div>
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-sub">records</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Từ ngày</div>
        <div class="metric-value" style="font-size:18px;">{df[date_col].min().strftime('%d/%m/%Y')}</div>
        <div class="metric-sub">ngày bắt đầu</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Đến ngày</div>
        <div class="metric-value" style="font-size:18px;">{df[date_col].max().strftime('%d/%m/%Y')}</div>
        <div class="metric-sub">ngày cuối</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Horizon dự báo</div>
        <div class="metric-value">{ckpt['horizon']}</div>
        <div class="metric-sub">ngày tiếp theo</div>
    </div>""", unsafe_allow_html=True)

# ── Target selector ─────────────────────────────────────────────────────────
# Lấy toàn bộ cột số có trong data (loại cột ngày) để gợi ý target thêm
_all_numeric = [c for c in df.columns if c != date_col]
# Checkpoint target
_ckpt_targets = target_cols  # target model đã train

# Nhóm: target có trong checkpoint (predict được ngay) vs chưa (cần retrain)
_predictable   = [c for c in _all_numeric if c in _ckpt_targets]
_not_available = [c for c in _all_numeric if c not in _ckpt_targets]

st.markdown('<div class="section-header">🎯 Chọn mặt hàng cần dự báo</div>', unsafe_allow_html=True)

_sel_col1, _sel_col2 = st.columns([2, 1])
with _sel_col1:
    selected_targets = st.multiselect(
        "Chọn mặt hàng (Target)",
        options=_ckpt_targets,
        default=[_ckpt_targets[0]],
        help="Chọn một hoặc nhiều mặt hàng để xem chi tiết dự báo.",
    )
with _sel_col2:
    if _not_available:
        with st.expander(f"➕ {len(_not_available)} target chưa có model"):
            st.caption("Các cột dưới đây có trong data nhưng chưa có model. Để dự báo, cần retrain với các target này:")
            for _t in _not_available:
                st.code(_t, language=None)
            st.caption("Sửa `TARGET_COLS` trong `train.py` rồi chạy lại `python train.py`.")

if not selected_targets:
    st.warning("⚠️ Vui lòng chọn ít nhất một mặt hàng để phân tích.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_future, tab_backtest, tab_data = st.tabs([
    "🔮 Dự báo tương lai",
    "📊 So sánh với lịch sử",
    "📋 Dữ liệu chi tiết",
])

# ═══════ TAB 1: FUTURE FORECAST ══════════════════════════════════════════════
with tab_future:
    st.markdown('<div class="section-header">🔮 Dự báo tương lai</div>', unsafe_allow_html=True)
    with st.spinner("Đang tính toán dự báo tương lai..."):
        try:
            future_results, gate_weights = predict_future(model, ckpt, df, device)
        except Exception as e:
            st.error(f"❌ Lỗi dự báo: {e}")
            st.stop()

    # Chart mỗi target được chọn
    for col in selected_targets:
        # Cắt kết quả theo n_future
        short_future = {k: v.head(n_future) for k, v in future_results.items()}
        fig = make_future_chart(df, short_future, col, date_col, n_hist)
        st.plotly_chart(fig, use_container_width=True)

    # Bảng kết quả
    st.markdown("**📋 Bảng giá dự báo (p10 | p50 | p90)**")
    merged = future_results["p50"][[date_col]].head(n_future).copy()
    for col in selected_targets:
        merged[f"{col}_p10"] = future_results["p10"][col].head(n_future).values
        merged[f"{col}_p50"] = future_results["p50"][col].head(n_future).values
        merged[f"{col}_p90"] = future_results["p90"][col].head(n_future).values

    merged[date_col] = merged[date_col].dt.strftime("%d/%m/%Y")
    st.dataframe(merged.round(2), use_container_width=True, hide_index=True)

    # Download
    csv_bytes = merged.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇️ Tải kết quả dự báo (CSV)", csv_bytes,
                       file_name="forecast_future.csv", mime="text/csv")

    # Gate weights
    with st.expander("⚙️ Gate weights (trọng số từng nhánh mô hình)"):
        w = gate_weights[0]
        wcol1, wcol2, wcol3 = st.columns(3)
        wcol1.metric("CNN", f"{w[0]:.4f}")
        wcol2.metric("GRU+Attn", f"{w[1]:.4f}")
        wcol3.metric("WaveletKAN", f"{w[2]:.4f}")

    # ── So sánh dự báo với giá THỰC TẾ khi đã có ────────────────────────────
    st.markdown('<div class="section-header">✅ Đối chiếu với giá thực tế (khi đã có)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Upload file chứa giá thực tế của kỳ dự báo (chỉ cần cột ngày + các cột giá mục tiêu). "
        "App sẽ tự ghép theo ngày và tính sai số."
    )

    actual_file = st.file_uploader(
        "📥 Upload giá thực tế (CSV / XLSX)",
        type=["csv", "xlsx", "xls"],
        key="actual_upload",
    )

    if actual_file is not None:
        try:
            act_name = actual_file.name.lower()
            if act_name.endswith((".xlsx", ".xls")):
                act_df = pd.read_excel(actual_file)
            else:
                act_df = pd.read_csv(actual_file)

            # Chuẩn hoá cột ngày
            date_candidates = [c for c in act_df.columns
                               if "ng" in c.lower() or "date" in c.lower() or "day" in c.lower()]
            if date_candidates and date_candidates[0] != date_col:
                act_df = act_df.rename(columns={date_candidates[0]: date_col})

            act_df[date_col] = pd.to_datetime(act_df[date_col], errors="coerce")
            act_df = act_df.dropna(subset=[date_col])

            # Lấy cột target có trong file thực tế
            act_targets = [c for c in target_cols if c in act_df.columns]
            if not act_targets:
                st.warning(
                    f"⚠️ File thực tế không có cột target nào trong: `{'`, `'.join(target_cols)}`"
                )
            else:
                # Lấy forecast dates để merge
                fut_dates_ts = pd.to_datetime(future_results["p50"][date_col])
                forecast_df = future_results["p50"][[date_col] + act_targets].copy()
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
                forecast_df = forecast_df.rename(
                    columns={c: f"{c}_pred" for c in act_targets}
                )

                # Merge thực tế vào dự báo
                cmp_df = forecast_df.merge(
                    act_df[[date_col] + act_targets],
                    on=date_col,
                    how="left",
                )

                # Số ngày có giá thực tế
                n_actual = cmp_df[act_targets[0]].notna().sum()

                if n_actual == 0:
                    st.warning("⚠️ Không tìm thấy ngày nào trùng giữa dự báo và file thực tế. "
                               "Kiểm tra lại định dạng ngày.")
                else:
                    st.success(f"✅ Đã khớp **{n_actual}/{len(cmp_df)} ngày** với giá thực tế.")

                    # ── Metrics ─────────────────────────────────────────────
                    st.markdown("**📉 Độ chính xác dự báo tương lai**")
                    m_cols = st.columns(len(act_targets))
                    for idx, col in enumerate(act_targets):
                        mask = cmp_df[col].notna()
                        if mask.sum() == 0:
                            continue
                        act_vals  = cmp_df.loc[mask, col].values
                        pred_vals = cmp_df.loc[mask, f"{col}_pred"].values
                        m = compute_metrics(act_vals, pred_vals)
                        with m_cols[idx]:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-title">{col}</div>
                                <div class="metric-value" style="font-size:20px;">MAE {m['MAE']:.2f}</div>
                                <div class="metric-sub">RMSE: {m['RMSE']:.2f} &nbsp;|&nbsp; MAPE: {m['MAPE (%)']:.2f}%</div>
                            </div>""", unsafe_allow_html=True)

                    # ── Biểu đồ so sánh ─────────────────────────────────────
                    st.markdown("")
                    for col in act_targets:
                        fig_cmp = go.Figure()

                        # Lịch sử (n_hist ngày cuối)
                        hist_sl = df.tail(n_hist)
                        fig_cmp.add_trace(go.Scatter(
                            x=hist_sl[date_col], y=hist_sl[col],
                            mode="lines",
                            line=dict(color=COLORS["actual"], width=1.5),
                            name="Lịch sử",
                        ))

                        # Band p10–p90 dự báo
                        all_fut_dates = pd.to_datetime(future_results["p50"][date_col])
                        last_date_hist = df[date_col].iloc[-1]
                        last_val_hist  = df[col].iloc[-1]

                        def _concat(s):
                            return pd.concat([pd.Series([last_val_hist]), s.reset_index(drop=True)],
                                             ignore_index=True)
                        def _concat_d(d):
                            return pd.concat([pd.Series([last_date_hist]), d.reset_index(drop=True)],
                                             ignore_index=True)

                        all_d = _concat_d(all_fut_dates)
                        fig_cmp.add_trace(go.Scatter(
                            x=all_d, y=_concat(future_results["p90"][col]),
                            mode="lines", line=dict(width=0), showlegend=False,
                        ))
                        fig_cmp.add_trace(go.Scatter(
                            x=all_d, y=_concat(future_results["p10"][col]),
                            mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=COLORS["band"],
                            name="Dải p10–p90",
                        ))

                        # Dự báo p50
                        fig_cmp.add_trace(go.Scatter(
                            x=all_d, y=_concat(future_results["p50"][col]),
                            mode="lines+markers",
                            line=dict(color=COLORS["pred"], width=2, dash="dash"),
                            marker=dict(size=6),
                            name="Dự báo (p50)",
                        ))

                        # Thực tế (chỉ các ngày có dữ liệu)
                        mask = cmp_df[col].notna()
                        fig_cmp.add_trace(go.Scatter(
                            x=cmp_df.loc[mask, date_col],
                            y=cmp_df.loc[mask, col],
                            mode="markers+lines",
                            marker=dict(size=10, symbol="star",
                                        color="#ffd700", line=dict(color="white", width=1)),
                            line=dict(color="#ffd700", width=2),
                            name="Thực tế (đã cập nhật)",
                        ))

                        fig_cmp.update_layout(
                            title=f"📍 Dự báo vs Thực tế – {col}",
                            xaxis_title="Ngày",
                            yaxis_title="Giá",
                            template="plotly_dark",
                            hovermode="x unified",
                            height=460,
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)

                    # ── Bảng đối chiếu ───────────────────────────────────────
                    with st.expander("📋 Bảng đối chiếu chi tiết"):
                        show_cmp = cmp_df.copy()
                        show_cmp[date_col] = show_cmp[date_col].dt.strftime("%d/%m/%Y")
                        st.dataframe(show_cmp.round(2), use_container_width=True, hide_index=True)
                        csv_cmp = show_cmp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Tải bảng đối chiếu (CSV)", csv_cmp,
                            file_name="forecast_vs_actual.csv", mime="text/csv",
                        )

        except Exception as e:
            st.error(f"❌ Lỗi đọc file thực tế: {e}")

# ═══════ TAB 2: BACKTEST / SO SÁNH LỊCH SỬ ══════════════════════════════════
with tab_backtest:
    st.markdown('<div class="section-header">📊 So sánh dự đoán với giá thực tế (In-sample backtest)</div>',
                unsafe_allow_html=True)

    if len(df) < ckpt["seq_len"] + 2:
        st.warning("⚠️ Dữ liệu quá ít để chạy backtest.")
    else:
        with st.spinner(f"Đang chạy backtest trên {n_bt} điểm cuối..."):
            try:
                bt_df = backtest(model, ckpt, df, device, n_samples=n_bt)
            except Exception as e:
                st.error(f"❌ Lỗi backtest: {e}")
                st.stop()

        # Metrics
        st.markdown("**📉 Độ chính xác của mô hình (h=+1 ngày)**")
        metrics_cols = st.columns(len(selected_targets))
        for idx, col in enumerate(selected_targets):
            actual = bt_df[f"{col}_actual"].values
            pred   = bt_df[f"{col}_p50"].values
            m = compute_metrics(actual, pred)
            with metrics_cols[idx]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">{col} – Metrics</div>
                    <div class="metric-value" style="font-size:20px;">MAE: {m['MAE']:.2f}</div>
                    <div class="metric-sub">RMSE: {m['RMSE']:.2f} &nbsp;|&nbsp; MAPE: {m['MAPE (%)']:.2f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Charts
        for col in selected_targets:
            fig = make_comparison_chart(bt_df, col, date_col)
            st.plotly_chart(fig, use_container_width=True)

        # Residuals chart
        st.markdown("**📐 Phân phối sai số (Residuals)**")
        for col in selected_targets:
            residuals = bt_df[f"{col}_actual"].values - bt_df[f"{col}_p50"].values
            fig_r = go.Figure()
            fig_r.add_trace(go.Histogram(
                x=residuals, nbinsx=50,
                marker_color=COLORS["pred"], opacity=0.75, name="Sai số",
            ))
            fig_r.add_vline(x=0, line_dash="dash", line_color=COLORS["actual"])
            fig_r.update_layout(
                title=f"Phân phối sai số – {col} (Thực tế − Dự đoán)",
                xaxis_title="Sai số",
                yaxis_title="Tần suất",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Bảng chi tiết
        with st.expander("📋 Xem bảng dữ liệu backtest"):
            show_bt = bt_df.copy()
            show_bt[date_col] = pd.to_datetime(show_bt[date_col]).dt.strftime("%d/%m/%Y")
            st.dataframe(show_bt.round(2), use_container_width=True, hide_index=True)
            csv_bt = show_bt.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Tải dữ liệu backtest (CSV)", csv_bt,
                               file_name="backtest_results.csv", mime="text/csv")

# ═══════ TAB 3: DATA ═════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-header">📋 Dữ liệu gốc</div>', unsafe_allow_html=True)

    col_filter = st.multiselect(
        "Chọn cột hiển thị",
        options=df.columns.tolist(),
        default=[date_col] + target_cols,
    )
    if col_filter:
        show_df = df[col_filter].copy()
        show_df[date_col] = show_df[date_col].dt.strftime("%d/%m/%Y")
        st.dataframe(show_df.tail(200).round(4), use_container_width=True, hide_index=True)

    # Line chart lịch sử
    st.markdown("**📈 Diễn biến lịch sử**")
    for col in selected_targets:
        if col in df.columns:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=df[date_col], y=df[col],
                mode="lines", line=dict(color=COLORS["actual"], width=1.5),
                name=col,
            ))
            fig_h.update_layout(
                title=f"Lịch sử giá – {col}",
                template="plotly_dark",
                height=350,
                hovermode="x unified",
            )
            st.plotly_chart(fig_h, use_container_width=True)