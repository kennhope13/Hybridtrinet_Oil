# app_retrain_forecast_monitor_1target_pct.py
# FULL Streamlit: Update Data -> Retrain -> Forecast -> Monitoring (1 target, pct)
#
# Run:
#   pip install streamlit torch pandas numpy openpyxl
#   streamlit run app_retrain_forecast_monitor_1target_pct.py

import os, math, time, random, uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Helpers: dedup columns + safe series + robust date parse
# ============================================================
def _dedup_columns(cols):
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def _get_col_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df.loc[:, col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def _parse_dates_any(x) -> pd.Series:
    s = pd.Series(x)

    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()

    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().astype(float)
        if len(vals) > 0:
            med = float(np.nanmedian(vals))
            # Excel serial days
            if 20000 <= med <= 60000:
                return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
            # Unix ms
            if med > 1e12:
                return pd.to_datetime(s, unit="ms", errors="coerce").dt.normalize()
            # Unix seconds
            if med > 1e9:
                return pd.to_datetime(s, unit="s", errors="coerce").dt.normalize()

    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.normalize()

def _norm_date(x) -> pd.Series:
    return _parse_dates_any(x)

def _tolist_tree(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _tolist_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tolist_tree(v) for v in obj]
    return obj

def _restore_scalers(scalers_like: Dict[str, Any]) -> Dict[str, Any]:
    s = scalers_like
    return {
        "all": {"mu": np.array(s["all"]["mu"], dtype=np.float32),
                "sd": np.array(s["all"]["sd"], dtype=np.float32)},
        "targets": {"mu": np.array(s["targets"]["mu"], dtype=np.float32),
                    "sd": np.array(s["targets"]["sd"], dtype=np.float32)},
    }

def _read_table_any(upload_or_path):
    name = getattr(upload_or_path, "name", None)
    if name is None:
        name = str(upload_or_path)

    if name.lower().endswith(".csv"):
        df = pd.read_csv(upload_or_path)
    else:
        # default first sheet
        df = pd.read_excel(upload_or_path)
    df.columns = _dedup_columns(df.columns)
    return df


# ============================================================
# UI (light)
# ============================================================
def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1280px; }

:root{
  --bg0: #f7f7fb;
  --bg1: #ffffff;
  --card: #ffffff;
  --bd: #e5e7eb;
  --txt: #0f172a;
  --muted: #64748b;
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 420px at 10% 0%, rgba(124,58,237,0.14), transparent 60%),
    radial-gradient(850px 420px at 90% 10%, rgba(34,197,94,0.10), transparent 60%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 65%, var(--bg0) 100%);
  color: var(--txt);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.86));
  border-right: 1px solid var(--bd);
}

.hdr{
  border: 1px solid var(--bd);
  background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(34,197,94,0.08));
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  margin-bottom: 12px;
}
.hdr h1{ margin: 0; font-size: 24px; letter-spacing: -0.02em; color: var(--txt); }
.hdr p{ margin: 6px 0 0 0; color: var(--muted); font-size: 13px; }

.card{
  border: 1px solid var(--bd);
  background: var(--card);
  border-radius: 16px;
  padding: 12px 12px 10px 12px;
  box-shadow: 0 10px 24px rgba(15,23,42,0.06);
}
.card .k{ color: var(--muted); font-size: 12px; margin-bottom: 6px;}
.card .v{ font-size: 18px; font-weight: 700; letter-spacing:-0.02em; color: var(--txt); }
.card .s{ color: var(--muted); font-size: 12px; margin-top: 6px;}

hr{ border:0; height:1px; background: var(--bd); margin: 12px 0; }

.stButton>button{
  border-radius: 12px !important;
  border: 1px solid rgba(124,58,237,0.25) !important;
  background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(34,197,94,0.80)) !important;
  color: white !important;
  font-weight: 700 !important;
  padding: 0.55rem 0.9rem !important;
  box-shadow: 0 10px 22px rgba(124,58,237,0.18) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Forecast history
# ============================================================
HISTORY_DEFAULT_PATH = "forecast_history_1target.csv"

def load_history(path: str) -> pd.DataFrame:
    if (not path) or (not os.path.exists(path)):
        return pd.DataFrame(columns=[
            "date","target","step","pred","actual",
            "abs_err","mape_%",
            "base_date","horizon","alpha",
            "created_at","run_id","model_id"
        ])
    h = pd.read_csv(path)
    if "date" in h.columns:
        h["date"] = _norm_date(h["date"])
    if "base_date" in h.columns:
        h["base_date"] = _norm_date(h["base_date"])
    for c in ["step","pred","actual","abs_err","mape_%","horizon","alpha"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    if "target" in h.columns:
        h["target"] = h["target"].astype(str)
    if "model_id" in h.columns:
        h["model_id"] = h["model_id"].astype(str)
    return h

def save_history(df: pd.DataFrame, path: str):
    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "base_date" in df2.columns:
        df2["base_date"] = pd.to_datetime(df2["base_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df2.to_csv(path, index=False, encoding="utf-8")

def recompute_errors(history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    if "actual" not in h.columns:
        h["actual"] = np.nan
    h["abs_err"] = np.where(h["actual"].notna(), np.abs(h["pred"] - h["actual"]), np.nan)
    h["mape_%"] = np.where(
        h["actual"].notna(),
        h["abs_err"] / np.maximum(np.abs(h["actual"]), 1e-6) * 100.0,
        np.nan
    )
    return h

def append_forecast_keep_first(history: pd.DataFrame, df_fore: pd.DataFrame) -> pd.DataFrame:
    """
    Giữ dự đoán LẦN ĐẦU cho (date,target). Các lần forecast sau không overwrite.
    """
    history = history.copy()
    df_fore = df_fore.copy()

    history["date"] = _norm_date(history["date"])
    df_fore["date"] = _norm_date(df_fore["date"])
    df_fore["base_date"] = _norm_date(df_fore["base_date"])

    if history.empty:
        out = df_fore
    else:
        existing = set(zip(history["date"].astype("datetime64[ns]"), history["target"].astype(str)))
        keep = []
        for d, t in zip(df_fore["date"].astype("datetime64[ns]"), df_fore["target"].astype(str)):
            keep.append((d, t) not in existing)
        out = pd.concat([history, df_fore.loc[np.array(keep, dtype=bool)]], ignore_index=True)

    out = out.sort_values(["target","date","step","created_at"], kind="mergesort").reset_index(drop=True)
    out = recompute_errors(out)
    return out

def update_history_with_actual(history: pd.DataFrame, actual_df: pd.DataFrame, date_col: str, value_col: str, target_col: str) -> pd.DataFrame:
    hist = history.copy()
    a = actual_df.copy()
    a.columns = _dedup_columns(a.columns)

    dser = _get_col_series(a, date_col)
    vser = _get_col_series(a, value_col)

    aa = pd.DataFrame({
        "date": _parse_dates_any(dser),
        "actual_new": pd.to_numeric(vser, errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")

    aa["target"] = str(target_col)
    aa = aa[["date","target","actual_new"]].drop_duplicates(subset=["date","target"])

    hist["date"] = _norm_date(hist["date"])
    hist["target"] = hist["target"].astype(str)

    merged = hist.merge(aa, on=["date","target"], how="left")
    if "actual" not in merged.columns:
        merged["actual"] = np.nan

    merged["actual"] = np.where(merged["actual"].isna(), merged["actual_new"], merged["actual"])
    merged = merged.drop(columns=["actual_new"])
    merged = recompute_errors(merged)
    return merged

def history_metrics(history: pd.DataFrame) -> Dict[str, Any]:
    if history is None or history.empty:
        return {"n_pred": 0, "n_actual": 0, "n_matched": 0}

    h = history.copy()
    n_pred = int(len(h))
    n_actual = int(pd.to_numeric(h.get("actual", pd.Series([], dtype=float)), errors="coerce").notna().sum()) if "actual" in h.columns else 0

    if "actual" not in h.columns:
        return {"n_pred": n_pred, "n_actual": 0, "n_matched": 0}

    hh = h[pd.to_numeric(h["actual"], errors="coerce").notna()].copy()
    if hh.empty:
        return {"n_pred": n_pred, "n_actual": n_actual, "n_matched": 0}

    y = hh["actual"].to_numpy(np.float32)
    p = hh["pred"].to_numpy(np.float32)
    mae = float(np.mean(np.abs(p - y)))
    mape = float(np.mean(np.abs(p - y) / np.maximum(np.abs(y), 1e-6)) * 100.0)

    return {"n_pred": n_pred, "n_actual": n_actual, "n_matched": int(len(hh)), "mae": mae, "mape_%": mape}


# ============================================================
# Config / Metrics
# ============================================================
@dataclass
class CFG:
    DATE_COL: str = "Ngày"
    DROP_COLS: Tuple[str, ...] = ("Unnamed: 0", "Unnamed: 1", "Unnamed: 2")

    # window + horizon
    K: int = 128
    H_MAX: int = 100
    TRAIN_H_LIST: Tuple[int, ...] = (5, 30)

    # training
    VAL_SIZE: int = 520
    BATCH: int = 128
    EPOCHS: int = 60
    LR: float = 5e-5
    WD: float = 1e-2
    DROPOUT: float = 0.20
    GRAD_CLIP: float = 1.0
    SEED: int = 42
    AMP: bool = True
    NUM_WORKERS: int = 0
    PATIENCE: int = 12

    # model dims
    D_MODEL: int = 96
    D_HIDDEN: int = 192
    N_HEADS: int = 4
    RBF_M: int = 8

    # pct label
    EPS_PCT: float = 1e-3
    PCT_CLIP: float = 0.10
    USE_HUBER: bool = True
    HUBER_BETA: float = 0.02

    # focus steps
    FOCUS_STEPS: int = 5
    FOCUS_WEIGHT: float = 3.0

    # alpha tuning
    ALPHA_TUNE_H: int = 5
    ALPHA_GRID: int = 101


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mae_np(y_true, y_pred): return float(np.mean(np.abs(y_pred - y_true)))
def mape_np(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


# ============================================================
# Data
# ============================================================
def load_and_prepare_xlsx(data_path: str, cfg: CFG) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    df.columns = _dedup_columns(df.columns)

    for c in cfg.DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if cfg.DATE_COL not in df.columns:
        raise ValueError(f"Missing date col: {cfg.DATE_COL}")

    df[cfg.DATE_COL] = _parse_dates_any(_get_col_series(df, cfg.DATE_COL))
    df = df.dropna(subset=[cfg.DATE_COL]).sort_values(cfg.DATE_COL).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c != cfg.DATE_COL]
    # handle row "Đơn vị tính": numeric -> NaN -> fill
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].interpolate(limit_direction="both").ffill().bfill()
    return df

def merge_dataset_by_date(df_base: pd.DataFrame, df_new: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Merge df_new into df_base by date:
    - union dates
    - for overlapping columns: take new non-null values to update/extend
    """
    a = df_base.copy()
    b = df_new.copy()

    a.columns = _dedup_columns(a.columns)
    b.columns = _dedup_columns(b.columns)

    a[date_col] = _parse_dates_any(_get_col_series(a, date_col))
    b[date_col] = _parse_dates_any(_get_col_series(b, date_col))

    a = a.dropna(subset=[date_col]).sort_values(date_col)
    b = b.dropna(subset=[date_col]).sort_values(date_col)

    # coerce numeric for non-date cols
    for c in a.columns:
        if c != date_col:
            a[c] = pd.to_numeric(a[c], errors="coerce")
    for c in b.columns:
        if c != date_col:
            b[c] = pd.to_numeric(b[c], errors="coerce")

    a = a.set_index(date_col)
    b = b.set_index(date_col)

    # align union
    idx = a.index.union(b.index)
    a2 = a.reindex(idx)
    b2 = b.reindex(idx)

    # update overlapping cols
    common_cols = [c for c in b2.columns if c in a2.columns]
    for c in common_cols:
        a2[c] = np.where(b2[c].notna(), b2[c], a2[c])

    # add new columns from b
    for c in b2.columns:
        if c not in a2.columns:
            a2[c] = b2[c]

    out = a2.reset_index().sort_values(date_col).reset_index(drop=True)
    # fill numeric
    feat_cols = [c for c in out.columns if c != date_col]
    out[feat_cols] = out[feat_cols].interpolate(limit_direction="both").ffill().bfill()
    return out


def make_scalers_train_only(df: pd.DataFrame, cfg: CFG, feature_cols: List[str], target_col: str):
    n = len(df)
    train_end = n - cfg.VAL_SIZE
    if train_end <= cfg.K + cfg.H_MAX + 10:
        raise ValueError("VAL_SIZE/K/H too large for dataset length.")

    tr = df.iloc[:train_end]
    mu = tr[feature_cols].mean().to_numpy(np.float32)
    sd = tr[feature_cols].std().replace(0, 1.0).to_numpy(np.float32)

    t_mu = tr[[target_col]].mean().to_numpy(np.float32)
    t_sd = tr[[target_col]].std().replace(0, 1.0).to_numpy(np.float32)
    return {"all": {"mu": mu, "sd": sd}, "targets": {"mu": t_mu, "sd": t_sd}}

def apply_scaling(df: pd.DataFrame, feature_cols: List[str], scalers):
    df2 = df.copy()
    mu, sd = scalers["all"]["mu"], scalers["all"]["sd"]
    df2[feature_cols] = (df2[feature_cols].to_numpy(np.float32) - mu) / sd
    return df2

def inverse_target_1d(y_scaled: np.ndarray, scalers) -> np.ndarray:
    mu = np.array(scalers["targets"]["mu"], dtype=np.float32).reshape(1)
    sd = np.array(scalers["targets"]["sd"], dtype=np.float32).reshape(1)
    return y_scaled * sd + mu


class WindowDatasetPct(Dataset):
    def __init__(self, X, Y, K, H, start_t, end_t, eps_pct, pct_clip):
        self.X, self.Y = X, Y
        self.K, self.H = K, H
        self.start_t, self.end_t = start_t, end_t
        self.eps_pct = eps_pct
        self.pct_clip = pct_clip

    def __len__(self): return self.end_t - self.start_t

    def __getitem__(self, i):
        t = self.start_t + i
        x = self.X[t-self.K+1:t+1]          # (K,D)
        base = self.Y[t]                    # (1,)
        y_fut = self.Y[t+1:t+1+self.H]      # (H,1)

        denom = np.abs(base)[None, :] + self.eps_pct
        pct = (y_fut - base[None, :]) / denom
        pct = np.clip(pct, -self.pct_clip, self.pct_clip)
        return torch.from_numpy(x), torch.from_numpy(base), torch.from_numpy(pct)


# ============================================================
# Model
# ============================================================
class RBFExpansion(nn.Module):
    def __init__(self, in_features, M=8):
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, M).repeat(in_features, 1)
        self.centers = nn.Parameter(centers)
        self.gamma = nn.Parameter(torch.ones(in_features, M))
        self.in_features, self.M = in_features, M
    def forward(self, x):
        B = x.shape[0]
        x_ = x.unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        g = torch.abs(self.gamma).unsqueeze(0) + 1e-6
        phi = torch.exp(-g * (x_ - c) ** 2)
        return phi.reshape(B, self.in_features * self.M)

class KANBlock(nn.Module):
    def __init__(self, in_features, hidden, M=8, dropout=0.1):
        super().__init__()
        self.rbf = RBFExpansion(in_features, M=M)
        self.fc1 = nn.Linear(in_features * M, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)
    def forward(self, x):
        z = self.rbf(x)
        z = F.gelu(self.fc1(z)); z = self.drop(z)
        z = F.gelu(self.fc2(z)); z = self.drop(z)
        return self.norm(z)

class AttPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
    def forward(self, x):
        B, K, d = x.shape
        q = self.q.expand(B, 1, d)
        att = torch.softmax((q @ x.transpose(1, 2)) / math.sqrt(d), dim=-1)
        return (att @ x).squeeze(1)

class HybridTriNet4Pct(nn.Module):
    def __init__(self, d_in, d_model, d_hidden, n_heads, H, T, dropout, rbf_M=8):
        super().__init__()
        self.H, self.T = H, T

        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(d_model, d_hidden, num_layers=1, batch_first=True)
        self.gru_norm = nn.LayerNorm(d_hidden)

        self.att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.att_norm = nn.LayerNorm(d_model)
        self.att_pool = AttPool(d_model)
        self.att_ff = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
        )

        self.pool = AttPool(d_model)
        self.kan = KANBlock(d_model, d_hidden, M=rbf_M, dropout=dropout)

        self.head_kan = nn.Linear(d_hidden, H * T)
        self.head_gru = nn.Linear(d_hidden, H * T)
        self.head_att = nn.Linear(d_hidden, H * T)

        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 3, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, T * 3),
        )

        self.mixer = nn.Sequential(
            nn.Linear(T * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, T),
        )

    def forward(self, x, base_y):
        z = self.in_proj(x)

        _, h = self.gru(z)
        h_gru = self.gru_norm(h[-1])

        att_out, _ = self.att(z, z, z, need_weights=False)
        att_out = self.att_norm(att_out + z)
        h_att = self.att_ff(self.att_pool(att_out))

        h_kan = self.kan(self.pool(z))

        yk = self.head_kan(h_kan).view(-1, self.H, self.T)
        yg = self.head_gru(h_gru).view(-1, self.H, self.T)
        ya = self.head_att(h_att).view(-1, self.H, self.T)

        w = torch.softmax(self.gate(torch.cat([h_kan, h_gru, h_att], dim=-1)).view(-1, self.T, 3), dim=-1)
        pct = (torch.stack([yk, yg, ya], dim=-1) * w[:, None, :, :]).sum(dim=-1)

        base_rep = base_y[:, None, :].expand(-1, self.H, self.T)
        pct = pct + self.mixer(torch.cat([pct, base_rep], dim=-1))
        return pct


# ============================================================
# Train / Forecast
# ============================================================
def horizon_weights(h: int, focus_steps: int, focus_weight: float, device):
    w = torch.ones(h, device=device)
    if focus_steps > 0:
        w[:focus_steps] = focus_weight
    return w

def loss_multi_horizon_pct(pred_pct, true_pct, train_h_list, focus_steps, focus_weight, use_huber, huber_beta):
    total = 0.0
    device = pred_pct.device
    for h in train_h_list:
        p = pred_pct[:, :h, :]
        y = true_pct[:, :h, :]
        w = horizon_weights(h, focus_steps, focus_weight, device=device)
        if use_huber:
            err = F.smooth_l1_loss(p, y, reduction="none", beta=huber_beta)
        else:
            err = (p - y) ** 2
        err = err * w[None, :, None]
        total = total + err.mean()
    return total / max(len(train_h_list), 1)

@torch.no_grad()
def predict_on_loader(model, loader, device):
    model.eval()
    all_pred_pct, all_true_pct, all_base = [], [], []
    for xb, base_y, pct in loader:
        xb = xb.to(device).float()
        base_y = base_y.to(device).float()
        pct = pct.to(device).float()
        pr = model(xb, base_y)
        all_pred_pct.append(pr.cpu().numpy())
        all_true_pct.append(pct.cpu().numpy())
        all_base.append(base_y.cpu().numpy())
    return np.concatenate(all_pred_pct, 0), np.concatenate(all_true_pct, 0), np.concatenate(all_base, 0)

def tune_alpha_1target(base_s, pred_pct_s, true_pct_s, h_tune, eps_pct, grid_n):
    N, H, T = pred_pct_s.shape
    h = min(int(h_tune), int(H))
    base = base_s[:, None, :]  # (N,1,1)
    denom = np.abs(base) + float(eps_pct)
    true_price = base + true_pct_s[:, :h, :] * denom[:, :1, :]

    alphas = np.linspace(0.0, 1.0, int(grid_n)).astype(np.float32)
    best_a, best_m = 0.0, 1e9
    for a in alphas:
        pred_price = base + (a * pred_pct_s[:, :h, :]) * denom[:, :1, :]
        m = mape_np(true_price.reshape(-1), pred_price.reshape(-1))
        if m < best_m:
            best_m = m
            best_a = float(a)
    return best_a, best_m

def train_one_target(df: pd.DataFrame, target_col: str, cfg: CFG, device: torch.device,
                     save_path: str, warm_start_ckpt: Optional[Dict[str, Any]] = None,
                     log_cb=None, progress_cb=None):
    set_seed(cfg.SEED)

    feature_cols = [c for c in df.columns if c != cfg.DATE_COL]
    if target_col not in feature_cols:
        raise ValueError(f"Target {target_col} not in dataset columns.")

    scalers = make_scalers_train_only(df, cfg, feature_cols, target_col)
    df_s = apply_scaling(df, feature_cols, scalers)

    X = df_s[feature_cols].to_numpy(np.float32)
    Y = df_s[[target_col]].to_numpy(np.float32)  # (N,1)

    n = len(df_s)
    train_end = n - cfg.VAL_SIZE

    t_min = cfg.K - 1
    t_max = n - cfg.H_MAX - 1
    train_stop = min(train_end - 1, t_max)
    val_start = train_stop
    val_stop = t_max

    if train_stop - t_min <= 200:
        raise ValueError("Train samples too small. Reduce K/H or VAL_SIZE.")

    train_ds = WindowDatasetPct(X, Y, cfg.K, cfg.H_MAX, t_min, train_stop, cfg.EPS_PCT, cfg.PCT_CLIP)
    val_ds = WindowDatasetPct(X, Y, cfg.K, cfg.H_MAX, val_start, val_stop, cfg.EPS_PCT, cfg.PCT_CLIP)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    model = HybridTriNet4Pct(
        d_in=X.shape[1],
        d_model=cfg.D_MODEL,
        d_hidden=cfg.D_HIDDEN,
        n_heads=cfg.N_HEADS,
        H=cfg.H_MAX,
        T=1,
        dropout=cfg.DROPOUT,
        rbf_M=cfg.RBF_M,
    ).to(device)

    # warm start nếu ckpt cũ compatible
    if warm_start_ckpt is not None:
        try:
            model.load_state_dict(warm_start_ckpt["state_dict"], strict=True)
            if log_cb: log_cb("Warm-start: loaded previous state_dict ✅")
        except Exception as e:
            if log_cb: log_cb(f"Warm-start failed (ignore): {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    use_amp = bool(cfg.AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    best_val = float("inf")
    best_ep = -1
    bad = 0
    patience = int(cfg.PATIENCE)
    train_h_list = list(cfg.TRAIN_H_LIST)

    if log_cb:
        log_cb(f"Device: {device} | AMP: {use_amp}")
        log_cb(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
        log_cb(f"Features D={X.shape[1]} | Target={target_col}")

    for ep in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        model.train()
        losses = []

        for xb, base_y, pct in train_loader:
            xb = xb.to(device).float()
            base_y = base_y.to(device).float()
            pct = pct.to(device).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_pct = model(xb, base_y)
                pred_pct = torch.clamp(pred_pct, -cfg.PCT_CLIP, cfg.PCT_CLIP)
                loss = loss_multi_horizon_pct(
                    pred_pct, pct, train_h_list,
                    cfg.FOCUS_STEPS, cfg.FOCUS_WEIGHT,
                    cfg.USE_HUBER, cfg.HUBER_BETA
                )

            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        # val
        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, base_y, pct in val_loader:
                xb = xb.to(device).float()
                base_y = base_y.to(device).float()
                pct = pct.to(device).float()
                pred_pct = model(xb, base_y)
                pred_pct = torch.clamp(pred_pct, -cfg.PCT_CLIP, cfg.PCT_CLIP)
                vloss = loss_multi_horizon_pct(
                    pred_pct, pct, train_h_list,
                    cfg.FOCUS_STEPS, cfg.FOCUS_WEIGHT,
                    cfg.USE_HUBER, cfg.HUBER_BETA
                )
                vlosses.append(float(vloss.detach().cpu().item()))

        tr = float(np.mean(losses))
        va = float(np.mean(vlosses))
        sched.step(va)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0

        if va < best_val - 1e-6:
            best_val = va
            best_ep = ep
            bad = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "cfg": _tolist_tree(asdict(cfg)),
                "feature_cols": list(feature_cols),
                "target": target_col,
                "scalers": _tolist_tree(scalers),
            }
            torch.save(ckpt, save_path)
        else:
            bad += 1

        if log_cb:
            log_cb(f"Epoch {ep:03d}/{cfg.EPOCHS} | lr={lr_now:.2e} | train={tr:.6f} val={va:.6f} | best={best_val:.6f} (ep {best_ep}) | {dt:.1f}s")
        if progress_cb:
            progress_cb(ep / cfg.EPOCHS)

        if bad >= patience:
            if log_cb: log_cb("Early stopping.")
            break

    ckpt_best = torch.load(save_path, map_location="cpu", weights_only=False)
    # tune alpha on val
    # rebuild val loader quickly
    model.load_state_dict(ckpt_best["state_dict"])
    model.to(device).eval()

    pred_pct_s, true_pct_s, base_s = predict_on_loader(model, val_loader, device)
    pred_pct_s = np.clip(pred_pct_s, -cfg.PCT_CLIP, cfg.PCT_CLIP)

    best_a, best_m = tune_alpha_1target(base_s, pred_pct_s, true_pct_s, cfg.ALPHA_TUNE_H, cfg.EPS_PCT, cfg.ALPHA_GRID)
    return ckpt_best, float(best_a), float(best_m)


@torch.no_grad()
def forecast_from_ckpt(ckpt: Dict[str, Any], df: pd.DataFrame, base_date: pd.Timestamp, H: int, alpha: float, device: torch.device):
    cfg_ckpt = ckpt.get("cfg", {})
    feature_cols = ckpt["feature_cols"]
    target_col = ckpt["target"]
    scalers = _restore_scalers(ckpt["scalers"])

    K = int(cfg_ckpt.get("K", 128))
    H_MAX = int(cfg_ckpt.get("H_MAX", 100))
    H = int(min(H, H_MAX))
    d_model = int(cfg_ckpt.get("D_MODEL", 96))
    d_hidden = int(cfg_ckpt.get("D_HIDDEN", 192))
    n_heads = int(cfg_ckpt.get("N_HEADS", 4))
    dropout = float(cfg_ckpt.get("DROPOUT", 0.2))
    rbf_m = int(cfg_ckpt.get("RBF_M", 8))
    eps_pct = float(cfg_ckpt.get("EPS_PCT", 1e-3))
    pct_clip = float(cfg_ckpt.get("PCT_CLIP", 0.10))

    # scale
    df_s = df.copy()
    mu, sd = scalers["all"]["mu"], scalers["all"]["sd"]
    df_s[feature_cols] = (df_s[feature_cols].to_numpy(np.float32) - mu) / sd
    X_scaled = df_s[feature_cols].to_numpy(np.float32)

    # base index
    dates_norm = _parse_dates_any(df_s["Ngày"])
    base_ts = pd.Timestamp(base_date).normalize()
    idxs = np.where(dates_norm.values == np.datetime64(base_ts))[0]
    if len(idxs) == 0:
        raise ValueError("Base date not found in dataset.")
    base_idx = int(idxs[-1])
    if base_idx < K - 1:
        raise ValueError(f"Base date too early for K={K}")

    model = HybridTriNet4Pct(
        d_in=X_scaled.shape[1],
        d_model=d_model,
        d_hidden=d_hidden,
        n_heads=n_heads,
        H=H_MAX,
        T=1,
        dropout=dropout,
        rbf_M=rbf_m,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    x_win = X_scaled[base_idx - K + 1: base_idx + 1]
    x_t = torch.from_numpy(x_win).unsqueeze(0).to(device).float()

    target_idx = feature_cols.index(target_col)
    base_y_s = float(X_scaled[base_idx, target_idx])
    base_y_t = torch.tensor([[base_y_s]], device=device, dtype=torch.float32)

    pred_pct = model(x_t, base_y_t)
    pred_pct = torch.clamp(pred_pct, -pct_clip, pct_clip).cpu().numpy()[0, :H, 0]

    denom = abs(base_y_s) + eps_pct
    pred_price_s = base_y_s + (alpha * pred_pct) * denom
    pred_price = inverse_target_1d(pred_price_s.reshape(-1, 1).astype(np.float32), scalers).reshape(-1)
    base_orig = inverse_target_1d(np.array([[base_y_s]], dtype=np.float32), scalers).reshape(-1)[0]

    # future dates use dataset if exists
    if base_idx + H < len(df):
        fut_dates = _parse_dates_any(df["Ngày"].iloc[base_idx + 1: base_idx + 1 + H]).tolist()
        actual = pd.to_numeric(df[target_col].iloc[base_idx + 1: base_idx + 1 + H], errors="coerce").to_numpy(np.float32)
    else:
        fut_dates = pd.bdate_range(base_ts + pd.Timedelta(days=1), periods=H).to_pydatetime().tolist()
        actual = None

    return {
        "target": target_col,
        "base_date": base_ts,
        "base_idx": base_idx,
        "base_value": float(base_orig),
        "future_dates": fut_dates,
        "pred": pred_price.astype(np.float32),
        "actual": None if actual is None else actual.astype(np.float32),
        "pred_pct": pred_pct.astype(np.float32),
    }


# ============================================================
# App
# ============================================================
def main():
    st.set_page_config(page_title="HybridTriNet • Retrain -> Forecast -> Monitor (1 target)", page_icon="🧠", layout="wide")
    inject_css()

    st.markdown(
        """
<div class="hdr">
  <h1>🧠 HybridTriNet (pct) • Auto Retrain khi có file mới</h1>
  <p>Upload dữ liệu mới → Merge → Retrain → Forecast từ ngày cuối → Theo dõi history & so sánh actual</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if "trained_ckpt_path" not in st.session_state:
        st.session_state.trained_ckpt_path = ""
    if "trained_best_alpha" not in st.session_state:
        st.session_state.trained_best_alpha = 1.0

    with st.sidebar:
        st.markdown("### 📦 Input")
        data_path = st.text_input("Dataset base (.xlsx)", value=r"D:\Anh_Thuy\Hybridtrinet_Oil\data\data_train\root.xlsx")
        ckpt_path = st.text_input("Checkpoint (.pt)", value=st.session_state.trained_ckpt_path or "best_hybrid_pct_1target.pt")

        st.markdown("---")
        has_cuda = torch.cuda.is_available()
        device_choice = st.selectbox("Device", options=(["cuda"] if has_cuda else []) + ["cpu"], index=0 if has_cuda else 0)
        device = torch.device(device_choice)

        st.markdown("---")
        history_path = st.text_input("History file (.csv)", value=HISTORY_DEFAULT_PATH)
        auto_append_history = st.checkbox("Append forecast vào history", value=True)

    tabs = st.tabs(["🧪 Train", "🔁 Update → Retrain → Forecast", "📊 Monitoring"])

    # ---------------- TAB 1: Train manual ----------------
    with tabs[0]:
        cfg = CFG()
        if not os.path.exists(data_path):
            st.error(f"Không tìm thấy dataset: {data_path}")
        else:
            df = load_and_prepare_xlsx(data_path, cfg)
            numeric_cols = [c for c in df.columns if c != cfg.DATE_COL and pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) == 0:
                st.error("Không có cột số để làm target.")
            else:
                target_col = st.selectbox("Chọn target để train", options=numeric_cols, index=0)
                save_path = st.text_input("Lưu checkpoint", value="best_hybrid_pct_1target.pt")

                run_train = st.button("🚀 Train", use_container_width=True)
                log_box = st.empty()
                prog = st.progress(0.0)

                if run_train:
                    logs = []
                    def log_cb(s: str):
                        logs.append(s)
                        log_box.code("\n".join(logs[-250:]), language="text")
                    def progress_cb(x: float):
                        prog.progress(min(max(float(x), 0.0), 1.0))

                    try:
                        ckpt_prev = None
                        ckpt_best, best_alpha, best_mape = train_one_target(
                            df=df,
                            target_col=target_col,
                            cfg=cfg,
                            device=device,
                            save_path=save_path,
                            warm_start_ckpt=ckpt_prev,
                            log_cb=log_cb,
                            progress_cb=progress_cb,
                        )
                        st.success(f"✅ Train xong! Saved: {save_path}")
                        st.session_state.trained_ckpt_path = save_path
                        st.session_state.trained_best_alpha = best_alpha
                        st.write("best_alpha:", best_alpha, "tune_mape:", best_mape)
                    except Exception as e:
                        st.error(f"Lỗi train: {e}")

    # ---------------- TAB 2: Upload new file -> merge -> retrain -> forecast ----------------
    with tabs[1]:
        st.markdown("### 🔁 Khi có file mới: Merge dữ liệu → Train lại → Forecast")

        cfg = CFG()
        if not os.path.exists(data_path):
            st.error(f"Không tìm thấy dataset base: {data_path}")
        else:
            df_base = load_and_prepare_xlsx(data_path, cfg)
            st.write("Base dataset date range:", df_base[cfg.DATE_COL].min(), "→", df_base[cfg.DATE_COL].max())

            up_new = st.file_uploader("Upload file dữ liệu mới (actual/dataset update) (.xlsx/.csv)", type=["xlsx","csv"], key="upload_new_dataset")
            if up_new is None:
                st.info("Upload file mới ở đây. File kiểu bạn gửi (có dòng 'Đơn vị tính') vẫn OK.")
            else:
                df_new = _read_table_any(up_new)
                st.markdown("**Preview file mới (20 dòng đầu):**")
                st.dataframe(df_new.head(20), use_container_width=True)

                # select columns
                all_cols = list(df_new.columns)
                dcol = st.selectbox("Cột NGÀY trong file mới", options=all_cols, index=0)
                # auto target list based on base dataset numeric
                numeric_cols_base = [c for c in df_base.columns if c != cfg.DATE_COL and pd.api.types.is_numeric_dtype(df_base[c])]
                target_col = st.selectbox("Target sẽ train/forecast", options=numeric_cols_base, index=0)

                # If file mới có cột target, ưu tiên; nếu không, vẫn merge được (chỉ update ngày)
                vcol_default = all_cols.index(target_col) if target_col in all_cols else 0
                vcol = st.selectbox("Cột GIÁ trị (target) trong file mới", options=all_cols, index=vcol_default)

                warm_start = st.checkbox("Warm-start từ checkpoint cũ (nếu có)", value=True)
                save_merged = st.checkbox("Lưu merged dataset ra file", value=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    H = st.select_slider("H forecast", options=[5, 10, 20, 30, 60, 100], value=5)
                with c2:
                    alpha_mode = st.selectbox("Alpha dùng khi forecast", options=["best_alpha_from_retrain", "manual"], index=0)
                with c3:
                    alpha_manual = st.slider("alpha (manual)", 0.0, 1.0, 1.0, 0.01)

                retrain_and_forecast = st.button("🔁 Retrain + Forecast (từ ngày cuối)", use_container_width=True)

                if retrain_and_forecast:
                    # build df_new minimal
                    df_new2 = df_new.copy()
                    df_new2.columns = _dedup_columns(df_new2.columns)

                    # Ensure date and target exist
                    if dcol not in df_new2.columns:
                        st.error("Bạn chọn sai cột NGÀY.")
                    else:
                        # create normalized new dataframe with same columns if possible
                        tmp = pd.DataFrame({
                            cfg.DATE_COL: _parse_dates_any(_get_col_series(df_new2, dcol)),
                        })
                        # attach value if possible
                        if vcol in df_new2.columns:
                            tmp[target_col] = pd.to_numeric(_get_col_series(df_new2, vcol), errors="coerce")
                        else:
                            tmp[target_col] = np.nan

                        # merge into base
                        df_merged = merge_dataset_by_date(df_base, tmp, cfg.DATE_COL)
                        st.success(f"✅ Đã merge! New date range: {df_merged[cfg.DATE_COL].min()} → {df_merged[cfg.DATE_COL].max()}")
                        st.write("Rows:", len(df_merged))

                        merged_path = None
                        if save_merged:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            merged_path = f"merged_dataset_{ts}.xlsx"
                            try:
                                df_merged.to_excel(merged_path, index=False)
                                st.info(f"Saved merged dataset: {os.path.abspath(merged_path)}")
                            except Exception as e:
                                st.warning(f"Không lưu được file merged: {e}")
                                merged_path = None

                        # load old ckpt for warm start
                        ckpt_prev = None
                        if warm_start and os.path.exists(ckpt_path):
                            try:
                                ckpt_prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                            except Exception:
                                ckpt_prev = None

                        # retrain
                        logs = []
                        log_box = st.empty()
                        prog = st.progress(0.0)

                        def log_cb(s: str):
                            logs.append(s)
                            log_box.code("\n".join(logs[-250:]), language="text")

                        def progress_cb(x: float):
                            prog.progress(min(max(float(x), 0.0), 1.0))

                        # save new ckpt path
                        ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_ckpt_path = f"ckpt_{target_col.replace('/','_')}_{ts2}.pt"

                        try:
                            with st.spinner("Đang retrain trên dữ liệu mới..."):
                                ckpt_best, best_alpha, best_mape = train_one_target(
                                    df=df_merged,
                                    target_col=target_col,
                                    cfg=cfg,
                                    device=device,
                                    save_path=new_ckpt_path,
                                    warm_start_ckpt=ckpt_prev,
                                    log_cb=log_cb,
                                    progress_cb=progress_cb,
                                )

                            st.success(f"✅ Retrain xong! Saved ckpt: {new_ckpt_path}")
                            st.write("best_alpha:", best_alpha, "tune_mape:", best_mape)

                            st.session_state.trained_ckpt_path = new_ckpt_path
                            st.session_state.trained_best_alpha = best_alpha

                            # forecast from last date
                            last_date = pd.Timestamp(df_merged[cfg.DATE_COL].max()).normalize()
                            alpha_use = best_alpha if alpha_mode == "best_alpha_from_retrain" else float(alpha_manual)

                            with st.spinner("Đang forecast..."):
                                out = forecast_from_ckpt(ckpt_best, df_merged, last_date, int(H), float(alpha_use), device)

                            df_out = pd.DataFrame({
                                "date": [pd.Timestamp(x).normalize() for x in out["future_dates"]],
                                "pred": out["pred"],
                                "pred_pct": out["pred_pct"],
                            })
                            if out["actual"] is not None:
                                df_out["actual"] = out["actual"]
                                df_out["abs_err"] = np.abs(df_out["pred"] - df_out["actual"])
                                df_out["mape_%"] = (df_out["abs_err"] / np.maximum(np.abs(df_out["actual"]), 1e-6)) * 100.0

                            st.markdown("### 📋 Forecast kết quả")
                            st.dataframe(df_out, use_container_width=True, height=420)

                            # append history
                            if auto_append_history:
                                run_id = str(uuid.uuid4())[:8]
                                model_id = os.path.basename(new_ckpt_path)
                                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                df_hist_add = df_out[["date", "pred"]].copy()
                                df_hist_add["target"] = str(out["target"])
                                df_hist_add["step"] = np.arange(1, len(df_hist_add) + 1, dtype=int)
                                df_hist_add["actual"] = np.nan
                                df_hist_add["base_date"] = out["base_date"]
                                df_hist_add["horizon"] = int(H)
                                df_hist_add["alpha"] = float(alpha_use)
                                df_hist_add["created_at"] = created_at
                                df_hist_add["run_id"] = run_id
                                df_hist_add["model_id"] = model_id
                                df_hist_add = df_hist_add[[
                                    "date","target","step","pred","actual",
                                    "base_date","horizon","alpha","created_at","run_id","model_id"
                                ]]

                                hist0 = load_history(history_path)
                                before = len(hist0)
                                hist1 = append_forecast_keep_first(hist0, df_hist_add)
                                after = len(hist1)
                                save_history(hist1, history_path)
                                st.success(f"✅ Saved history. Added {after-before} rows. Total {after} rows.")

                        except Exception as e:
                            st.error(f"Lỗi retrain/forecast: {e}")

    # ---------------- TAB 3: Monitoring ----------------
    with tabs[2]:
        st.markdown("### 📊 Monitoring • Upload actual để fill và tính chỉ số")

        abs_hist = os.path.abspath(history_path)
        st.code(f"history_path = {abs_hist}\nexists = {os.path.exists(abs_hist)}")

        hist = load_history(history_path)
        m = history_metrics(hist)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("#Pred rows", m.get("n_pred", 0))
        c2.metric("#Actual filled", m.get("n_actual", 0))
        c3.metric("#Matched", m.get("n_matched", 0))
        c4.metric("Overall MAPE (%)", "-" if m.get("n_matched", 0) == 0 else f"{m['mape_%']:.3f}")

        st.markdown("---")
        up = st.file_uploader("Upload actual (.xlsx/.csv)", type=["xlsx","csv"], key="actual_uploader")
        if up is not None:
            df_act = _read_table_any(up)
            st.markdown("**Preview actual (20 dòng đầu):**")
            st.dataframe(df_act.head(20), use_container_width=True)

            if hist.empty:
                st.warning("History đang trống → bạn phải forecast ít nhất 1 lần trước.")
            else:
                targets_in_hist = sorted(hist["target"].astype(str).unique().tolist())
                pick_target = st.selectbox("Target cần update", options=targets_in_hist, index=0)

                all_cols = list(df_act.columns)
                dcol = st.selectbox("Cột NGÀY", options=all_cols, index=0)
                v_idx = all_cols.index(pick_target) if pick_target in all_cols else 0
                vcol = st.selectbox("Cột GIÁ trị", options=all_cols, index=v_idx)

                # debug matched / missing
                act_dates = _parse_dates_any(_get_col_series(df_act, dcol))
                st.write("Actual date min/max:", act_dates.min(), "→", act_dates.max())

                hist_t = hist[hist["target"].astype(str) == str(pick_target)].copy()
                hist_t["date"] = _norm_date(hist_t["date"])
                need = sorted(hist_t[hist_t["actual"].isna()]["date"].dropna().astype("datetime64[ns]").unique())
                act_set = set(act_dates.dropna().astype("datetime64[ns]").unique())
                need_set = set(need)

                inter = sorted(list(need_set & act_set))
                missing = sorted(list(need_set - act_set))
                st.info(f"Matched dates = {len(inter)}")
                if len(missing) > 0:
                    st.warning("Actual chưa có các ngày forecast này (nên actual vẫn None):")
                    st.write([str(pd.Timestamp(x).date()) for x in missing[:15]])

                if st.button("✅ Cập nhật history bằng actual", use_container_width=True):
                    try:
                        hist2 = update_history_with_actual(hist, df_act, dcol, vcol, pick_target)
                        save_history(hist2, history_path)
                        st.success("✅ Updated history!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi update: {e}")

        st.markdown("#### 📋 History (tail 300)")
        hist = load_history(history_path)
        st.dataframe(hist.tail(300), use_container_width=True, height=420)


if __name__ == "__main__":
    main()