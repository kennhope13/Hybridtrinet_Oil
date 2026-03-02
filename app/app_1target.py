# app_train_and_forecast_1target_pct.py
# Streamlit: Train + Forecast + Monitoring (forecast_history) for HybridTriNet4Pct (pct) - 1 target
#
# ✅ Train: early stopping, save ckpt (.pt), tune alpha on VAL
# ✅ Forecast: predict from base date, append forecast_history (keep first prediction per date), export CSV
# ✅ Monitoring: upload actual file many times, update history cumulatively, track MAE/MAPE overall + by step
# ✅ DEBUG upload: preview actual, show matched dates count, show samples if 0 match
# ✅ LIGHT background theme
#
# Run:
#   pip install streamlit torch pandas numpy openpyxl
#   streamlit run app_train_and_forecast_1target_pct.py

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


# =========================
# Forecast history settings
# =========================
HISTORY_DEFAULT_PATH = "forecast_history_1target.csv"


def _norm_date(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _guess_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["Ngày", "ngày", "date", "Date", "DATE"]:
        if c in df.columns:
            return c
    for c in df.columns:
        tmp = pd.to_datetime(df[c], errors="coerce")
        if tmp.notna().mean() > 0.8:
            return c
    return None


def _guess_value_col(df: pd.DataFrame, target_col: str) -> Optional[str]:
    if target_col in df.columns:
        return target_col
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[0] if num_cols else None


def load_history(path: str) -> pd.DataFrame:
    if (not path) or (not os.path.exists(path)):
        return pd.DataFrame(
            columns=[
                "date", "target", "step", "pred", "actual",
                "abs_err", "mape_%",
                "base_date", "horizon", "alpha",
                "created_at", "run_id"
            ]
        )
    h = pd.read_csv(path)
    if "date" in h.columns:
        h["date"] = _norm_date(h["date"])
    if "base_date" in h.columns:
        h["base_date"] = _norm_date(h["base_date"])
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
    Keep FIRST prediction per (date,target).
    New forecast rows with the same (date,target) will be ignored (no overwrite).
    """
    history = history.copy()
    df_fore = df_fore.copy()

    history["date"] = _norm_date(history["date"])
    df_fore["date"] = _norm_date(df_fore["date"])
    df_fore["base_date"] = _norm_date(df_fore["base_date"])

    if history.empty:
        out = df_fore
    else:
        existing = set(
            zip(
                history["date"].astype("datetime64[ns]"),
                history["target"].astype(str)
            )
        )
        keep_mask = []
        for d, t in zip(df_fore["date"].astype("datetime64[ns]"), df_fore["target"].astype(str)):
            keep_mask.append((d, t) not in existing)
        keep_mask = np.array(keep_mask, dtype=bool)
        out = pd.concat([history, df_fore.loc[keep_mask]], ignore_index=True)

    out = out.sort_values(["target", "date", "step", "created_at"], kind="mergesort").reset_index(drop=True)
    out = recompute_errors(out)
    return out


def update_history_with_actual(
    history: pd.DataFrame,
    actual_df: pd.DataFrame,
    date_col: str,
    value_col: str,
    target_col: str
) -> pd.DataFrame:
    history = history.copy()
    a = actual_df.copy()

    a[date_col] = _norm_date(a[date_col])
    a = a.dropna(subset=[date_col]).sort_values(date_col)
    a = a.rename(columns={date_col: "date", value_col: "actual_new"})
    a["target"] = target_col
    a = a[["date", "target", "actual_new"]].drop_duplicates(subset=["date", "target"])

    history["date"] = _norm_date(history["date"])

    merged = history.merge(a, on=["date", "target"], how="left")

    if "actual" not in merged.columns:
        merged["actual"] = np.nan

    # fill only if NaN
    merged["actual"] = np.where(merged["actual"].isna(), merged["actual_new"], merged["actual"])
    merged = merged.drop(columns=["actual_new"])

    merged = recompute_errors(merged)
    return merged


def history_metrics(history: pd.DataFrame) -> Dict[str, Any]:
    h = history.copy()
    if "actual" not in h.columns:
        return {"n_pred": len(h), "n_matched": 0}

    n_pred = int(len(h))
    n_actual = int(h["actual"].notna().sum())
    hh = h[h["actual"].notna()].copy()
    if hh.empty:
        return {"n_pred": n_pred, "n_actual": n_actual, "n_matched": 0}

    y = hh["actual"].to_numpy(np.float32)
    p = hh["pred"].to_numpy(np.float32)

    mae = float(np.mean(np.abs(p - y)))
    mape = float(np.mean(np.abs(p - y) / np.maximum(np.abs(y), 1e-6)) * 100.0)

    by_step = (
        hh.groupby("step")
        .apply(lambda g: pd.Series({
            "n": int(len(g)),
            "mae": float(np.mean(np.abs(g["pred"] - g["actual"]))),
            "mape_%": float(np.mean(np.abs(g["pred"] - g["actual"]) / np.maximum(np.abs(g["actual"]), 1e-6)) * 100.0),
        }))
        .reset_index()
        .sort_values("step")
    )

    return {
        "n_pred": n_pred,
        "n_actual": n_actual,
        "n_matched": int(len(hh)),
        "mae": mae,
        "mape_%": mape,
        "by_step": by_step
    }


# =========================
# Pretty UI (LIGHT THEME)
# =========================
def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1200px; }

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

.small-note{ color: var(--muted); font-size: 12px; }

hr{
  border: 0;
  height: 1px;
  background: var(--bd);
  margin: 12px 0;
}

.stButton>button{
  border-radius: 12px !important;
  border: 1px solid rgba(124,58,237,0.25) !important;
  background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(34,197,94,0.80)) !important;
  color: white !important;
  font-weight: 700 !important;
  padding: 0.55rem 0.9rem !important;
  box-shadow: 0 10px 22px rgba(124,58,237,0.18) !important;
}
.stDownloadButton>button{
  border-radius: 12px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: rgba(255,255,255,0.90) !important;
  color: #0f172a !important;
  font-weight: 700 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Config (train)
# =========================
@dataclass
class CFG:
    DATE_COL: str = "Ngày"
    DROP_COLS: Tuple[str, ...] = ("Unnamed: 0",)

    # window + horizon
    K: int = 128
    H_MAX: int = 100
    TRAIN_H_LIST: Tuple[int, ...] = (5, 30)

    # training
    VAL_SIZE: int = 520
    BATCH: int = 128
    EPOCHS: int = 80
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

    # step weights
    FOCUS_STEPS: int = 5
    FOCUS_WEIGHT: float = 3.0

    # feature engineering
    ADD_RETURNS: bool = True
    RETURNS_CLIP: float = 0.15
    RETURN_COLS: Tuple[str, ...] = (
        "MG95", "MG92", "DO 0.001%", "DO 0.05%",
        "BRT DTD", "BRT KH", "WTI", "USD_Index", "GPRD"
    )

    # alpha tuning
    ALPHA_GRID: int = 101
    ALPHA_TUNE_H: int = 5


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mae_np(y_true, y_pred): return float(np.mean(np.abs(y_pred - y_true)))
def rmse_np(y_true, y_pred): return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
def mape_np(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)
def r2_np(y_true, y_pred, eps=1e-12):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + eps)

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
        "all": {
            "mu": np.array(s["all"]["mu"], dtype=np.float32),
            "sd": np.array(s["all"]["sd"], dtype=np.float32),
        },
        "targets": {
            "mu": np.array(s["targets"]["mu"], dtype=np.float32),
            "sd": np.array(s["targets"]["sd"], dtype=np.float32),
        },
    }


# =========================
# Data
# =========================
def load_and_prepare_xlsx(data_path: str, cfg: CFG) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    for c in cfg.DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if cfg.DATE_COL not in df.columns:
        raise ValueError(f"Missing date col: {cfg.DATE_COL}")

    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce")
    df = df.dropna(subset=[cfg.DATE_COL]).sort_values(cfg.DATE_COL).reset_index(drop=True)

    if cfg.ADD_RETURNS:
        for c in cfg.RETURN_COLS:
            if c in df.columns:
                ret = df[c].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                ret = ret.clip(-cfg.RETURNS_CLIP, cfg.RETURNS_CLIP)
                df[f"ret_{c}"] = ret

    feat_cols = [c for c in df.columns if c != cfg.DATE_COL]
    df[feat_cols] = df[feat_cols].interpolate(limit_direction="both").ffill().bfill()
    return df

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


# =========================
# Model
# =========================
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
        B, K, D = x.shape
        z = self.in_proj(x)

        _, h = self.gru(z)
        h_gru = self.gru_norm(h[-1])

        att_out, _ = self.att(z, z, z, need_weights=False)
        att_out = self.att_norm(att_out + z)
        h_att = self.att_ff(self.att_pool(att_out))

        h_kan = self.kan(self.pool(z))

        yk = self.head_kan(h_kan).view(B, self.H, self.T)
        yg = self.head_gru(h_gru).view(B, self.H, self.T)
        ya = self.head_att(h_att).view(B, self.H, self.T)

        w = torch.softmax(
            self.gate(torch.cat([h_kan, h_gru, h_att], dim=-1)).view(B, self.T, 3),
            dim=-1
        )
        pct = (torch.stack([yk, yg, ya], dim=-1) * w[:, None, :, :]).sum(dim=-1)

        base_rep = base_y[:, None, :].expand(B, self.H, self.T)
        pct = pct + self.mixer(torch.cat([pct, base_rep], dim=-1))
        return pct


# =========================
# Loss / Eval
# =========================
def horizon_weights(h: int, focus_steps: int, focus_weight: float, device):
    w = torch.ones(h, device=device)
    if focus_steps > 0:
        w[:focus_steps] = focus_weight
    return w

def loss_multi_horizon_pct(pred_pct, true_pct, train_h_list, focus_steps, focus_weight, use_huber, huber_beta):
    device = pred_pct.device
    total = 0.0
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
    assert T == 1
    h = min(h_tune, H)
    base = base_s[:, None, :]           # (N,1,1)
    denom = np.abs(base) + eps_pct
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

def eval_horizons_1target(pred_price_s, true_price_s, scalers, horizons):
    pred = inverse_target_1d(pred_price_s.reshape(-1, 1), scalers).reshape(pred_price_s.shape[0], pred_price_s.shape[1])
    true = inverse_target_1d(true_price_s.reshape(-1, 1), scalers).reshape(true_price_s.shape[0], true_price_s.shape[1])

    out = {}
    for h in horizons:
        ph = pred[:, :h].reshape(-1)
        th = true[:, :h].reshape(-1)
        out[h] = {
            "mae": mae_np(th, ph),
            "mape_%": mape_np(th, ph),
            "rmse": rmse_np(th, ph),
            "r2": r2_np(th, ph),
        }
    return out


# =========================
# Train core (streamlit)
# =========================
def train_streamlit(
    df: pd.DataFrame,
    target_col: str,
    save_path: str,
    cfg: CFG,
    device: torch.device,
    log_cb,
    progress_cb,
):
    set_seed(cfg.SEED)

    feature_cols = [c for c in df.columns if c != cfg.DATE_COL]
    if target_col not in feature_cols:
        raise ValueError(f"Target {target_col} not in feature cols.")

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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    use_amp = bool(cfg.AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    best_val = float("inf")
    best_ep = -1
    bad = 0
    patience = int(cfg.PATIENCE)
    train_h_list = list(cfg.TRAIN_H_LIST)

    log_cb(f"Device: {device} | AMP: {use_amp}")
    log_cb(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    log_cb(f"Features D={X.shape[1]} | Target={target_col}")
    log_cb(f"K={cfg.K} | H_MAX={cfg.H_MAX} | TRAIN_H_LIST={train_h_list}")

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

        log_cb(f"Epoch {ep:03d}/{cfg.EPOCHS} | lr={lr_now:.2e} | train={tr:.6f} val={va:.6f} | best={best_val:.6f} (ep {best_ep}) | {dt:.1f}s")
        progress_cb(ep / cfg.EPOCHS)

        if bad >= patience:
            log_cb("Early stopping.")
            break

    try:
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pred_pct_s, true_pct_s, base_s = predict_on_loader(model, val_loader, device)
    pred_pct_s = np.clip(pred_pct_s, -cfg.PCT_CLIP, cfg.PCT_CLIP)

    denom = np.abs(base_s)[:, None, :] + cfg.EPS_PCT
    true_price_s = base_s[:, None, :] + true_pct_s * denom
    naive_price_s = np.repeat(base_s[:, None, :], cfg.H_MAX, axis=1)

    best_a, best_m = tune_alpha_1target(base_s, pred_pct_s, true_pct_s, cfg.ALPHA_TUNE_H, cfg.EPS_PCT, cfg.ALPHA_GRID)
    tuned_price_s = base_s[:, None, :] + (best_a * pred_pct_s) * denom

    horizons = [5, 30, 60, 100]
    horizons = [h for h in horizons if h <= cfg.H_MAX]
    naive_metrics = eval_horizons_1target(naive_price_s[:, :, 0], true_price_s[:, :, 0], _restore_scalers(ckpt["scalers"]), horizons)
    tuned_metrics = eval_horizons_1target(tuned_price_s[:, :, 0], true_price_s[:, :, 0], _restore_scalers(ckpt["scalers"]), horizons)

    return {
        "save_path": save_path,
        "best_alpha": float(best_a),
        "tune_mape": float(best_m),
        "naive_metrics": naive_metrics,
        "tuned_metrics": tuned_metrics,
        "horizons": horizons,
        "ckpt": ckpt,
    }


# =========================
# Forecast (single point)
# =========================
@torch.no_grad()
def forecast_one(
    model: nn.Module,
    X_scaled: np.ndarray,           # (N,D)
    df_raw: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    scalers: Dict[str, Any],
    base_idx: int,
    K: int,
    H: int,
    eps_pct: float,
    pct_clip: float,
    alpha: float,
    device: torch.device,
):
    x_win = X_scaled[base_idx - K + 1: base_idx + 1]  # (K,D)
    x_t = torch.from_numpy(x_win).unsqueeze(0).to(device).float()

    target_idx = feature_cols.index(target_col)
    base_y_s = float(X_scaled[base_idx, target_idx])
    base_y_t = torch.tensor([[base_y_s]], device=device, dtype=torch.float32)

    pred_pct = model(x_t, base_y_t)  # (1,Hmax,1)
    pred_pct = torch.clamp(pred_pct, -pct_clip, pct_clip).cpu().numpy()[0, :H, 0]  # (H,)

    denom = abs(base_y_s) + eps_pct
    pred_price_s = base_y_s + (alpha * pred_pct) * denom  # (H,)
    pred_price_s = pred_price_s.reshape(-1, 1).astype(np.float32)

    pred_price = inverse_target_1d(pred_price_s, scalers).reshape(-1)
    base_orig = inverse_target_1d(np.array([[base_y_s]], dtype=np.float32), scalers).reshape(-1)[0]

    base_date = pd.Timestamp(df_raw[date_col].iloc[base_idx]).normalize()
    if base_idx + H < len(df_raw):
        fut_dates = pd.to_datetime(df_raw[date_col].iloc[base_idx + 1: base_idx + 1 + H]).dt.normalize().tolist()
        actual = df_raw[target_col].iloc[base_idx + 1: base_idx + 1 + H].to_numpy(dtype=np.float32)
    else:
        fut_dates = pd.bdate_range(base_date + pd.Timedelta(days=1), periods=H).to_pydatetime().tolist()
        actual = None

    return {
        "base_date": base_date,
        "base_idx": base_idx,
        "base_value": float(base_orig),
        "future_dates": fut_dates,
        "pred": pred_price.astype(np.float32),
        "actual": None if actual is None else actual.astype(np.float32),
        "pred_pct": pred_pct.astype(np.float32),
        "alpha": float(alpha),
    }


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="HybridTriNet • Train + Forecast + Monitoring (1 target)", page_icon="🧠", layout="wide")
    inject_css()

    st.markdown(
        """
<div class="hdr">
  <h1>🧠 HybridTriNet (pct) • Train + Forecast + Monitoring</h1>
  <p>Train 1 target • Forecast • Lưu lịch sử dự đoán (giữ lần đầu) • Upload actual nhiều lần để theo dõi dài hạn</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if "trained_ckpt_path" not in st.session_state:
        st.session_state.trained_ckpt_path = ""
    if "trained_best_alpha" not in st.session_state:
        st.session_state.trained_best_alpha = 1.0

    # Sidebar global
    with st.sidebar:
        st.markdown("### 📦 Input")
        data_path = st.text_input("Dataset (.xlsx)", value=r"D:\Anh_Thuy\Hybridtrinet_Oil\data\data_train\root.xlsx")
        default_ckpt = st.session_state.trained_ckpt_path or "best_hybrid_pct_1target.pt"
        ckpt_path = st.text_input("Checkpoint (.pt) dùng để Forecast", value=default_ckpt)

        st.markdown("---")
        has_cuda = torch.cuda.is_available()
        device_choice = st.selectbox("Device", options=(["cuda"] if has_cuda else []) + ["cpu"], index=0 if has_cuda else 0)
        device = torch.device(device_choice)

        st.markdown("---")
        st.markdown("### 🧾 Forecast history")
        history_path = st.text_input("History file (.csv)", value=HISTORY_DEFAULT_PATH)
        auto_append_history = st.checkbox("Tự động append forecast vào history", value=True)

        st.markdown("<div class='small-note'>Gợi ý: Forecast trước để tạo history, sau đó Monitoring → upload actual để cập nhật & cộng dồn metrics.</div>", unsafe_allow_html=True)

    if not os.path.exists(data_path):
        st.error(f"Không tìm thấy dataset: {data_path}")
        st.stop()

    tabs = st.tabs(["🧪 Train", "📈 Forecast", "📊 Monitoring"])

    # -------------------------
    # TRAIN TAB
    # -------------------------
    with tabs[0]:
        st.markdown("### 🧪 Train model (pct) cho 1 target")

        cfg = CFG()

        with st.spinner("Đang đọc dữ liệu..."):
            df = load_and_prepare_xlsx(data_path, cfg)

        numeric_cols = [c for c in df.columns if c != cfg.DATE_COL and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            st.error("Không có cột số để làm target.")
            st.stop()

        col1, col2 = st.columns([1.2, 1.0], gap="large")
        with col1:
            target_col = st.selectbox("Chọn target", options=numeric_cols, index=0)
            save_path = st.text_input("Lưu checkpoint ra file", value="best_hybrid_pct_1target.pt")

            cA, cB, cC, cD = st.columns(4)
            with cA: cfg.K = st.number_input("K", min_value=16, max_value=512, value=cfg.K, step=8)
            with cB: cfg.H_MAX = st.number_input("H_MAX", min_value=5, max_value=300, value=cfg.H_MAX, step=5)
            with cC: cfg.VAL_SIZE = st.number_input("VAL_SIZE", min_value=50, max_value=5000, value=cfg.VAL_SIZE, step=10)
            with cD: cfg.BATCH = st.number_input("BATCH", min_value=16, max_value=1024, value=cfg.BATCH, step=16)

            train_h_opts = [5, 10, 20, 30, 60, 100, 150, 200]
            chosen = st.multiselect("TRAIN_H_LIST", options=train_h_opts, default=list(cfg.TRAIN_H_LIST))
            cfg.TRAIN_H_LIST = tuple(sorted([int(x) for x in chosen if int(x) <= int(cfg.H_MAX)])) or (min(5, int(cfg.H_MAX)),)

            cE, cF, cG = st.columns(3)
            with cE: cfg.EPOCHS = st.number_input("EPOCHS", min_value=5, max_value=500, value=cfg.EPOCHS, step=5)
            with cF: cfg.LR = st.number_input("LR", min_value=1e-6, max_value=5e-3, value=cfg.LR, format="%.6f")
            with cG: cfg.WD = st.number_input("WD", min_value=0.0, max_value=0.2, value=cfg.WD, format="%.6f")

            cH, cI, cJ = st.columns(3)
            with cH: cfg.DROPOUT = st.slider("DROPOUT", 0.0, 0.6, float(cfg.DROPOUT), 0.01)
            with cI: cfg.GRAD_CLIP = st.slider("GRAD_CLIP", 0.0, 5.0, float(cfg.GRAD_CLIP), 0.1)
            with cJ: cfg.PATIENCE = st.number_input("PATIENCE", min_value=3, max_value=50, value=cfg.PATIENCE, step=1)

        with col2:
            st.markdown("#### 🧠 Model config")
            c1, c2, c3 = st.columns(3)
            with c1: cfg.D_MODEL = st.number_input("D_MODEL", 32, 256, cfg.D_MODEL, 8)
            with c2: cfg.D_HIDDEN = st.number_input("D_HIDDEN", 64, 512, cfg.D_HIDDEN, 16)
            with c3: cfg.N_HEADS = st.number_input("N_HEADS", 1, 16, cfg.N_HEADS, 1)

            cfg.RBF_M = st.number_input("RBF_M", 4, 32, cfg.RBF_M, 1)

            st.markdown("#### 📌 Pct / Loss")
            cfg.EPS_PCT = st.number_input("EPS_PCT", 1e-6, 1e-1, cfg.EPS_PCT, format="%.6f")
            cfg.PCT_CLIP = st.slider("PCT_CLIP", 0.02, 0.30, float(cfg.PCT_CLIP), 0.01)
            cfg.USE_HUBER = st.checkbox("USE_HUBER", value=cfg.USE_HUBER)
            cfg.HUBER_BETA = st.slider("HUBER_BETA", 0.005, 0.10, float(cfg.HUBER_BETA), 0.005)

            st.markdown("#### 🎯 Focus steps")
            cfg.FOCUS_STEPS = st.number_input("FOCUS_STEPS", 0, 30, cfg.FOCUS_STEPS, 1)
            cfg.FOCUS_WEIGHT = st.slider("FOCUS_WEIGHT", 1.0, 10.0, float(cfg.FOCUS_WEIGHT), 0.5)

            st.markdown("#### 🧾 Returns features")
            cfg.ADD_RETURNS = st.checkbox("ADD_RETURNS", value=cfg.ADD_RETURNS)
            cfg.RETURNS_CLIP = st.slider("RETURNS_CLIP", 0.02, 0.50, float(cfg.RETURNS_CLIP), 0.01)
            cfg.AMP = st.checkbox("AMP (chỉ hiệu lực nếu CUDA)", value=cfg.AMP)

            st.markdown("#### 🔧 Alpha tune")
            cfg.ALPHA_TUNE_H = st.number_input("ALPHA_TUNE_H", 1, int(cfg.H_MAX), int(min(cfg.ALPHA_TUNE_H, cfg.H_MAX)), 1)
            cfg.ALPHA_GRID = st.number_input("ALPHA_GRID", 11, 501, cfg.ALPHA_GRID, 10)

        st.markdown("<hr/>", unsafe_allow_html=True)

        run_train = st.button("🚀 Train ngay", use_container_width=True)

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
                with st.spinner("Đang train..."):
                    result = train_streamlit(
                        df=df,
                        target_col=target_col,
                        save_path=save_path,
                        cfg=cfg,
                        device=device,
                        log_cb=log_cb,
                        progress_cb=progress_cb,
                    )

                st.success(f"✅ Train xong! Saved: {result['save_path']}")
                st.session_state.trained_ckpt_path = result["save_path"]
                st.session_state.trained_best_alpha = result["best_alpha"]

                m1, m2, m3 = st.columns(3, gap="large")
                with m1:
                    st.markdown(f"<div class='card'><div class='k'>Best alpha</div><div class='v'>{result['best_alpha']:.3f}</div><div class='s'>tune H={cfg.ALPHA_TUNE_H}</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='card'><div class='k'>Tune MAPE</div><div class='v'>{result['tune_mape']:.3f}%</div><div class='s'>(VAL, H=tune)</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='card'><div class='k'>Target</div><div class='v'>{target_col}</div><div class='s'>H_MAX={cfg.H_MAX}</div></div>", unsafe_allow_html=True)

                st.markdown("### 📊 Metrics (VAL)")
                rows = []
                for h in result["horizons"]:
                    n = result["naive_metrics"][h]
                    t = result["tuned_metrics"][h]
                    rows.append({
                        "H": h,
                        "NAIVE_MAE": n["mae"],
                        "NAIVE_MAPE_%": n["mape_%"],
                        "TUNED_MAE": t["mae"],
                        "TUNED_MAPE_%": t["mape_%"],
                        "TUNED_R2": t["r2"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.info("👉 Forecast để tạo history, rồi Monitoring → upload actual để cập nhật & cộng dồn metrics.")
            except Exception as e:
                st.error(f"Lỗi train: {e}")

    # -------------------------
    # FORECAST TAB
    # -------------------------
    with tabs[1]:
        st.markdown("### 📈 Forecast từ checkpoint")

        if not os.path.exists(ckpt_path):
            st.warning("Chưa có checkpoint hợp lệ. Bạn có thể train ở tab Train hoặc nhập đúng đường dẫn ckpt (.pt).")
            st.stop()

        try:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            st.error(f"Lỗi load checkpoint: {e}")
            st.stop()

        feature_cols = ckpt.get("feature_cols", None)
        target_col = ckpt.get("target", None) or (ckpt.get("targets", [None])[0] if isinstance(ckpt.get("targets", None), list) else None)
        cfg_ckpt = ckpt.get("cfg", {})

        if not feature_cols or not target_col:
            st.error("Checkpoint thiếu `feature_cols` hoặc `target`.")
            st.stop()

        cfg_use = CFG()
        for k, v in cfg_ckpt.items():
            if hasattr(cfg_use, k):
                try:
                    setattr(cfg_use, k, v)
                except Exception:
                    pass

        with st.spinner("Đang đọc & chuẩn hoá dữ liệu..."):
            df = load_and_prepare_xlsx(data_path, cfg_use)

        missing = [c for c in feature_cols if c != cfg_use.DATE_COL and c not in df.columns]
        if missing:
            st.error("Dataset thiếu feature so với lúc train. Thiếu:\n- " + "\n- ".join(missing[:30]) + ("..." if len(missing) > 30 else ""))
            st.stop()

        scalers = _restore_scalers(ckpt["scalers"])
        df_s = apply_scaling(df, feature_cols, scalers)
        X_scaled = df_s[feature_cols].to_numpy(np.float32)

        H_MAX = int(cfg_ckpt.get("H_MAX", cfg_use.H_MAX))
        d_model = int(cfg_ckpt.get("D_MODEL", cfg_use.D_MODEL))
        d_hidden = int(cfg_ckpt.get("D_HIDDEN", cfg_use.D_HIDDEN))
        n_heads = int(cfg_ckpt.get("N_HEADS", cfg_use.N_HEADS))
        dropout = float(cfg_ckpt.get("DROPOUT", cfg_use.DROPOUT))
        rbf_m = int(cfg_ckpt.get("RBF_M", cfg_use.RBF_M))
        K = int(cfg_ckpt.get("K", cfg_use.K))

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

        colA, colB = st.columns([1.3, 1.0], gap="large")
        with colA:
            dmin = pd.Timestamp(df[cfg_use.DATE_COL].min()).date()
            dmax = pd.Timestamp(df[cfg_use.DATE_COL].max()).date()
            base_date = st.date_input("Base date (ngày t)", value=dmax, min_value=dmin, max_value=dmax)

            H = st.select_slider(
                "Horizon H",
                options=[h for h in [5, 30, 60, 100] if h <= H_MAX] or [min(30, H_MAX)],
                value=min(30, H_MAX)
            )
            alpha_default = float(st.session_state.trained_best_alpha) if st.session_state.trained_ckpt_path == ckpt_path else 1.0
            alpha = st.slider("Alpha (0=NAIVE, 1=full)", 0.0, 1.0, float(alpha_default), 0.01)

            run_fc = st.button("🚀 Forecast", use_container_width=True)
            st.markdown(
                f"<div class='small-note'>Target: <b>{target_col}</b> • K={K} • H_MAX={H_MAX} • device={device.type}</div>",
                unsafe_allow_html=True
            )

        with colB:
            st.markdown("### 🗂️ Thông tin")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='card'><div class='k'>Rows</div><div class='v'>{len(df):,}</div><div class='s'>Clean/fill</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='card'><div class='k'>Date range</div><div class='v'>{dmin}</div><div class='s'>→ {dmax}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='card'><div class='k'>History</div><div class='v'>{os.path.basename(history_path)}</div><div class='s'>Auto-append: {auto_append_history}</div></div>", unsafe_allow_html=True)

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("**Tip:** Forecast xong sẽ tự lưu vào history. Monitoring → upload actual để cộng dồn metrics.")

        if not run_fc:
            st.stop()

        base_ts = pd.Timestamp(base_date).normalize()
        dates_norm = pd.to_datetime(df[cfg_use.DATE_COL]).dt.normalize()
        idxs = np.where(dates_norm.values == np.datetime64(base_ts))[0]
        if len(idxs) == 0:
            st.error("Không tìm thấy base date trong dataset.")
            st.stop()
        base_idx = int(idxs[-1])
        if base_idx < K - 1:
            st.error(f"Base date quá sớm so với K={K}.")
            st.stop()

        eps_pct = float(cfg_ckpt.get("EPS_PCT", cfg_use.EPS_PCT))
        pct_clip = float(cfg_ckpt.get("PCT_CLIP", cfg_use.PCT_CLIP))

        with st.spinner("Đang dự đoán..."):
            out = forecast_one(
                model=model,
                X_scaled=X_scaled,
                df_raw=df,
                date_col=cfg_use.DATE_COL,
                target_col=target_col,
                feature_cols=feature_cols,
                scalers=scalers,
                base_idx=base_idx,
                K=K,
                H=int(H),
                eps_pct=eps_pct,
                pct_clip=pct_clip,
                alpha=float(alpha),
                device=device,
            )

        df_out = pd.DataFrame({
            "date": [pd.Timestamp(x).normalize() for x in out["future_dates"]],
            "pred": out["pred"],
            "pred_pct": out["pred_pct"],
        })
        if out["actual"] is not None:
            df_out["actual"] = out["actual"]
            df_out["abs_err"] = np.abs(df_out["pred"] - df_out["actual"])
            df_out["mape_%"] = (df_out["abs_err"] / np.maximum(np.abs(df_out["actual"]), 1e-6)) * 100.0

        # append to history
        if auto_append_history:
            run_id = str(uuid.uuid4())[:8]
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_hist_add = df_out.copy()
            df_hist_add["target"] = target_col
            df_hist_add["step"] = np.arange(1, len(df_hist_add) + 1, dtype=int)
            df_hist_add["actual"] = np.nan
            df_hist_add["base_date"] = out["base_date"]
            df_hist_add["horizon"] = int(H)
            df_hist_add["alpha"] = float(alpha)
            df_hist_add["created_at"] = created_at
            df_hist_add["run_id"] = run_id
            df_hist_add = df_hist_add[[
                "date", "target", "step", "pred", "actual",
                "base_date", "horizon", "alpha", "created_at", "run_id"
            ]]

            hist = load_history(history_path)
            before = len(hist)
            hist = append_forecast_keep_first(hist, df_hist_add)
            after = len(hist)
            save_history(hist, history_path)
            st.success(f"✅ Đã lưu vào history (run_id={run_id}). Thêm mới: {after - before} dòng. Tổng: {after} dòng.")

        st.markdown("### 📊 Kết quả")
        m1, m2, m3, m4 = st.columns(4, gap="large")
        with m1:
            st.markdown(f"<div class='card'><div class='k'>Base date</div><div class='v'>{out['base_date'].date()}</div><div class='s'>idx={out['base_idx']}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='card'><div class='k'>Base value</div><div class='v'>{out['base_value']:,.0f}</div><div class='s'>{target_col}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='card'><div class='k'>Horizon</div><div class='v'>H={int(H)}</div><div class='s'>alpha={alpha:.2f}</div></div>", unsafe_allow_html=True)
        with m4:
            if out["actual"] is not None:
                mae = mae_np(df_out["actual"].values, df_out["pred"].values)
                mape = mape_np(df_out["actual"].values, df_out["pred"].values)
                st.markdown(f"<div class='card'><div class='k'>Metrics</div><div class='v'>MAPE {mape:.3f}%</div><div class='s'>MAE {mae:,.3f}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card'><div class='k'>Metrics</div><div class='v'>N/A</div><div class='s'>Chưa có actual tương lai</div></div>", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📈 Chart", "📋 Table"])
        with tab1:
            plot_df = pd.DataFrame({
                "date": [out["base_date"]] + list(df_out["date"].values),
                "pred": [out["base_value"]] + list(df_out["pred"].values),
            })
            if out["actual"] is not None:
                plot_df["actual"] = [out["base_value"]] + [float(x) for x in df_out["actual"].values]
            st.line_chart(plot_df.set_index("date"))

        with tab2:
            st.dataframe(df_out, use_container_width=True, height=420)
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Tải forecast CSV",
                data=csv_bytes,
                file_name=f"forecast_{target_col}_{out['base_date'].date()}_H{int(H)}.csv",
                mime="text/csv",
            )

    # -------------------------
    # MONITORING TAB
    # -------------------------
    with tabs[2]:
        st.markdown("### 📊 Monitoring • forecast_history (giữ lần dự đoán đầu tiên)")
        st.markdown("<div class='small-note'>Upload actual nhiều lần. App sẽ cộng dồn và theo dõi MAE/MAPE theo thời gian.</div>", unsafe_allow_html=True)

        hist = load_history(history_path)

        # summary cards (always)
        m = history_metrics(hist) if not hist.empty else {"n_pred": 0, "n_actual": 0, "n_matched": 0}
        c1, c2, c3, c4 = st.columns(4, gap="large")
        with c1:
            st.markdown(f"<div class='card'><div class='k'>#Pred rows</div><div class='v'>{m.get('n_pred',0):,}</div><div class='s'>Trong history</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='card'><div class='k'>#Actual filled</div><div class='v'>{m.get('n_actual',0):,}</div><div class='s'>Đã update actual</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='card'><div class='k'>#Matched</div><div class='v'>{m.get('n_matched',0):,}</div><div class='s'>Có pred & actual</div></div>", unsafe_allow_html=True)
        with c4:
            if m.get("n_matched", 0) > 0:
                st.markdown(f"<div class='card'><div class='k'>Overall MAPE</div><div class='v'>{m['mape_%']:.3f}%</div><div class='s'>MAE {m['mae']:.3f}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card'><div class='k'>Overall MAPE</div><div class='v'>N/A</div><div class='s'>Chưa có match</div></div>", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("#### 📎 Upload file actual để cập nhật history")
        up = st.file_uploader("Upload actual (xlsx/csv)", type=["xlsx", "csv"], key="actual_uploader")

        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    df_act = pd.read_csv(up)
                else:
                    df_act = pd.read_excel(up)
            except Exception as e:
                st.error(f"Lỗi đọc file actual: {e}")
                st.stop()

            st.markdown("**Preview file actual (20 dòng đầu):**")
            st.dataframe(df_act.head(20), use_container_width=True)

            dcol = _guess_date_col(df_act)

            targets_in_hist = sorted(hist["target"].astype(str).unique().tolist()) if (not hist.empty and "target" in hist.columns) else []
            if len(targets_in_hist) == 0:
                st.warning("History đang trống. Hãy Forecast ít nhất 1 lần trước.")
                pick_target = None
            else:
                pick_target = st.selectbox("Target cần update actual", options=targets_in_hist, index=0)

            vcol = _guess_value_col(df_act, pick_target) if pick_target else None
            st.write("Nhận diện cột:", {"date_col": dcol, "value_col": vcol})

            # pre-check matched dates
            if (pick_target is not None) and (dcol is not None) and (vcol is not None):
                act2 = df_act[[dcol, vcol]].copy()
                act2[dcol] = _norm_date(act2[dcol])
                act2 = act2.dropna(subset=[dcol])
                act2 = act2.rename(columns={dcol: "date", vcol: "actual_val"})
                act2["target"] = pick_target

                hist_t = hist[hist["target"].astype(str) == str(pick_target)].copy()
                hist_t["date"] = _norm_date(hist_t["date"])

                hist_dates = set(hist_t["date"].dropna().astype("datetime64[ns]").unique())
                act_dates = set(act2["date"].dropna().astype("datetime64[ns]").unique())
                inter = sorted(hist_dates & act_dates)

                st.info(
                    f"Actual parsed rows={len(act2):,} | History rows(target)={len(hist_t):,} | Matched dates={len(inter):,}"
                )

                if len(inter) == 0:
                    st.warning("⚠️ 0 ngày khớp. Thường do lệch format ngày / chọn sai cột / hoặc dự đoán BusinessDay.")
                    st.write("Ví dụ 5 ngày dự đoán (history):", sorted(list(hist_dates))[:5])
                    st.write("Ví dụ 5 ngày trong actual:", sorted(list(act_dates))[:5])

            if st.button("✅ Cập nhật history bằng actual (cộng dồn)", use_container_width=True):
                if hist.empty:
                    st.error("History trống → Forecast ít nhất 1 lần trước.")
                elif pick_target is None:
                    st.error("Không có target trong history để update.")
                elif dcol is None:
                    st.error("Không nhận ra cột ngày trong file actual.")
                elif vcol is None:
                    st.error("Không nhận ra cột giá trị trong file actual.")
                else:
                    hist2 = update_history_with_actual(hist, df_act, dcol, vcol, pick_target)
                    save_history(hist2, history_path)
                    st.success("✅ Đã cập nhật history bằng actual!")

                    # reload + show metrics
                    hist = load_history(history_path)
                    m2 = history_metrics(hist[hist["target"].astype(str) == str(pick_target)])
                    if m2.get("n_matched", 0) == 0:
                        st.warning("Đã update nhưng vẫn chưa có ngày nào khớp để tính metrics.")
                    else:
                        st.success(f"Match={m2['n_matched']} | MAE={m2['mae']:.4f} | MAPE={m2['mape_%']:.3f}%")
                        st.markdown("##### Metrics theo step")
                        st.dataframe(m2["by_step"], use_container_width=True)

                    st.markdown("**History (tail 60):**")
                    st.dataframe(hist.tail(60), use_container_width=True)

        st.markdown("#### 📋 Xem history")
        hist = load_history(history_path)
        if hist.empty:
            st.info("History trống.")
        else:
            targets = sorted(hist["target"].astype(str).unique().tolist())
            sel_t = st.selectbox("Lọc target", options=["(ALL)"] + targets, index=0)
            view = hist.copy()
            if sel_t != "(ALL)":
                view = view[view["target"].astype(str) == sel_t]
            view = view.sort_values("date").reset_index(drop=True)

            st.dataframe(view.tail(300), use_container_width=True, height=420)

            csv_bytes = view.copy()
            if "date" in csv_bytes.columns:
                csv_bytes["date"] = pd.to_datetime(csv_bytes["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            if "base_date" in csv_bytes.columns:
                csv_bytes["base_date"] = pd.to_datetime(csv_bytes["base_date"], errors="coerce").dt.strftime("%Y-%m-%d")

            st.download_button(
                "⬇️ Tải history CSV (đang lọc)",
                data=csv_bytes.to_csv(index=False).encode("utf-8"),
                file_name="forecast_history_filtered.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()