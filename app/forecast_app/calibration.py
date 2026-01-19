# forecast_app/calibration.py
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


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
