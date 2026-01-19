# forecast_app/metrics.py
import numpy as np
import pandas as pd


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


def _r2_stats(true_arr, pred_arr):
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

    from .data_helpers import _parse_dates_any
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
