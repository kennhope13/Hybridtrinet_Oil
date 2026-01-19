# forecast_app/history_eval.py
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from .metrics import _nan_mae, _nan_mape, _nan_mse, _nan_rmse, _r2_stats
from .data_helpers import _parse_dates_any
from src.utils.paths import RUN_OUTPUT_DIR


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

        macro__n = 0
        macro__sse = np.nan
        macro__sum_y = np.nan
        macro__sum_y2 = np.nan

        per = {}
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
            "macro__n": macro__n,
            "macro__sse": macro__sse,
            "macro__sum_y": macro__sum_y,
            "macro__sum_y2": macro__sum_y2,
        }

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
