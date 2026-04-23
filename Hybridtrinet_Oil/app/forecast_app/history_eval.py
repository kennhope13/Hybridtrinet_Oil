# forecast_app/history_eval.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


TARGETS_DEFAULT = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
DATE_CANDIDATES = ["date", "ngay", "ngày", "datetime", "time", "ds"]


def _norm(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _pick_date_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    norm_cols = [_norm(c) for c in cols]

    for cand in DATE_CANDIDATES:
        candn = _norm(cand)
        if candn in norm_cols:
            return cols[norm_cols.index(candn)]

    best_col, best_ok = None, -1
    for c in cols[:10]:
        s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        ok = int(s.notna().sum())
        if ok > best_ok:
            best_ok = ok
            best_col = c
    if best_col is None or best_ok <= 0:
        raise ValueError("Không tìm thấy cột ngày (date) hợp lệ trong file.")
    return best_col


def _find_target_cols(df: pd.DataFrame, targets: List[str]) -> Dict[str, str]:
    cols = list(df.columns)
    norm_cols = [_norm(c) for c in cols]

    out: Dict[str, str] = {}
    for t in targets:
        tn = _norm(t)

        if tn in norm_cols:
            out[t] = cols[norm_cols.index(tn)]
            continue

        hits = []
        for c, cn in zip(cols, norm_cols):
            if tn in cn:
                hits.append(c)

        if hits:

            def score(name: str) -> int:
                n = _norm(name)
                if "cal" in n or "calib" in n:
                    return 3
                if "pred" in n or "forecast" in n or "yhat" in n:
                    return 2
                return 1

            hits = sorted(hits, key=score, reverse=True)
            out[t] = hits[0]

    return out


@dataclass
class HistoryFileMeta:
    path: Path
    forecast_until: Optional[pd.Timestamp]
    horizon_h: Optional[int]
    asof: Optional[pd.Timestamp]


# ✅ hỗ trợ cả YYYYMMDD và YYYYMMDD_HHMMSS
_HISTORY_RE = re.compile(
    r"forecast_until_(?P<until>\d{8})_H(?P<h>\d+?)_(?P<asof>\d{8}(?:_\d{6})?)(?:\D|$)"
)


def _parse_history_filename(p: Path) -> HistoryFileMeta:
    m = _HISTORY_RE.search(p.name)
    if not m:
        return HistoryFileMeta(p, None, None, None)

    until = pd.to_datetime(m.group("until"), format="%Y%m%d", errors="coerce")
    h = int(m.group("h")) if m.group("h") else None

    asof_raw = m.group("asof")
    if "_" in asof_raw:
        asof = pd.to_datetime(asof_raw, format="%Y%m%d_%H%M%S", errors="coerce")
    else:
        asof = pd.to_datetime(asof_raw, format="%Y%m%d", errors="coerce")

    return HistoryFileMeta(p, until, h, asof)


def _read_forecast_file(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(path)
        preferred = None
        for s in xls.sheet_names:
            sn = _norm(s)
            if sn in ["forecast", "pred", "predict", "output", "sheet1"]:
                preferred = s
                break
        sheet = preferred or xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Không hỗ trợ định dạng file: {path.name}")


def load_forecast_history_long(
    history_dir: str | Path,
    targets: List[str] = TARGETS_DEFAULT,
) -> pd.DataFrame:
    """
    Output long-form:
      [date, target, yhat, source_file, asof, forecast_until, horizon_h]
    """
    history_dir = Path(history_dir)
    files = sorted([p for p in history_dir.glob("*") if p.is_file()])

    rows = []
    for fp in files:
        meta = _parse_history_filename(fp)
        try:
            df = _read_forecast_file(fp)
        except Exception:
            continue

        date_col = _pick_date_col(df)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df = df[df[date_col].notna()]
        df["date"] = df[date_col].dt.normalize()

        tcols = _find_target_cols(df, targets)
        if not tcols:
            continue

        for t, c in tcols.items():
            tmp = df[["date", c]].rename(columns={c: "yhat"})
            tmp["target"] = t
            tmp["source_file"] = fp.name
            tmp["asof"] = meta.asof
            tmp["forecast_until"] = meta.forecast_until
            tmp["horizon_h"] = meta.horizon_h
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=["date", "target", "yhat", "source_file", "asof", "forecast_until", "horizon_h"]
        )

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["yhat"] = pd.to_numeric(out["yhat"], errors="coerce")
    out = out.dropna(subset=["date", "target", "yhat"])
    return out


def load_actual_root(
    root_xlsx: str | Path,
    targets: List[str] = TARGETS_DEFAULT,
) -> pd.DataFrame:
    """
    Output wide:
      [date, MG95, MG92, DO 0.001%, DO 0.05%] (chỉ lấy các cột tìm thấy)
    """
    root_xlsx = Path(root_xlsx)
    xls = pd.ExcelFile(root_xlsx)
    df = pd.read_excel(root_xlsx, sheet_name=xls.sheet_names[0])

    date_col = _pick_date_col(df)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df[df[date_col].notna()]
    df["date"] = df[date_col].dt.normalize()

    tcols = _find_target_cols(df, targets)

    keep = ["date"]
    rename = {}
    for t, c in tcols.items():
        keep.append(c)
        rename[c] = t

    out = df[keep].rename(columns=rename)
    for t in targets:
        if t in out.columns:
            out[t] = pd.to_numeric(out[t], errors="coerce")

    out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out


def build_compare_actual_vs_pred(
    actual_wide: pd.DataFrame,
    history_long: pd.DataFrame,
    strategy: str = "latest_asof",
) -> pd.DataFrame:
    targets = [c for c in actual_wide.columns if c != "date"]
    act_long = actual_wide.melt(id_vars=["date"], value_vars=targets, var_name="target", value_name="actual")
    act_long = act_long.dropna(subset=["actual"])

    hist = history_long.copy()
    if strategy == "latest_asof":
        hist["_asof_sort"] = pd.to_datetime(hist["asof"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        hist = hist.sort_values(["date", "target", "_asof_sort"]).drop_duplicates(["date", "target"], keep="last")
        hist = hist.drop(columns=["_asof_sort"])

    merged = act_long.merge(
        hist[["date", "target", "yhat", "asof", "source_file"]],
        on=["date", "target"],
        how="inner",
    ).rename(columns={"yhat": "pred"})

    merged = merged.sort_values(["target", "date"]).reset_index(drop=True)
    return merged


def compute_metrics(compare_long: pd.DataFrame) -> pd.DataFrame:
    """
    MAE, MAPE, MSE, RMSE, R2 theo target + overall.
    Ẩn n (không hiển thị số mẫu).
    An toàn khi compare_long rỗng / thiếu actual (NaN) -> trả NaN, không crash.
    """
    def r2(y, yhat):
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    cols = ["target", "MAE", "MAPE_%", "MSE", "RMSE", "R2"]

    if compare_long is None or len(compare_long) == 0:
        return pd.DataFrame(columns=cols)

    # ép numeric + lọc cặp hợp lệ (để ngày lễ thiếu actual không làm lỗi)
    df = compare_long.copy()
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["pred"]   = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["target"])  # target bắt buộc
    # chỉ tính metric trên các dòng có đủ actual+pred
    df_valid = df.dropna(subset=["actual", "pred"])

    rows = []

    # per-target
    for t, g in df_valid.groupby("target"):
        y = g["actual"].to_numpy(dtype=float)
        yhat = g["pred"].to_numpy(dtype=float)

        err = y - yhat
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs(err) / np.clip(np.abs(y), 1e-9, None)) * 100.0)
        rows.append([t, mae, mape, mse, rmse, r2(y, yhat)])  # <-- 6 phần tử

    # overall (nếu không có cặp hợp lệ thì overall = NaN)
    if len(df_valid) == 0:
        rows.append(["__OVERALL__", np.nan, np.nan, np.nan, np.nan, np.nan])
    else:
        y = df_valid["actual"].to_numpy(dtype=float)
        yhat = df_valid["pred"].to_numpy(dtype=float)
        err = y - yhat
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs(err) / np.clip(np.abs(y), 1e-9, None)) * 100.0)
        rows.append(["__OVERALL__", mae, mape, mse, rmse, r2(y, yhat)])  # <-- 6 phần tử

    return pd.DataFrame(rows, columns=cols)
