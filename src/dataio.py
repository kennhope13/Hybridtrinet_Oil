import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .features import _add_calendar_features, _norm
from .external_api import _fetch_usd_index_fred, _fetch_gprd


def _ensure_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out = out.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return out


def _read_any_upload(file):
    name = file.name.lower()
    bio = io.BytesIO(file.getvalue())
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    elif name.endswith(".xlsx"):
        return pd.read_excel(bio, engine="openpyxl")
    elif name.endswith(".xls"):
        try:
            return pd.read_excel(bio, engine="xlrd")
        except Exception:
            st.error("Đọc .xls cần 'xlrd' (pip install xlrd).")
            raise
    else:
        st.error("Định dạng chưa hỗ trợ. Dùng .csv/.xlsx/.xls")
        return None


def _align_union_columns(df1: pd.DataFrame, df2: pd.DataFrame, date_col: str):
    cols_u = list(
        dict.fromkeys(
            [date_col]
            + [c for c in df1.columns if c != date_col]
            + [c for c in df2.columns if c != date_col]
        )
    )
    return df1.reindex(columns=cols_u), df2.reindex(columns=cols_u), cols_u


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    bio.seek(0)
    return bio.getvalue()


def build_merged(up_pp, clean_path: str, date_col: str, fill_mode: str, fred_api_key: str):
    assert clean_path, "Chưa nhập đường dẫn du_lieu_noi_suy_clean (.xlsx)."
    p = Path(clean_path)
    assert p.exists(), f"Không tìm thấy file: {clean_path}"

    base = pd.read_excel(p, engine="openpyxl")
    assert date_col in base.columns, f"Thiếu cột ngày '{date_col}' trong du_lieu_noi_suy_clean"
    base = _ensure_date(base, date_col)
    last_date = base[date_col].max()

    df_pp = _read_any_upload(up_pp)
    df_pp = df_pp.loc[:, ~df_pp.columns.str.contains("^Unnamed")]

    if len(df_pp):
        first_row_str = " ".join(str(x).lower() for x in df_pp.iloc[0].values)
        if ("đơn vị" in first_row_str) or ("don vi" in first_row_str) or ("usd" in first_row_str):
            df_pp = df_pp.iloc[1:].reset_index(drop=True)

    date_pp = date_col if date_col in df_pp.columns else None
    if date_pp is None:
        for c in df_pp.columns:
            if _norm(c) in ("ngay", "date"):
                date_pp = c
                break
    assert date_pp is not None, "Không tìm được cột ngày trong price_petroleum"

    df_pp = _ensure_date(df_pp, date_pp)
    df_pp = df_pp[df_pp[date_pp] > last_date].copy()
    if df_pp.empty:
        return None, None, "price_petroleum không có ngày nào > last_date, không có dữ liệu mới."

    def _find(dfX, tokens):
        for c in dfX.columns:
            if any(t in _norm(c) for t in tokens):
                return c
        return None

    pp_map = {
        "MG95": _find(df_pp, ["mg95", "ron95", "xang ron95"]),
        "MG92": _find(df_pp, ["mg92", "ron92", "xang ron92"]),
        "DO 0.001%": _find(df_pp, ["do 0 001", "do 0 001 percent", "diesel 0 001"]),
        "DO 0.05%": _find(df_pp, ["do 0 05", "do 0 05 percent", "diesel 0 05"]),
        "BRT DTD": _find(df_pp, ["brt dtd", "brent dated", "brent dtd"]),
        "BRT KH": _find(df_pp, ["brt kh", "brent kh", "brent futures"]),
        "WTI": _find(df_pp, ["wti"]),
    }

    keep_pp = [date_pp] + [c for c in pp_map.values() if c is not None]
    rename_pp = {src: dst for dst, src in pp_map.items() if src is not None}

    df_pp_small = (
        df_pp[keep_pp]
        .rename(columns=rename_pp)
        .rename(columns={date_pp: date_col})
    )

    allowed_dates = set(pd.to_datetime(df_pp_small[date_col]).dt.normalize())

    df_dtw = _fetch_usd_index_fred(allowed_dates, fred_api_key, date_col)
    df_gpr_out = _fetch_gprd(allowed_dates, date_col)

    df_new = (
        df_pp_small
        .merge(df_dtw, on=date_col, how="left")
        .merge(df_gpr_out, on=date_col, how="left")
    )
    df_new = df_new.sort_values(date_col).reset_index(drop=True)

    numeric_cols = [c for c in df_new.columns if c != date_col]

    if fill_mode == "ffill":
        df_new[numeric_cols] = df_new[numeric_cols].ffill()
    elif fill_mode == "ffill+bfill":
        df_new[numeric_cols] = df_new[numeric_cols].ffill().bfill()
    elif fill_mode == "drop rows with any NaN":
        df_new = df_new.dropna(subset=numeric_cols)

    df_new = _add_calendar_features(df_new, date_col)

    desired = [
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
    order = [c for c in desired if c in df_new.columns] + [c for c in df_new.columns if c not in desired]
    df_new = df_new[order]

    info = f"baseline=price_petroleum | new_rows={len(df_new)} | last_date={last_date.date()}"
    return df_new, base, info
