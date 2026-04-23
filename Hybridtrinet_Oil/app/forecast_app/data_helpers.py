# forecast_app/data_helpers.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st


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
