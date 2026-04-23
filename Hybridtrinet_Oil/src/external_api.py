from typing import Iterable

import pandas as pd
import requests


def _fetch_usd_index_fred(allowed_dates: Iterable[pd.Timestamp], api_key: str, date_col: str) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Chưa nhập FRED API key")

    allowed_dates = [pd.to_datetime(d).normalize() for d in allowed_dates]
    if not allowed_dates:
        raise ValueError("allowed_dates rỗng")

    dates_sorted = sorted(allowed_dates)
    start = dates_sorted[0].strftime("%Y-%m-%d")
    end = dates_sorted[-1].strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DTWEXBGS",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        raise ValueError("Không lấy được dữ liệu DTWEXBGS từ FRED")

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df["date"] = df["date"].dt.normalize()
    df = df[df["date"].isin(dates_sorted)].copy()
    df = df.rename(columns={"date": date_col, "value": "USD_Index"})
    return df[[date_col, "USD_Index"]]


def _fetch_gprd(allowed_dates: Iterable[pd.Timestamp], date_col: str) -> pd.DataFrame:
    url_gprd = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    g = pd.read_excel(url_gprd, sheet_name=0)

    day_str = g["DAY"].astype(str).str.zfill(8)
    g["Date"] = pd.to_datetime(day_str, format="%Y%m%d", errors="coerce")

    if "GPRD" not in g.columns:
        raise ValueError("Không tìm thấy cột GPRD trong file GPRD")

    g["GPRD"] = pd.to_numeric(g["GPRD"], errors="coerce")
    g = g.dropna(subset=["Date", "GPRD"])
    g["Date"] = g["Date"].dt.normalize()

    allowed_dates = [pd.to_datetime(d).normalize() for d in allowed_dates]
    g = g[g["Date"].isin(allowed_dates)].copy()
    g = g.rename(columns={"Date": date_col})
    return g[[date_col, "GPRD"]]
