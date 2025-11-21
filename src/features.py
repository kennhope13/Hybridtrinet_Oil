import re
from typing import List

import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col], errors="coerce")
    dow = d.dt.weekday
    df["NgayTrongTuan"] = (dow + 1).where(dow < 5, 0)
    df["ThangTrongNam"] = d.dt.month
    df["QuyTrongNam"] = ((df["ThangTrongNam"] - 1) // 3) + 1
    df["Nam"] = d.dt.year
    df["NgayLe"] = 0
    df["SuKienDacBiet"] = 0
    return df


def _coerce_targets_numeric(df: pd.DataFrame, targets: List[str]) -> np.ndarray:
    dfT = df[targets].copy()
    for c in targets:
        s = dfT[c]
        if ptypes.is_datetime64_any_dtype(s):
            dfT[c] = np.nan
            continue
        if s.dtype == object:
            s = (
                s.astype(str)
                .str.replace(r"[\s,]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.replace("−", "-", regex=False)
            )
        dfT[c] = pd.to_numeric(s, errors="coerce")
    dfT = dfT.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    still_nan = [c for c in targets if dfT[c].isna().any()]
    if still_nan:
        raise ValueError(f"Các cột target vẫn còn NaN sau khi chuyển số: {still_nan}")
    return dfT.values.astype(float)


def _force_numeric_for_plot(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c]
        if ptypes.is_numeric_dtype(s):
            continue
        s = (
            s.astype(str)
            .str.replace(r"[\s,]", "", regex=True)
            .str.replace("%", "", regex=False)
            .str.replace("−", "-", regex=False)
        )
        out[c] = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
