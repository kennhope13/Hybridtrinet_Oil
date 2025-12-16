import pandas as pd
from pathlib import Path

def cut_from_20251119_to_new_file(src_path: str, dst_path: str, date_col: str = "Ngày"):
    df = pd.read_excel(src_path, engine="openpyxl")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = pd.Timestamp("2025-11-20")
    df_cut = df[df[date_col] < cutoff].reset_index(drop=True)
    df_cut.to_excel(dst_path, index=False, engine="openpyxl")
    return df_cut



src = r"D:\HybridTrinet_oil\data\base\du_lieu_noi_suy_clean.xlsx"
dst = r"D:\HybridTrinet_oil\data\base\root.xlsx"

df_cut = cut_from_20251119_to_new_file(src, dst, date_col="Ngày")
