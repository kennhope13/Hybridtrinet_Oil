"""data_pipeline.py — Bộ kiểm định schema, phân loại và hợp nhất dữ liệu cho Production Hub.

Độc lập hoàn toàn với Streamlit UI để đảm bảo logic nghiệp vụ nhất quán, không lỗi rerun.
Theo đúng quy chuẩn tại KE_HOACH_FRONTEND_VA_LUONG_TU_DONG.md:
- Kiểm tra độc lập từng file (1 file sai không làm hủy các file đúng).
- Nhận diện SHA-256 nội dung.
- Kiểm tra cột Ngày, các cột giá mục tiêu (MG95, MG92, DO...).
- Phân loại: Dữ liệu mới, Dữ liệu trùng, Dữ liệu điều chỉnh, File sai định dạng.
- Hợp nhất an toàn và ghi nguyên tử vào datasets/.
"""

import hashlib
import io
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
HISTORY_FILE = ROOT / "ingestion_history.json"

TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
DATE_COL = "Ngày"
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB


def sanitize_filename(filename: str) -> str:
    """Loại bỏ ký tự nguy hiểm và path traversal."""
    cleaned = Path(filename).name
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", cleaned)
    cleaned = cleaned.strip()
    return cleaned or f"upload_{uuid.uuid4().hex[:8]}.csv"


def compute_sha256(content_bytes: bytes) -> str:
    """Tính mã băm SHA-256 của nội dung file."""
    return hashlib.sha256(content_bytes).hexdigest()


def normalize_date_column_name(columns: List[str]) -> Dict[str, str]:
    """Tìm và map cột ngày sang tên chuẩn DATE_COL ('Ngày')."""
    mapping = {}
    for col in columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()
        normalized = re.sub(r"[^a-z0-9]+", "", col_lower)
        # Chỉ nhận alias rõ ràng. Từ khóa ngắn như "ng" từng có thể biến nhầm
        # một cột giá chứa chữ "xăng" thành cột Ngày.
        if normalized in {"ngay", "date", "datetime", "timestamp", "time"}:
            mapping[col] = DATE_COL
            break
    return mapping


def parse_and_validate_dataframe(
    raw_content: Union[bytes, io.BytesIO, Path, str],
    filename: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, Any]]:
    """Đọc và kiểm tra schema chi tiết của một file tải lên.

    Trả về: (df_clean, error_message, metadata)
    """
    meta = {
        "filename": filename,
        "is_valid": False,
        "error": None,
        "rows": 0,
        "min_date": None,
        "max_date": None,
        "price_cols": [],
        "sha256": None,
        "new_rows_count": 0,
    }

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        meta["error"] = f"Định dạng {ext} không được hỗ trợ. Chỉ nhận file .xlsx, .xls, .csv."
        return None, meta["error"], meta

    # Lấy bytes
    if isinstance(raw_content, (str, Path)):
        p = Path(raw_content)
        if not p.exists():
            meta["error"] = f"File không tồn tại: {filename}"
            return None, meta["error"], meta
        content_bytes = p.read_bytes()
    elif isinstance(raw_content, io.BytesIO):
        content_bytes = raw_content.getvalue()
    elif isinstance(raw_content, bytes):
        content_bytes = raw_content
    else:
        # File-like object từ Streamlit (UploadedFile)
        try:
            content_bytes = raw_content.read()
            if hasattr(raw_content, "seek"):
                raw_content.seek(0)
        except Exception as e:
            meta["error"] = f"Không đọc được luồng file: {e}"
            return None, meta["error"], meta

    if len(content_bytes) == 0:
        meta["error"] = "File rỗng (0 bytes). Hãy chọn file có dữ liệu."
        return None, meta["error"], meta

    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        meta["error"] = f"Dung lượng vượt quá giới hạn 200MB ({len(content_bytes) / 1024 / 1024:.1f}MB)."
        return None, meta["error"], meta

    meta["sha256"] = compute_sha256(content_bytes)

    # Đọc DataFrame
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(io.BytesIO(content_bytes))
        else:
            # Thử utf-8, fallback utf-8-sig / cp1252 / latin1
            try:
                df = pd.read_csv(io.BytesIO(content_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(io.BytesIO(content_bytes), encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(content_bytes), encoding="latin1")
    except Exception as e:
        meta["error"] = f"Không phân tích cú pháp được file (lỗi cú pháp bảng tính): {e}"
        return None, meta["error"], meta

    if df.empty or len(df) == 0:
        meta["error"] = "Bảng tính không có dòng dữ liệu nào."
        return None, meta["error"], meta

    # Chuẩn hóa tên cột
    df.columns = [str(c).strip() for c in df.columns]
    date_map = normalize_date_column_name(list(df.columns))
    if date_map:
        df = df.rename(columns=date_map)

    if DATE_COL not in df.columns:
        meta["error"] = "Thiếu cột 'Ngày'. File cần có cột ngày tháng (ví dụ: 'Ngày', 'Date')."
        return None, meta["error"], meta

    # Chuyển đổi Ngày sang datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        meta["error"] = "Cột 'Ngày' không chứa định dạng ngày tháng hợp lệ nào đọc được."
        return None, meta["error"], meta

    df[DATE_COL] = df[DATE_COL].dt.normalize()

    # Tìm các cột giá mục tiêu
    detected_price_cols = [c for c in TARGET_COLS if c in df.columns]
    if not detected_price_cols:
        # Tìm các cột có tên tương tự
        for c in df.columns:
            if c != DATE_COL and any(target.lower() in c.lower() for target in ["mg95", "mg92", "do", "ron95", "ron92"]):
                detected_price_cols.append(c)

    if not detected_price_cols:
        meta["error"] = (
            "Không tìm thấy cột giá mục tiêu nào (MG95, MG92, DO 0.001%, DO 0.05%). "
            "Hãy đối chiếu file mẫu hướng dẫn."
        )
        return None, meta["error"], meta

    # Kiểm tra kiểu số và giá trị âm / bất hợp lý
    for col in detected_price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Kiểm tra nếu toàn bộ là NaN
        if df[col].isna().all():
            meta["error"] = f"Cột giá '{col}' không có số liệu hợp lệ."
            return None, meta["error"], meta
        # Cảnh báo giá âm
        if (df[col] < 0).any():
            meta["error"] = f"Cột giá '{col}' chứa giá trị âm bất hợp lý."
            return None, meta["error"], meta

    # Nội suy điền khuyết nhẹ cho các cột giá
    for col in detected_price_cols:
        df[col] = df[col].interpolate().bfill().ffill()

    meta["is_valid"] = True
    meta["rows"] = len(df)
    meta["min_date"] = df[DATE_COL].min()
    meta["max_date"] = df[DATE_COL].max()
    meta["price_cols"] = detected_price_cols
    # Chỉ sống trong request hiện tại, dùng để commit đúng dữ liệu đã kiểm tra.
    meta["_validated_df"] = df

    return df, None, meta


def classify_dataframe_records(
    df: pd.DataFrame,
    system_max_date: Optional[pd.Timestamp],
    existing_records: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Phân loại dữ liệu thành: Dữ liệu mới, Dữ liệu đã có, Dữ liệu điều chỉnh giá."""
    if df.empty or DATE_COL not in df.columns:
        return {
            "has_new_data": False,
            "new_rows_count": 0,
            "existing_rows_count": 0,
            "modified_rows_count": 0,
            "new_dates": [],
            "modified_dates": [],
            "modified_details": [],
        }

    dates = df[DATE_COL].drop_duplicates().sort_values()

    if system_max_date is None or pd.isna(system_max_date):
        new_dates = list(dates)
        return {
            "has_new_data": len(new_dates) > 0,
            "new_rows_count": len(df),
            "existing_rows_count": 0,
            "modified_rows_count": 0,
            "new_dates": new_dates,
            "modified_dates": [],
            "modified_details": [],
        }

    new_dates = [d for d in dates if d > system_max_date]
    existing_dates = [d for d in dates if d <= system_max_date]

    modified_dates = []
    modified_details = []
    if existing_records is not None and not existing_records.empty and DATE_COL in existing_records.columns:
        # So sánh giá các ngày cũ
        merged = pd.merge(
            df[df[DATE_COL].isin(existing_dates)],
            existing_records,
            on=DATE_COL,
            suffixes=("_new", "_old"),
            how="inner",
        )
        for col in TARGET_COLS:
            c_new = f"{col}_new"
            c_old = f"{col}_old"
            if c_new in merged.columns and c_old in merged.columns:
                diff = (merged[c_new] - merged[c_old]).abs()
                diff_mask = diff > 0.05  # ngưỡng sai khác giá điều chỉnh
                if diff_mask.any():
                    mod_d = list(merged.loc[diff_mask, DATE_COL].unique())
                    modified_dates.extend(mod_d)
                    for _, row in merged.loc[diff_mask].iterrows():
                        modified_details.append({
                            "date": row[DATE_COL],
                            "column": col,
                            "old_value": float(row[c_old]),
                            "new_value": float(row[c_new]),
                        })
        modified_dates = sorted(list(set(modified_dates)))
        modified_details.sort(key=lambda d: (d["date"], d["column"]))

    return {
        "has_new_data": len(new_dates) > 0,
        "new_rows_count": len(df[df[DATE_COL].isin(new_dates)]),
        "existing_rows_count": len(existing_dates),
        "modified_rows_count": len(modified_dates),
        "new_dates": new_dates,
        "modified_dates": modified_dates,
        "modified_details": modified_details,
        "new_min_date": min(new_dates) if new_dates else None,
        "new_max_date": max(new_dates) if new_dates else None,
    }


def preview_uploaded_files(
    uploaded_files: List[Any],
    system_max_date: Optional[pd.Timestamp] = None,
    existing_records: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Phân tích và trả về thông tin xem trước cho từng file trong danh sách upload."""
    results = []
    for f in uploaded_files:
        name = getattr(f, "name", str(f))
        df, err, meta = parse_and_validate_dataframe(f, name)
        if meta["is_valid"] and df is not None:
            classification = classify_dataframe_records(df, system_max_date, existing_records)
            meta.update(classification)
        results.append(meta)
    return results


def commit_valid_files(
    validated_files_info: List[Dict[str, Any]],
    raw_files_map: Dict[str, Any],
    datasets_dir: Path = DATASETS_DIR,
    overwrite_confirmed_files: Optional[set] = None,
) -> Dict[str, Any]:
    """Lưu các file hợp lệ có ngày mới vào thư mục datasets/ một cách an toàn (atomic).

    Mặc định chỉ lưu file hợp lệ và có ngày mới. Nếu sha256 (nội dung) của file có trong
    `overwrite_confirmed_files`, các ngày cũ có giá điều chỉnh (modified_dates)
    của đúng file đó cũng được ghi kèm — người dùng đã xác nhận ghi đè ở UI.
    Dùng sha256 thay vì tên file để so khớp: nếu 2 file trùng tên nhưng khác nội dung
    được tải cùng lượt, xác nhận đúng 1 file sẽ không vô tình áp dụng luôn cho file kia.
    Ghi nhật ký ingestion vào ingestion_history.json.
    """
    datasets_dir.mkdir(parents=True, exist_ok=True)
    batch_id = f"INGEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    overwrite_confirmed_files = overwrite_confirmed_files or set()

    saved_files = []
    rejected_files = []
    no_new_data_files = []
    total_new_rows = 0
    total_overwritten_rows = 0

    for info in validated_files_info:
        fname = info["filename"]
        if not info["is_valid"]:
            rejected_files.append({
                "filename": fname,
                "reason": info.get("error", "Lỗi định dạng không xác định"),
            })
            continue

        has_new = info.get("has_new_data", False)
        modified_count = info.get("modified_rows_count", 0)
        overwrite_ok = info.get("sha256", fname) in overwrite_confirmed_files and modified_count > 0

        if not has_new and not overwrite_ok:
            no_new_data_files.append({
                "filename": fname,
                "reason": f"Dữ liệu đã tồn tại (dữ liệu đến {info.get('max_date', '')}). Không có ngày mới.",
            })
            continue

        # Có ngày mới hoặc đã xác nhận ghi đè -> Tiến hành lưu
        raw_obj = raw_files_map.get(fname)
        if raw_obj is None:
            rejected_files.append({"filename": fname, "reason": "Mất luồng dữ liệu file"})
            continue

        clean_name = sanitize_filename(fname)
        # UUID loại bỏ va chạm khi hai file cùng tên được xử lý trong cùng một giây.
        stem = Path(clean_name).stem
        target_path = datasets_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

        # Ghi tạm rồi rename (atomic)
        tmp_path = target_path.with_suffix(".csv.tmp")
        try:
            validated_df = info.get("_validated_df")
            if not isinstance(validated_df, pd.DataFrame):
                raise ValueError("Không tìm thấy dữ liệu đã kiểm tra để lưu")
            keep_dates = set(info.get("new_dates", []))
            overwritten_rows = 0
            if overwrite_ok:
                mod_dates = set(info.get("modified_dates", []))
                keep_dates |= mod_dates
                overwritten_rows = len(mod_dates)
            commit_df = validated_df[validated_df[DATE_COL].isin(keep_dates)].copy()
            if commit_df.empty:
                raise ValueError("Không còn bản ghi mới sau khi kiểm tra")
            commit_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            os.replace(tmp_path, target_path)

            saved_files.append({
                "filename": fname,
                "saved_as": target_path.name,
                "sha256": info.get("sha256"),
                "new_rows": info.get("new_rows_count", 0),
                "overwritten_rows": overwritten_rows,
                "date_range": (
                    f"{info.get('min_date').strftime('%d/%m/%Y')} → {info.get('max_date').strftime('%d/%m/%Y')}"
                    if info.get("min_date") and info.get("max_date")
                    else ""
                ),
            })
            total_new_rows += info.get("new_rows_count", 0)
            total_overwritten_rows += overwritten_rows
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            rejected_files.append({"filename": fname, "reason": f"Lỗi ghi file: {e}"})

    # Ghi nhật ký ingestion
    ingestion_entry = {
        "batch_id": batch_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(validated_files_info),
        "saved_count": len(saved_files),
        "rejected_count": len(rejected_files),
        "no_new_data_count": len(no_new_data_files),
        "total_new_rows": total_new_rows,
        "total_overwritten_rows": total_overwritten_rows,
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "no_new_data_files": no_new_data_files,
    }

    history_file = HISTORY_FILE if datasets_dir.resolve() == DATASETS_DIR.resolve() else datasets_dir.parent / "ingestion_history.json"
    _append_ingestion_history(ingestion_entry, history_file)

    return {
        "batch_id": batch_id,
        "success": len(saved_files) > 0,
        "total_new_rows": total_new_rows,
        "total_overwritten_rows": total_overwritten_rows,
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "no_new_data_files": no_new_data_files,
    }


def _append_ingestion_history(entry: Dict[str, Any], history_file: Path = HISTORY_FILE):
    """Ghi bổ sung một bản ghi vào ingestion_history.json an toàn."""
    current = []
    if history_file.exists():
        try:
            current = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            current = []
    current.insert(0, entry)  # Mới nhất ở đầu
    tmp = history_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(current[:100], ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, history_file)


def get_ingestion_history() -> List[Dict[str, Any]]:
    """Lấy danh sách lịch sử nạp dữ liệu."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
