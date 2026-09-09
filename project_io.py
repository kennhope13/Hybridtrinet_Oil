"""Restricted checkpoint loading and atomic application data storage."""

import io
import hashlib
import json
import os
import time
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def dataset_fingerprint(directory, horizons):
    digest = hashlib.sha256()
    files = sorted(p for p in Path(directory).glob('*')
                   if p.suffix.lower() in {'.csv', '.xls', '.xlsx'}
                   and not p.name.startswith('~$'))
    for path in files:
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\0')
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        digest.update(b'\0')
    digest.update(str(tuple(horizons)).encode('ascii'))
    return digest.hexdigest()


def process_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
    except (TypeError, ValueError):
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            return ctypes.get_last_error() != 87  # Invalid PID; access denied stays busy.
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def load_checkpoint(path, map_location="cpu"):
    # Legacy checkpoints contain StandardScaler and NumPy arrays as metadata.
    numpy_core = np._core if hasattr(np, "_core") else np.core
    allowed = [StandardScaler, MinMaxScaler, np.ndarray, np.dtype,
               numpy_core.multiarray._reconstruct, numpy_core.multiarray.scalar]
    allowed.extend(type(np.dtype(kind)) for kind in ("float32", "float64", "int32", "int64"))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=map_location, weights_only=True)


def save_upload(directory, filename, content):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename.replace("\\", "/")).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Unsupported dataset file type")
    # The supplied name never becomes a filesystem path; existing files stay intact.
    with tempfile.NamedTemporaryFile(dir=directory, prefix="upload_", suffix=suffix,
                                     delete=False) as handle:
        path = Path(handle.name)
        try:
            handle.write(content)
        except Exception:
            handle.close()
            path.unlink(missing_ok=True)
            raise
    return path


def _replace_with_retry(tmp, dest, attempts=6, delay=0.05):
    """os.replace() trên Windows có thể ném PermissionError [WinError 32] nếu file đích đang bị
    tiến trình khác mở đúng lúc đó. simulation_cache.json bị ghi bởi 2 hệ thống nền độc lập
    (run_backtest_job.py và pipeline_engine.py) nên nguy cơ đụng độ thoáng qua là có thật — thử
    lại vài lần thay vì để lỗi văng ra ngoài."""
    for i in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def write_cache(path, fingerprint, frame):
    path = Path(path)
    payload = {"fp": fingerprint, "df": json.loads(frame.to_json(orient="table", date_format="iso"))}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        _replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_cache(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["fp"], pd.read_json(io.StringIO(json.dumps(payload["df"])), orient="table")
