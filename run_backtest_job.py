"""run_backtest_job.py — chạy đối chiếu (backtest) NGẦM, TÁCH HẲN khỏi tiến trình Streamlit.

Vì sao tách thành tiến trình riêng thay vì Thread trong session Streamlit: 1 Thread gắn với 1
session cụ thể sẽ bị dọn/mất khi session đó rerun hoặc người dùng đóng tab — không đảm bảo
"tồn tại qua rerun/chuyển trang". Một tiến trình hệ điều hành độc lập (subprocess) thì không,
nó cứ chạy tới khi xong bất kể app_main.py đang làm gì.

Trạng thái được ghi vào 2 file trên đĩa (không dùng session_state làm nguồn thật):
  - .backtest.lock       : ai đang giữ quyền chạy (job_id, pid, fingerprint) — khóa nguyên tử,
                            cùng kiểu mode='x' + tiếp quản theo job_id như .training.lock.
  - .backtest_job.json   : job_id, fingerprint, status (pending/running/success/failed),
                            started_at, finished_at, error — để app_main.py đọc mà không cần
                            hỏi qua session_state (đọc được từ BẤT KỲ session/trang nào).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cùng lỗi đã gặp ở train_all_horizons.py: stdout mặc định trên Windows dùng bảng mã cp1252,
# ném UnicodeEncodeError ngay dòng print() đầu tiên có emoji/tiếng Việt dấu — làm job chết ngay
# lập tức (dù bị redirect ra DEVNULL, Python vẫn encode trước khi ghi nên vẫn lỗi như thường).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import backtest_worker as bw  # noqa: E402
from project_io import write_cache, process_alive  # noqa: E402

LOCK_FILE = ROOT / ".backtest.lock"
STATUS_FILE = ROOT / ".backtest_job.json"
CACHE_FILE = ROOT / "simulation_cache.json"


def _pid_alive(pid):
    try:
        return process_alive(pid)
    except PermissionError:
        return True
    except OSError:
        return False
    except Exception:
        return False


def _replace_with_retry(tmp, dest, attempts=6, delay=0.05):
    """os.replace() trên Windows có thể ném PermissionError [WinError 32] nếu file đích đang bị
    tiến trình khác (VD: app_main.py đang đọc/ghi cùng lúc) mở đúng lúc đó — tranh chấp thoáng
    qua, không phải lỗi thật. Thử lại vài lần cách nhau vài chục mili-giây để tự vượt qua."""
    for i in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def _atomic_write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _replace_with_retry(tmp, path)


def write_status(**fields):
    current = {}
    if STATUS_FILE.exists():
        try:
            current = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(fields)
    _atomic_write_json(STATUS_FILE, current)


def acquire_or_takeover_lock(job_id, fingerprint):
    """Y hệt cơ chế khóa huấn luyện (job_id + mode='x' nguyên tử + tiếp quản đúng job_id) —
    chỉ khác đối tượng khóa là backtest thay vì training, và file khóa riêng biệt."""
    my_pid = os.getpid()
    info = None
    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        except Exception:
            info = None

    if info:
        existing_pid = info.get("pid")
        existing_job_id = info.get("job_id")
        if job_id and existing_job_id == job_id:
            info["pid"] = my_pid
            info["fingerprint"] = fingerprint
            tmp = LOCK_FILE.with_suffix(".lock.tmp")
            tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
            _replace_with_retry(tmp, LOCK_FILE)
            return job_id
        if existing_pid and existing_pid != my_pid and _pid_alive(existing_pid):
            print(f"❌ Đã có 1 backtest job khác đang chạy (PID {existing_pid}, job_id {existing_job_id}). Dừng ngay.", flush=True)
            sys.exit(1)
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    actual_job_id = job_id
    lock_content = json.dumps({"job_id": actual_job_id, "pid": my_pid, "fingerprint": fingerprint}, ensure_ascii=False)
    try:
        with open(LOCK_FILE, mode="x", encoding="utf-8") as f:
            f.write(lock_content)
    except FileExistsError:
        info = None
        try:
            info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        if info and info.get("job_id") == actual_job_id:
            tmp = LOCK_FILE.with_suffix(".lock.tmp")
            info["pid"] = my_pid
            tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
            _replace_with_retry(tmp, LOCK_FILE)
        elif info and info.get("pid") and _pid_alive(info.get("pid")):
            print(f"❌ Xung đột khóa: job khác ({info.get('job_id')}) vừa chiếm quyền chạy.", flush=True)
            sys.exit(1)
        else:
            tmp = LOCK_FILE.with_suffix(".lock.tmp")
            tmp.write_text(lock_content, encoding="utf-8")
            _replace_with_retry(tmp, LOCK_FILE)
    return actual_job_id


def release_own_lock(job_id):
    try:
        if LOCK_FILE.exists():
            info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            if info.get("job_id") == job_id:
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--cutoff-date", required=True, help="ISO date string, vd 2026-01-01")
    p.add_argument("--models", nargs="+", default=["GUMNet"])
    p.add_argument("--files", nargs="*", default=[], help="Đường dẫn các file dữ liệu cần đối chiếu")
    return p.parse_args()


if __name__ == "__main__":
    import pandas as pd

    args = parse_args()
    job_id = acquire_or_takeover_lock(args.job_id, args.fingerprint)
    import atexit
    atexit.register(release_own_lock, job_id)

    write_status(job_id=job_id, fingerprint=args.fingerprint, status="running",
                 started_at=pd.Timestamp.now().isoformat(), finished_at=None, error=None)
    print(f"🚀 Backtest job {job_id} bắt đầu (fingerprint={args.fingerprint})", flush=True)

    try:
        cutoff = pd.Timestamp(args.cutoff_date)
        file_paths = [Path(p) for p in args.files]
        result = bw.run_upload_simulation(
            str(bw.BUILTIN_CSV), file_paths, cutoff,
            sel_horizons=bw.HORIZONS, sel_models=args.models,
            log_fn=lambda m: print(m, flush=True),
        )
        write_cache(CACHE_FILE, args.fingerprint, result)
        write_status(status="success", finished_at=pd.Timestamp.now().isoformat(), error=None)
        print(f"✅ Backtest job {job_id} hoàn tất, {len(result)} dòng kết quả.", flush=True)
    except Exception as e:
        write_status(status="failed", finished_at=pd.Timestamp.now().isoformat(), error=str(e))
        print(f"❌ Backtest job {job_id} thất bại: {e}", flush=True)
        sys.exit(1)
