"""Lightweight watchdog for the PyTorch pipeline worker.

This process intentionally imports only the standard library.  It survives a
native crash in PyTorch/NumPy and turns the worker exit code into a durable UI
status instead of leaving Streamlit with only a stale PID.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATUS_FILE = ROOT / ".pipeline_status.json"
LOCK_FILE = ROOT / ".pipeline.lock"


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_status(data, attempts=10, delay=0.05):
    """Ghi trạng thái có thử lại chống WinError 32.

    Đây là lần ghi QUAN TRỌNG NHẤT của cả hệ thống: nó là thứ duy nhất báo được
    "worker đã chết" khi worker chết kiểu native (không kịp tự báo). Nó lại ghi đúng
    lúc app_main.py đang đọc file này mỗi 2 giây trong màn hình khóa — nếu để
    os.replace() trần và trúng xung đột, chính watchdog cũng chết theo và trạng thái
    kẹt "đang chạy" vĩnh viễn. Hết số lần thử thì ghi thẳng đè lên, thà mất tính
    nguyên tử còn hơn mất hẳn trạng thái.
    """
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    for i in range(attempts):
        try:
            os.replace(tmp, STATUS_FILE)
            return
        except PermissionError:
            if i == attempts - 1:
                STATUS_FILE.write_text(payload, encoding="utf-8")
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return
            time.sleep(delay)


def _release_lock(pipeline_id):
    lock = _read_json(LOCK_FILE)
    if lock.get("pipeline_id") == pipeline_id:
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--new-rows", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    # Give the launcher time to persist the initial state before the worker can
    # update it. This also removes the Popen/status race on fast machines.
    for _ in range(50):
        state = _read_json(STATUS_FILE)
        if state.get("pipeline_id") == args.pipeline_id:
            break
        time.sleep(0.1)

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "pipeline_engine.py"),
        "--pipeline-id",
        args.pipeline_id,
        "--batch-id",
        args.batch_id,
        "--new-rows",
        str(args.new_rows),
    ]
    if args.force_retrain:
        cmd.append("--force-retrain")

    worker = subprocess.Popen(cmd, cwd=str(ROOT), env=os.environ.copy())
    return_code = worker.wait()

    state = _read_json(STATUS_FILE)
    if state.get("pipeline_id") == args.pipeline_id and state.get("is_running"):
        state["status"] = "failed"
        state["is_running"] = False
        state["step_title"] = "Tiến trình tính toán đã dừng"
        state["error"] = f"Worker tính toán kết thúc ngoài dự kiến (mã thoát {return_code})."
        state["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for step in state.get("steps", []):
            if step.get("state") == "running":
                step["state"] = "failed"
        _write_status(state)
        print(
            f"[{state['updated_at']}] [SUPERVISOR] Worker dừng ngoài dự kiến, exit_code={return_code}",
            flush=True,
        )

    _release_lock(args.pipeline_id)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
