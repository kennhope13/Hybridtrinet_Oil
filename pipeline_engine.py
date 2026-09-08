"""pipeline_engine.py — Bộ điều phối tự động hóa ngầm (Pipeline Orchestrator) và State Machine.

Thực hiện đúng 100% đặc tả tại KE_HOACH_FRONTEND_VA_LUONG_TU_DONG.md:
1. Quản lý trạng thái bền vững qua .pipeline_status.json (không mất khi rerun hoặc đổi tab).
2. Thanh trạng thái toàn cục 5 bước:
   [✓] File hợp lệ
   [✓] Đã cập nhật dữ liệu
   [●] Đang đối chiếu độ chính xác
   [○] Tối ưu GUMNet nếu cần
   [○] Hoàn tất
3. Chạy Backtest ngầm độc lập qua subprocess.
4. Tự động kiểm tra ngưỡng quyết định huấn luyện:
   - new_rows >= MIN_NEW_ROWS (5)
   - eval_pairs >= MIN_EVAL_PAIRS (10)
   - MAPE > MAPE_THRESHOLD (10.0%)
5. Huấn luyện GUMNet candidate cô lập, đối đầu với Production:
   - Tốt hơn -> Atomic Promote sang checkpoints_multi/
   - Không tốt hơn / Lỗi -> Giữ nguyên Production hiện tại
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and getattr(sys.stderr, "encoding", None) and sys.stderr.encoding.lower() != "utf-8":
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
STATUS_FILE = ROOT / ".pipeline_status.json"
LOCK_FILE = ROOT / ".pipeline.lock"
CKPT_DIR = ROOT / "checkpoints_multi"
BACKUP_DIR = ROOT / "checkpoints_backup"
CANDIDATE_DIR = ROOT / "checkpoints_candidate"

# Cấu hình nghiệp vụ tự động hóa
MIN_NEW_ROWS = 5
MIN_EVAL_PAIRS = 10
MAPE_RETRAIN_THRESHOLD = 10.0  # %
HORIZONS = [1, 5, 10, 15, 20, 30, 60]


def _summarize_backtest(frame: Any) -> Dict[str, Any]:
    """Return metrics from the DataFrame contract of backtest_worker."""
    if frame is None or not hasattr(frame, "columns"):
        raise ValueError("Backtest không trả về bảng kết quả")
    if frame.empty:
        return {"mape": 0.0, "mae": 0.0, "total_pts": 0}
    required = {"% Lệch", "Sai lệch"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Kết quả backtest thiếu cột {sorted(missing)}")
    mape = float(frame["% Lệch"].dropna().mean())
    mae = float(frame["Sai lệch"].dropna().mean())
    if not (mape >= 0 and mape < float("inf") and mae >= 0 and mae < float("inf")):
        raise ValueError("Chỉ số backtest không hợp lệ")
    return {"mape": mape, "mae": mae, "total_pts": int(len(frame))}


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        p = int(pid)
        if p <= 0:
            return False
        os.kill(p, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False
    except Exception:
        return False


def _replace_with_retry(tmp: Path, target: Path, attempts: int = 10, delay: float = 0.05):
    """os.replace() dùng chung, chống xung đột WinError 32 trên Windows (file đích đang bị
    tiến trình khác — thường là app_main.py đọc liên tục — mở đúng lúc ghi đè)."""
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def _atomic_write_json(path: Path, data: Dict[str, Any], attempts: int = 10, delay: float = 0.05):
    """Ghi JSON nguyên tử có xử lý thử lại (retry) chống xung đột WinError 32 trên Windows."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try:
        _replace_with_retry(tmp, path, attempts=attempts, delay=delay)
    except PermissionError:
        # Sau khi hết số lần thử lại vẫn bị chặn: ghi trực tiếp đè lên file đích thay vì
        # bỏ cuộc, để trạng thái không bị mất hẳn.
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.unlink(missing_ok=True)


def _acquire_pipeline_lock(pipeline_id: str) -> bool:
    """Chiếm khóa pipeline nguyên tử (O_EXCL), tự động giải phóng lock rác nếu quá hạn hoặc PID chết."""
    payload = json.dumps({"pipeline_id": pipeline_id, "pid": os.getpid(), "created_at": time.time()})
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return True
    except FileExistsError:
        try:
            current = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            lock_pid = int(current.get("pid", 0))
            lock_pipe_id = current.get("pipeline_id")
            lock_age = time.time() - float(current.get("created_at", 0))

            # 1. Cùng pipeline_id (do app đặt giữ chỗ trước khi spawn worker): tiếp quản lock
            if lock_pipe_id == pipeline_id:
                _atomic_write_json(LOCK_FILE, {
                    "pipeline_id": pipeline_id,
                    "pid": os.getpid(),
                    "created_at": current.get("created_at", time.time()),
                })
                return True

            # 2. Lock cũ của PID đã chết HOẶC quá 10 phút (stale lock timeout): giải phóng và lấy lại
            if not _pid_alive(lock_pid) or lock_age > 600:
                LOCK_FILE.unlink(missing_ok=True)
                return _acquire_pipeline_lock(pipeline_id)

            # 3. PID khác đang thực sự chạy: từ chối
            return False
        except Exception:
            LOCK_FILE.unlink(missing_ok=True)
            return _acquire_pipeline_lock(pipeline_id)


def _release_pipeline_lock(pipeline_id: str) -> None:
    try:
        if LOCK_FILE.exists():
            current = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            if current.get("pipeline_id") == pipeline_id or not _pid_alive(int(current.get("pid", 0))):
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _validate_candidate(candidate_dir: Path, job_id: str) -> Dict[int, float]:
    losses: Dict[int, float] = {}
    for horizon in HORIZONS:
        path = candidate_dir / f"gumnet_h{horizon}.pt"
        if not path.exists():
            raise ValueError(f"Thiếu checkpoint mốc {horizon} ngày")
        checkpoint = load_checkpoint(path, map_location="cpu")
        required = {"model_state_dict", "horizon", "feature_cols", "target_cols", "job_id", "best_val_loss"}
        missing = required - set(checkpoint) if isinstance(checkpoint, dict) else required
        if missing:
            raise ValueError(f"Checkpoint h{horizon} thiếu {sorted(missing)}")
        if checkpoint["horizon"] != horizon or checkpoint["job_id"] != job_id:
            raise ValueError(f"Checkpoint h{horizon} sai horizon/job_id")
        if not isinstance(checkpoint["model_state_dict"], dict) or not checkpoint["model_state_dict"]:
            raise ValueError(f"Checkpoint h{horizon} có model_state_dict rỗng")
        loss = float(checkpoint["best_val_loss"])
        if not (loss >= 0 and loss < float("inf")):
            raise ValueError(f"Checkpoint h{horizon} có validation loss không hợp lệ")
        losses[horizon] = loss
    return losses


def _production_losses() -> Optional[Dict[int, float]]:
    losses: Dict[int, float] = {}
    try:
        for horizon in HORIZONS:
            checkpoint = load_checkpoint(CKPT_DIR / f"gumnet_h{horizon}.pt", map_location="cpu")
            loss = float(checkpoint["best_val_loss"])
            if not (loss >= 0 and loss < float("inf")):
                return None
            losses[horizon] = loss
    except Exception:
        # Checkpoint cũ có thể chưa chứa best_val_loss. Dùng phiên training
        # thành công gần nhất làm baseline chuyển tiếp; thiếu đủ 7 mốc thì fail closed.
        try:
            history = json.loads((CKPT_DIR / "training_history.json").read_text(encoding="utf-8"))
            for entry in history:
                if entry.get("status") != "success":
                    continue
                results = entry.get("results", {})
                fallback = {}
                for horizon in HORIZONS:
                    value = results.get(f"GUMNet_h{horizon}", results.get(f"h{horizon}"))
                    if value is None:
                        break
                    fallback[horizon] = float(value)
                if len(fallback) == len(HORIZONS):
                    return fallback
        except Exception:
            pass
        return None
    return losses


def _candidate_is_better(candidate: Dict[int, float], production: Optional[Dict[int, float]]) -> bool:
    # Fail closed: không có baseline đáng tin cậy thì không tự thay production.
    if not production or set(candidate) != set(production):
        return False
    candidate_avg = sum(candidate.values()) / len(candidate)
    production_avg = sum(production.values()) / len(production)
    return candidate_avg < production_avg


def _promote_with_rollback(candidate_dir: Path, backup_dir: Path) -> None:
    """Replace all horizons transactionally; restore every old file on failure."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        source = CKPT_DIR / f"gumnet_h{horizon}.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, backup_dir / source.name)
    try:
        for horizon in HORIZONS:
            source = candidate_dir / f"gumnet_h{horizon}.pt"
            target = CKPT_DIR / source.name
            tmp = target.with_suffix(".pt.promote.tmp")
            shutil.copy2(source, tmp)
            _replace_with_retry(tmp, target)
    except Exception:
        for horizon in HORIZONS:
            backup = backup_dir / f"gumnet_h{horizon}.pt"
            if backup.exists():
                tmp = CKPT_DIR / f"gumnet_h{horizon}.pt.rollback.tmp"
                shutil.copy2(backup, tmp)
                _replace_with_retry(tmp, CKPT_DIR / f"gumnet_h{horizon}.pt")
        raise


def get_pipeline_status() -> Dict[str, Any]:
    """Đọc trạng thái hiện tại của Pipeline từ đĩa."""
    default_state = {
        "pipeline_id": None,
        "status": "idle",  # idle, running, complete, failed
        "step_index": 0,
        "step_title": "Sẵn sàng",
        "details": "Hệ thống đang hoạt động với mô hình GUMNet hiện tại.",
        "steps": [
            {"title": "File hợp lệ", "state": "waiting"},
            {"title": "Đã cập nhật dữ liệu", "state": "waiting"},
            {"title": "Đối chiếu độ chính xác", "state": "waiting"},
            {"title": "Tối ưu GUMNet nếu cần", "state": "waiting"},
            {"title": "Hoàn tất", "state": "waiting"},
        ],
        "is_running": False,
        "started_at": None,
        "updated_at": None,
        "batch_info": None,
        "backtest_result": None,
        "retrain_decision": None,
        "candidate_result": None,
        "error": None,
        "pid": None,
    }

    if not STATUS_FILE.exists():
        return default_state

    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        # Kiểm tra nếu ghi nhận is_running nhưng tiến trình đã chết. Trước đây chỗ này chỉ sửa
        # tạm trong bộ nhớ để trả về, KHÔNG ghi lại xuống file — nên banner cứ đọc lại đúng file
        # cũ ("running") ở mọi lượt sau, hiển thị "đang chạy" mãi dù tiến trình đã chết từ lâu.
        # Giờ ghi thẳng trạng thái thất bại (kèm sửa lại step_title/steps cho khớp) xuống đĩa
        # ngay khi phát hiện, để báo đúng và dứt khoát thay vì "treo" vô thời hạn.
        pid = data.get("pid")
        if data.get("is_running") and pid is not None:
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                pid_int = 0

            # Tính thời gian chạy để áp dụng grace period chống false-alarm lúc mới spawn
            started_at_str = data.get("started_at")
            age_sec = 999.0
            if started_at_str:
                try:
                    t_start = datetime.datetime.strptime(started_at_str, "%Y-%m-%d %H:%M:%S")
                    age_sec = (datetime.datetime.now() - t_start).total_seconds()
                except Exception:
                    pass

            # 1. Quá hạn tối đa 15 phút (900s) không xong: Tự phục hồi an toàn (self-healing)
            # 2. Hoặc đã khởi động qua grace period (>10s) và tiến trình PID đã thực sự biến mất
            is_stale_timeout = age_sec > 900.0
            is_pid_dead = (pid_int > 0 and age_sec > 10.0 and not _pid_alive(pid_int))

            if is_stale_timeout or is_pid_dead:
                data["is_running"] = False
                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["error"] = "Tiến trình chạy quá thời gian tối đa." if is_stale_timeout else "Tiến trình nền bị gián đoạn ngoài ý muốn (tiến trình dừng)."
                    data["step_title"] = "Hết thời gian chờ" if is_stale_timeout else "Đã dừng do gián đoạn"
                    for s in data.get("steps", []):
                        if s.get("state") == "running":
                            s["state"] = "failed"
                    try:
                        _atomic_write_json(STATUS_FILE, data)
                    except Exception:
                        pass
        return data
    except Exception:
        return default_state


def reset_pipeline_status() -> Dict[str, Any]:
    """Đặt lại trạng thái Pipeline về mặc định (idle) và dọn khóa an toàn."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    default_state = {
        "pipeline_id": None,
        "status": "idle",
        "step_index": 0,
        "step_title": "Sẵn sàng",
        "details": "Hệ thống sẵn sàng tiếp nhận dữ liệu mới.",
        "steps": [
            {"title": "File hợp lệ", "state": "waiting"},
            {"title": "Đã cập nhật dữ liệu", "state": "waiting"},
            {"title": "Đối chiếu độ chính xác", "state": "waiting"},
            {"title": "Tối ưu GUMNet nếu cần", "state": "waiting"},
            {"title": "Hoàn tất", "state": "waiting"},
        ],
        "is_running": False,
        "started_at": None,
        "updated_at": None,
        "batch_info": None,
        "backtest_result": None,
        "retrain_decision": None,
        "candidate_result": None,
        "error": None,
        "pid": None,
    }
    _atomic_write_json(STATUS_FILE, default_state)
    return default_state


def update_pipeline_status(**fields) -> Dict[str, Any]:
    """Cập nhật trạng thái Pipeline một cách nguyên tử."""
    current = get_pipeline_status()
    current.update(fields)
    current["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(STATUS_FILE, current)
    return current


def set_pipeline_step(step_idx: int, step_title: str, details: str = "", step_state: str = "running"):
    """Cập nhật bước hiện tại trong 5 bước của banner."""
    current = get_pipeline_status()
    steps = current.get("steps", [])
    for i, s in enumerate(steps):
        if i < step_idx:
            s["state"] = "done"
        elif i == step_idx:
            s["state"] = step_state
        else:
            s["state"] = "waiting"

    current["step_index"] = step_idx
    current["step_title"] = step_title
    current["details"] = details
    current["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(STATUS_FILE, current)


def launch_pipeline_background(batch_id: str, new_rows: int, file_names: List[str], force_retrain: bool = False):
    """Khởi động bộ điều phối ngầm bằng một tiến trình độc lập hoàn toàn với Streamlit."""
    pipeline_id = f"PIPE-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    current = get_pipeline_status()
    if current.get("is_running") and _pid_alive(int(current.get("pid") or 0)):
        return {"started": False, "reason": "pipeline_running", "pipeline_id": current.get("pipeline_id")}
    if not _acquire_pipeline_lock(pipeline_id):
        return {"started": False, "reason": "pipeline_locked", "pipeline_id": None}

    initial_steps = [
        {"title": "File hợp lệ", "state": "done"},
        {"title": "Đã cập nhật dữ liệu", "state": "done"},
        {"title": "Đối chiếu độ chính xác", "state": "running"},
        {"title": "Tối ưu GUMNet nếu cần", "state": "waiting"},
        {"title": "Hoàn tất", "state": "waiting"},
    ]

    action_label = "Đang buộc huấn luyện lại..." if force_retrain else "Đang chạy đối chiếu dự báo với giá thực tế..."

    # Chạy tiến trình nền độc lập
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-u",
        str(ROOT / "pipeline_engine.py"),
        "--pipeline-id",
        pipeline_id,
        "--batch-id",
        batch_id,
        "--new-rows",
        str(new_rows),
    ]
    if force_retrain:
        cmd.append("--force-retrain")

    env = dict(os.environ, PYTHONIOENCODING="utf-8", FOR_DISABLE_CONSOLE_CTRL_HANDLER="1")
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    try:
        log_file = ROOT / "pipeline_worker.log"
        log_handle = open(log_file, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            env=env,
        )
        # Ghi trạng thái khởi đầu sạch hoàn toàn (tránh rò rỉ bất kỳ giá trị cũ nào từ đợt trước)
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_state = {
            "pipeline_id": pipeline_id,
            "status": "running",
            "step_index": 2,
            "step_title": "Đang đối chiếu độ chính xác",
            "details": f"Đã ghi nhận {new_rows} ngày mới từ {len(file_names)} file. {action_label}",
            "steps": initial_steps,
            "is_running": True,
            "started_at": now_ts,
            "updated_at": now_ts,
            "batch_info": {"batch_id": batch_id, "new_rows": new_rows, "files": file_names, "force_retrain": force_retrain},
            "backtest_result": None,
            "retrain_decision": None,
            "candidate_result": None,
            "error": None,
            "pid": proc.pid,
        }
        _atomic_write_json(STATUS_FILE, new_state)
        return {"started": True, "pipeline_id": pipeline_id, "pid": proc.pid}
    except Exception as e:
        update_pipeline_status(
            status="failed",
            is_running=False,
            error=f"Không thể khởi động tiến trình nền: {e}",
            pid=None,
        )
        _release_pipeline_lock(pipeline_id)
        return {"started": False, "reason": "start_failed", "pipeline_id": pipeline_id}


def run_pipeline_task(pipeline_id: str, batch_id: str, new_rows: int, force_retrain: bool = False):
    """Tiến trình thực thi chính của bộ điều phối ngầm (chạy trong tiến trình riêng)."""
    my_pid = os.getpid()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] [INFO] Khởi động worker cho pipeline_id: {pipeline_id} (PID: {my_pid}), batch: {batch_id}, new_rows: {new_rows}, force_retrain: {force_retrain}", flush=True)

    lock_owned = _acquire_pipeline_lock(pipeline_id)
    if not lock_owned:
        err_msg = f"Worker không thể lấy khóa pipeline (đang bị chiếm bởi tác vụ khác). Dừng worker an toàn."
        print(f"[{now_str}] [ERROR] {err_msg}", flush=True)
        update_pipeline_status(status="failed", is_running=False, error=err_msg, step_title="Kẹt khóa pipeline")
        return

    update_pipeline_status(pid=my_pid, is_running=True, status="running", error=None)

    try:
        # 1. BƯỚC 3: Chạy Backtest đối chiếu thực tế với dự báo cũ
        set_pipeline_step(
            2,
            "Đang đối chiếu độ chính xác",
            "Đang tính toán sai số MAE/MAPE trên các dự báo đã đến hạn...",
            "running",
        )

        backtest_result = None
        try:
            import backtest_worker as bw
            from project_io import write_cache

            # Quét lại file và tính toán
            data_dir = ROOT / "datasets"
            files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"] and not f.name.startswith("~$")]
            hz_str = "_".join(str(x) for x in HORIZONS)
            fingerprint = f"{len(files)}_{max(f.stat().st_mtime for f in files) if files else 0}_{hz_str}"

            # Xác định cửa sổ đánh giá 365 ngày theo ngày dữ liệu mới nhất.
            base_df = bw.load_df(bw.BUILTIN_CSV)
            extra_dfs = [bw.load_df(f) for f in files]
            date_candidates = []
            for frame in [base_df] + extra_dfs:
                if frame is not None and not frame.empty and "Ngày" in frame.columns:
                    date_candidates.append(frame["Ngày"].max())
            latest_date = max(date_candidates) if date_candidates else datetime.datetime.now()
            start_date = latest_date - datetime.timedelta(days=365)

            def _worker_log(msg: str):
                now_t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now_t}] {msg}", flush=True)

            # backtest_worker nhận đường dẫn và trả về đúng một DataFrame.
            comb = bw.run_upload_simulation(
                bw.BUILTIN_CSV,
                files,
                start_date,
                sel_horizons=HORIZONS,
                sel_models=["GUMNet"],
                log_fn=_worker_log,
            )
            stats = _summarize_backtest(comb)

            # Lưu cache đối chiếu nguyên tử
            cache_path = ROOT / "simulation_cache.json"
            write_cache(cache_path, fingerprint, comb)

            mape_val = float(stats.get("mape", 0.0))
            mae_val = float(stats.get("mae", 0.0))
            pts_count = int(stats.get("total_pts", len(comb) if comb is not None else 0))

            backtest_result = {
                "fingerprint": fingerprint,
                "mape": round(mape_val, 2),
                "mae": round(mae_val, 2),
                "sample_count": pts_count,
                "calculated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            update_pipeline_status(backtest_result=backtest_result)
        except Exception as e:
            import traceback
            tb_err = traceback.format_exc()
            print(f"[{datetime.datetime.now()}] [ERROR] Lỗi đối chiếu backtest: {tb_err}", flush=True)
            update_pipeline_status(
                backtest_result={"error": f"Lỗi đối chiếu: {e}"},
                error=f"Đối chiếu thực tế gặp sự cố: {e}. Dự báo hiện tại vẫn tiếp tục hoạt động.",
            )
            set_pipeline_step(2, "Đối chiếu chưa hoàn tất", "Kết quả cũ vẫn được giữ nguyên. Có thể thử lại sau.", "failed")
            update_pipeline_status(status="failed", is_running=False)
            return

        # 2. BƯỚC 4: Đánh giá nhu cầu huấn luyện lại
        mape = backtest_result.get("mape", 0.0) if backtest_result else 0.0
        samples = backtest_result.get("sample_count", 0) if backtest_result else 0

        retrain_needed = False
        decision_reason = ""

        if force_retrain:
            retrain_needed = True
            decision_reason = f"Kích hoạt tối ưu GUMNet thủ công theo yêu cầu (MAPE hiện tại: {mape:.2f}%)."
        elif new_rows < MIN_NEW_ROWS:
            decision_reason = f"Dữ liệu mới ({new_rows} ngày) chưa đủ ngưỡng tối thiểu ({MIN_NEW_ROWS} ngày) để kích hoạt tối ưu."
        elif samples < MIN_EVAL_PAIRS:
            decision_reason = f"Chưa đủ số cặp đối chiếu dự báo ({samples}/{MIN_EVAL_PAIRS} cặp) để đánh giá độ tin cậy."
        elif mape <= MAPE_RETRAIN_THRESHOLD:
            decision_reason = f"Độ chính xác GUMNet hiện tại rất tốt (MAPE {mape:.2f}% ≤ ngưỡng {MAPE_RETRAIN_THRESHOLD}%). Giữ nguyên mô hình chuẩn."
        else:
            retrain_needed = True
            decision_reason = f"MAPE đạt {mape:.2f}% (vượt ngưỡng {MAPE_RETRAIN_THRESHOLD}%) và đủ {new_rows} ngày mới -> Tự động tối ưu GUMNet candidate."

        update_pipeline_status(retrain_decision={"needed": retrain_needed, "reason": decision_reason})

        if not retrain_needed:
            # Không cần huấn luyện -> Hoàn tất ngay
            set_pipeline_step(
                3,
                "Không cần tối ưu GUMNet",
                decision_reason,
                "done",
            )
            time.sleep(1)
            set_pipeline_step(
                4,
                "Hoàn tất",
                f"Dữ liệu đã cập nhật ({samples} điểm đối chiếu). GUMNet hiện tại đạt MAPE {mape:.2f}% (dưới ngưỡng 10%) nên tiếp tục được giữ làm mô hình chuẩn.",
                "done",
            )
            update_pipeline_status(status="complete", is_running=False)
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Hoàn tất pipeline an toàn (không cần huấn luyện).", flush=True)
            return

        # 3. NẾU CẦN HUẤN LUYỆN: Huấn luyện GUMNet Candidate ngầm
        set_pipeline_step(
            3,
            "Đang tối ưu GUMNet Candidate",
            f"MAPE hiện tại ({mape:.2f}%). Đang huấn luyện candidate ngầm...",
            "running",
        )

        candidate_job_id = f"CAND-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        candidate_out_dir = CANDIDATE_DIR / candidate_job_id
        candidate_out_dir.mkdir(parents=True, exist_ok=True)

        production_losses = _production_losses()

        backup_job_dir = BACKUP_DIR / candidate_job_id
        backup_job_dir.mkdir(parents=True, exist_ok=True)
        try:
            for horizon in HORIZONS:
                source = CKPT_DIR / f"gumnet_h{horizon}.pt"
                if not source.exists():
                    raise FileNotFoundError(source)
                shutil.copy2(source, backup_job_dir / source.name)
                shutil.copy2(source, candidate_out_dir / source.name)
        except Exception as exc:
            update_pipeline_status(status="failed", is_running=False, error=f"Không thể tạo bản sao lưu: {exc}")
            set_pipeline_step(3, "Không thể tạo bản sao lưu", "GUMNet hiện tại không bị thay đổi.", "failed")
            return

        cmd_train = [
            sys.executable,
            "-u",
            str(ROOT / "train_all_horizons.py"),
            "--job-id",
            candidate_job_id,
            "--models",
            "GUMNet",
            "--horizons",
            "1", "5", "10", "15", "20", "30", "60",
            "--output_dir",
            str(candidate_out_dir),
        ]

        train_success = False
        try:
            proc_train = subprocess.run(
                cmd_train,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            train_success = (proc_train.returncode == 0)
        except Exception as e:
            print(f"[{datetime.datetime.now()}] [ERROR] Huấn luyện candidate lỗi: {e}", flush=True)
            train_success = False

        candidate_promoted = False
        candidate_msg = ""

        if train_success:
            try:
                candidate_losses = _validate_candidate(candidate_out_dir, candidate_job_id)
                if _candidate_is_better(candidate_losses, production_losses):
                    _promote_with_rollback(candidate_out_dir, backup_job_dir)
                    candidate_promoted = True
                    candidate_msg = "GUMNet Candidate đã được xác thực, tốt hơn Production và đã được áp dụng."
                elif production_losses is None:
                    candidate_msg = "Không có chỉ số baseline Production đáng tin cậy; giữ nguyên model hiện tại để an toàn."
                else:
                    candidate_msg = "Candidate không tốt hơn GUMNet Production trên cùng tiêu chí; giữ nguyên model hiện tại."
            except Exception as exc:
                candidate_msg = f"Candidate không vượt qua kiểm tra ({exc}). Giữ nguyên Production hiện tại."
        else:
            candidate_msg = "Tối ưu candidate không thành công. GUMNet Production hiện tại tiếp tục hoạt động an toàn."

        update_pipeline_status(
            candidate_result={
                "promoted": candidate_promoted,
                "details": candidate_msg,
                "candidate_job_id": candidate_job_id,
            }
        )

        # Hoàn tất
        set_pipeline_step(
            3,
            "Đã tối ưu GUMNet" if candidate_promoted else "Giữ nguyên GUMNet hiện tại",
            candidate_msg,
            "done",
        )
        time.sleep(1)
        set_pipeline_step(
            4,
            "Hoàn tất",
            f"Quy trình xử lý hoàn tất. {candidate_msg}",
            "done",
        )
        update_pipeline_status(status="complete", is_running=False)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Worker đã kết thúc: {candidate_msg}", flush=True)
    except Exception as general_err:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL] Lỗi không mong đợi trong worker: {tb_str}", flush=True)
        update_pipeline_status(
            status="failed",
            is_running=False,
            error=f"Tiến trình bị gián đoạn: {general_err}",
            step_title="Sự cố xử lý",
        )
    finally:
        _release_pipeline_lock(pipeline_id)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Đã giải phóng khóa an toàn {pipeline_id}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-id", type=str, required=True)
    parser.add_argument("--batch-id", type=str, required=True)
    parser.add_argument("--new-rows", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true", default=False)
    args = parser.parse_args()

    run_pipeline_task(args.pipeline_id, args.batch_id, args.new_rows, args.force_retrain)
