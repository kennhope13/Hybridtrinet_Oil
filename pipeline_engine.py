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

from project_io import load_checkpoint

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


def _summarize_backtest(frame):
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False
    except Exception:
        return False


def _atomic_write_json(path: Path, data: Dict[str, Any]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _acquire_pipeline_lock(pipeline_id: str) -> bool:
    """Reserve the single background pipeline slot using O_EXCL."""
    payload = json.dumps({"pipeline_id": pipeline_id, "pid": os.getpid(), "created_at": time.time()})
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return True
    except FileExistsError:
        try:
            current = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            if _pid_alive(int(current.get("pid", 0))):
                if current.get("pipeline_id") != pipeline_id:
                    return False
                # App giữ chỗ trước khi Popen; worker cùng pipeline_id tiếp quản
                # và ghi PID thật để lock có thể được thu hồi nếu worker chết.
                _atomic_write_json(LOCK_FILE, {
                    "pipeline_id": pipeline_id,
                    "pid": os.getpid(),
                    "created_at": current.get("created_at", time.time()),
                })
                return True
            LOCK_FILE.unlink(missing_ok=True)
            return _acquire_pipeline_lock(pipeline_id)
        except Exception:
            return False


def _release_pipeline_lock(pipeline_id: str) -> None:
    try:
        current = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if current.get("pipeline_id") == pipeline_id:
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
            os.replace(tmp, target)
    except Exception:
        for horizon in HORIZONS:
            backup = backup_dir / f"gumnet_h{horizon}.pt"
            if backup.exists():
                tmp = CKPT_DIR / f"gumnet_h{horizon}.pt.rollback.tmp"
                shutil.copy2(backup, tmp)
                os.replace(tmp, CKPT_DIR / f"gumnet_h{horizon}.pt")
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
        if data.get("is_running") and pid:
            if not _pid_alive(pid):
                data["is_running"] = False
                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["error"] = "Tiến trình nền bị gián đoạn ngoài ý muốn (tiến trình dừng)."
                    data["step_title"] = "Đã dừng do gián đoạn"
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


def launch_pipeline_background(batch_id: str, new_rows: int, file_names: List[str]):
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

    update_pipeline_status(
        pipeline_id=pipeline_id,
        status="running",
        step_index=2,
        step_title="Đang đối chiếu độ chính xác",
        details=f"Đã ghi nhận {new_rows} ngày mới từ {len(file_names)} file. Đang chạy đối chiếu dự báo với giá thực tế...",
        steps=initial_steps,
        is_running=True,
        started_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        batch_info={"batch_id": batch_id, "new_rows": new_rows, "files": file_names},
        error=None,
    )

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

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = 0
    if os.name == "nt":
        # DETACHED_PROCESS trên Windows
        flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            env=env,
        )
        update_pipeline_status(pid=proc.pid)
        return {"started": True, "pipeline_id": pipeline_id, "pid": proc.pid}
    except Exception as e:
        update_pipeline_status(
            status="failed",
            is_running=False,
            error=f"Không thể khởi động tiến trình nền: {e}",
        )
        _release_pipeline_lock(pipeline_id)
        return {"started": False, "reason": "start_failed", "pipeline_id": pipeline_id}


def run_pipeline_task(pipeline_id: str, batch_id: str, new_rows: int):
    """Tiến trình thực thi chính của bộ điều phối ngầm (chạy trong tiến trình riêng)."""
    my_pid = os.getpid()
    lock_owned = _acquire_pipeline_lock(pipeline_id)
    if not lock_owned:
        return
    update_pipeline_status(pid=my_pid, is_running=True)

    # 1. BƯỚC 3: Chạy Backtest đối chiếu thực tế với dự báo cũ
    set_pipeline_step(
        2,
        "Đang đối chiếu độ chính xác",
        f"Đang tính toán sai số MAE/MAPE trên các dự báo đã đến hạn...",
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

        # backtest_worker nhận đường dẫn và trả về đúng một DataFrame.
        comb = bw.run_upload_simulation(
            bw.BUILTIN_CSV,
            files,
            start_date,
            sel_horizons=HORIZONS,
            sel_models=["GUMNet"],
            log_fn=lambda msg: None,
        )
        stats = _summarize_backtest(comb)

        # Lưu cache đối chiếu
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
        time.sleep(1)  # Giãn nhẹ để người dùng kịp quan sát tiến trình
    except Exception as e:
        # Nếu backtest lỗi, giữ kết quả cũ và ghi nhận lỗi nhưng không ngắt toàn hệ thống
        update_pipeline_status(
            backtest_result={"error": f"Lỗi đối chiếu: {e}"},
            error=f"Đối chiếu thực tế gặp sự cố: {e}. Dự báo hiện tại vẫn tiếp tục hoạt động.",
        )
        set_pipeline_step(2, "Đối chiếu chưa hoàn tất", "Kết quả cũ vẫn được giữ nguyên. Có thể thử lại sau.", "failed")
        update_pipeline_status(status="failed", is_running=False)
        _release_pipeline_lock(pipeline_id)
        return

    # 2. BƯỚC 4: Đánh giá nhu cầu huấn luyện lại
    mape = backtest_result.get("mape", 0.0) if backtest_result else 0.0
    samples = backtest_result.get("sample_count", 0) if backtest_result else 0

    retrain_needed = False
    decision_reason = ""

    if new_rows < MIN_NEW_ROWS:
        decision_reason = f"Dữ liệu mới ({new_rows} ngày) chưa đủ ngưỡng tối thiểu ({MIN_NEW_ROWS} ngày) để kích hoạt tối ưu."
    elif samples < MIN_EVAL_PAIRS:
        decision_reason = f"Chưa đủ số cặp đối chiếu dự báo ({samples}/{MIN_EVAL_PAIRS} cặp) để đánh giá độ tin cậy."
    elif mape <= MAPE_RETRAIN_THRESHOLD:
        decision_reason = f"Độ chính xác GUMNet hiện tại rất tốt (MAPE {mape:.2f}% ≤ ngưỡng {MAPE_RETRAIN_THRESHOLD}%). Giữ nguyên mô hình."
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
            f"Dữ liệu đã cập nhật. GUMNet hiện tại tiếp tục phục vụ dự báo (MAPE: {mape:.2f}%).",
            "done",
        )
        update_pipeline_status(status="complete", is_running=False)
        _release_pipeline_lock(pipeline_id)
        return

    # 3. NẾU CẦN HUẤN LUYỆN: Huấn luyện GUMNet Candidate ngầm
    set_pipeline_step(
        3,
        "Đang tối ưu GUMNet Candidate",
        f"MAPE hiện tại ({mape:.2f}%) cần cải thiện. Đang huấn luyện candidate ngầm...",
        "running",
    )

    candidate_job_id = f"CAND-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    candidate_out_dir = CANDIDATE_DIR / candidate_job_id
    candidate_out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline production phải đọc được trước khi huấn luyện. Không có baseline
    # thì vẫn có thể tạo candidate để kiểm tra, nhưng tuyệt đối không auto-promote.
    production_losses = _production_losses()

    # Sao lưu an toàn Production hiện tại
    backup_job_dir = BACKUP_DIR / candidate_job_id
    backup_job_dir.mkdir(parents=True, exist_ok=True)
    try:
        for horizon in HORIZONS:
            source = CKPT_DIR / f"gumnet_h{horizon}.pt"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, backup_job_dir / source.name)
            # Seed candidate bằng production hiện tại để đây là finetune thật,
            # không phải train ngẫu nhiên từ đầu trong output_dir rỗng.
            shutil.copy2(source, candidate_out_dir / source.name)
    except Exception as exc:
        update_pipeline_status(status="failed", is_running=False, error=f"Không thể tạo bản an toàn: {exc}")
        set_pipeline_step(3, "Không thể tạo bản an toàn", "GUMNet hiện tại không bị thay đổi.", "failed")
        _release_pipeline_lock(pipeline_id)
        return

    # Huấn luyện Candidate
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
            timeout=300,  # 5 phút tối đa cho demo
        )
        train_success = (proc_train.returncode == 0)
    except Exception as e:
        train_success = False

    # Đánh giá Candidate
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
    _release_pipeline_lock(pipeline_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-id", type=str, required=True)
    parser.add_argument("--batch-id", type=str, required=True)
    parser.add_argument("--new-rows", type=int, default=0)
    args = parser.parse_args()

    run_pipeline_task(args.pipeline_id, args.batch_id, args.new_rows)
