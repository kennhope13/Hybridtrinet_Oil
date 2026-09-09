import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys
import json
from types import SimpleNamespace

import pandas as pd

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline_engine as pe


class PipelineEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = (pe.ROOT, pe.LOCK_FILE, pe.STATUS_FILE, pe.CKPT_DIR)
        pe.ROOT = self.root
        pe.LOCK_FILE = self.root / ".pipeline.lock"
        pe.STATUS_FILE = self.root / ".pipeline_status.json"
        pe.CKPT_DIR = self.root / "production"
        pe.CKPT_DIR.mkdir()

    def tearDown(self):
        pe.ROOT, pe.LOCK_FILE, pe.STATUS_FILE, pe.CKPT_DIR = self.old
        self.temp.cleanup()

    def checkpoint(self, folder, horizon, job_id, loss):
        torch.save({
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "horizon": horizon,
            "feature_cols": ["x"],
            "target_cols": ["MG95"],
            "job_id": job_id,
            "best_val_loss": loss,
        }, folder / f"gumnet_h{horizon}.pt")

    def test_dead_pid_running_status_is_persisted_as_failed_on_disk(self):
        # Trước đây get_pipeline_status() chỉ sửa "running" -> "failed" TẠM trong bộ nhớ để trả
        # về, không ghi lại xuống file -> banner đọc lại đúng file cũ ở lượt sau, hiển thị "đang
        # chạy" mãi mãi dù tiến trình đã chết. Test này xác nhận giờ đã ghi thẳng xuống đĩa.
        dead_pid = 999999999
        pe.STATUS_FILE.write_text(json.dumps({
            "pipeline_id": "PIPE-X", "status": "running", "is_running": True, "pid": dead_pid,
            "step_title": "Đang đối chiếu độ chính xác",
            "steps": [
                {"title": "File hợp lệ", "state": "done"},
                {"title": "Đối chiếu độ chính xác", "state": "running"},
                {"title": "Hoàn tất", "state": "waiting"},
            ],
            "error": None,
        }), encoding="utf-8")

        first = pe.get_pipeline_status()
        self.assertEqual(first["status"], "failed")
        self.assertFalse(first["is_running"])

        # Đọc thẳng lại từ đĩa (không qua get_pipeline_status) để chắc chắn đã ghi xuống thật,
        # không chỉ sửa tạm trong bộ nhớ để trả về.
        on_disk = json.loads(pe.STATUS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], "failed")
        self.assertFalse(on_disk["is_running"])
        self.assertEqual(on_disk["steps"][1]["state"], "failed")

    # Lỗi thật đã sửa: khóa bị cướp sau 600s DÙ tiến trình vẫn sống, trong khi 1 lượt chạy
    # hợp lệ (đối chiếu + huấn luyện candidate) có thể lâu hơn thế -> 2 pipeline chạy song song.
    def test_live_lock_is_not_stolen_before_stale_timeout(self):
        import os
        import time as _time

        # Khóa của 1 tiến trình CÒN SỐNG (chính tiến trình test), đã giữ 11 phút —
        # dài hơn thời gian huấn luyện tối đa nhưng vẫn trong hạn cho phép.
        pe.LOCK_FILE.write_text(json.dumps({
            "pipeline_id": "PIPE-DANG-HUAN-LUYEN",
            "pid": os.getpid(),
            "created_at": _time.time() - (pe.TRAIN_TIMEOUT + 60),
        }), encoding="utf-8")

        self.assertFalse(
            pe._acquire_pipeline_lock("PIPE-KHAC"),
            "Không được cướp khóa của tiến trình đang chạy thật trong lúc huấn luyện",
        )
        still = json.loads(pe.LOCK_FILE.read_text(encoding="utf-8"))
        self.assertEqual(still["pipeline_id"], "PIPE-DANG-HUAN-LUYEN")

    def test_stale_timeout_must_exceed_train_timeout(self):
        self.assertGreater(
            pe.STALE_LOCK_TIMEOUT, pe.TRAIN_TIMEOUT,
            "Hạn coi khóa là rác phải dài hơn thời gian huấn luyện, nếu không sẽ tự cướp khóa của chính mình",
        )

    # Lỗi thật đã sửa: chỉ chặn 1 chiều (hệ cũ nhường pipeline), thiếu chiều ngược lại nên
    # job đối chiếu cũ đang chạy mà bấm "Xử lý" là 2 tiến trình cùng dùng GPU + ghi chung cache.
    def test_launch_refuses_while_old_backtest_job_is_running(self):
        import os

        (self.root / ".backtest.lock").write_text(
            json.dumps({"job_id": "BT-1", "pid": os.getpid()}), encoding="utf-8"
        )
        result = pe.launch_pipeline_background("BATCH-X", 10, ["a.csv"])
        self.assertFalse(result["started"])
        self.assertEqual(result["reason"], "backtest_running")
        self.assertFalse(pe.LOCK_FILE.exists(), "Bị từ chối thì không được để lại khóa mồ côi")

    def test_launch_ignores_orphaned_backtest_lock(self):
        # Khóa mồ côi (PID đã chết) thì không được chặn oan.
        (self.root / ".backtest.lock").write_text(
            json.dumps({"job_id": "BT-CU", "pid": 999999999}), encoding="utf-8"
        )
        self.assertFalse(pe._backtest_job_running())

    def test_pipeline_lock_allows_only_owner(self):
        self.assertTrue(pe._acquire_pipeline_lock("one"))
        self.assertFalse(pe._acquire_pipeline_lock("two"))
        pe._release_pipeline_lock("two")
        self.assertTrue(pe.LOCK_FILE.exists())
        pe._release_pipeline_lock("one")
        self.assertFalse(pe.LOCK_FILE.exists())

    def test_candidate_requires_all_valid_job_checkpoints(self):
        candidate = self.root / "candidate"
        candidate.mkdir()
        for horizon in pe.HORIZONS:
            self.checkpoint(candidate, horizon, "right", 1.0)
        self.assertEqual(set(pe._validate_candidate(candidate, "right")), set(pe.HORIZONS))
        with self.assertRaises(ValueError):
            pe._validate_candidate(candidate, "wrong")

    def test_missing_baseline_never_promotes(self):
        candidate = {h: 0.1 for h in pe.HORIZONS}
        self.assertFalse(pe._candidate_is_better(candidate, None))
        self.assertTrue(pe._candidate_is_better(candidate, {h: 0.2 for h in pe.HORIZONS}))

    # Lỗi thật đã sửa: lịch sử huấn luyện tồn tại 2 định dạng trạng thái ("success" của bản mới
    # và "Hoàn thành 100%" của bản cũ). Code chỉ so khớp đúng chuỗi "success" nên bỏ qua hết bản
    # ghi cũ -> không có baseline -> candidate dù huấn luyện thành công cũng không bao giờ được
    # áp dụng. Đây chính là lý do khâu "tối ưu" bế tắc trên máy thật.
    def test_production_loss_accepts_legacy_vietnamese_status(self):
        results = {f"GUMNet_h{h}": 0.2 + h / 1000 for h in pe.HORIZONS}
        (pe.CKPT_DIR / "training_history.json").write_text(
            json.dumps([{"status": "Hoàn thành 100%", "results": results}]), encoding="utf-8"
        )
        losses = pe._production_losses()
        self.assertIsNotNone(losses, "Bản ghi định dạng cũ vẫn phải dùng được làm baseline")
        self.assertEqual(losses[1], results["GUMNet_h1"])

    def test_production_loss_rejects_failed_training_entry(self):
        # Không được nhận nhầm phiên LỖI làm baseline (giữ nguyên tắc fail-closed).
        results = {f"GUMNet_h{h}": 0.2 for h in pe.HORIZONS}
        (pe.CKPT_DIR / "training_history.json").write_text(
            json.dumps([{"status": "Lỗi: hết bộ nhớ", "results": results}]), encoding="utf-8"
        )
        self.assertIsNone(pe._production_losses())

    def test_production_loss_falls_back_to_training_history(self):
        results = {f"GUMNet_h{h}": 0.2 + h / 1000 for h in pe.HORIZONS}
        (pe.CKPT_DIR / "training_history.json").write_text(
            json.dumps([{"status": "success", "results": results}]), encoding="utf-8"
        )
        self.assertEqual(pe._production_losses()[1], results["GUMNet_h1"])

    def test_promotion_rolls_back_all_files_on_copy_failure(self):
        candidate = self.root / "candidate"
        backup = self.root / "backup"
        candidate.mkdir()
        for horizon in pe.HORIZONS:
            (pe.CKPT_DIR / f"gumnet_h{horizon}.pt").write_bytes(f"old-{horizon}".encode())
            (candidate / f"gumnet_h{horizon}.pt").write_bytes(f"new-{horizon}".encode())
        real_copy = pe.shutil.copy2
        calls = {"candidate": 0}

        def fail_midway(source, target, *args, **kwargs):
            if Path(source).parent == candidate:
                calls["candidate"] += 1
                if calls["candidate"] == 3:
                    raise OSError("disk failure")
            return real_copy(source, target, *args, **kwargs)

        with mock.patch.object(pe.shutil, "copy2", side_effect=fail_midway):
            with self.assertRaises(OSError):
                pe._promote_with_rollback(candidate, backup)
        for horizon in pe.HORIZONS:
            self.assertEqual(
                (pe.CKPT_DIR / f"gumnet_h{horizon}.pt").read_bytes(),
                f"old-{horizon}".encode(),
            )

    def test_run_pipeline_task_uses_real_backtest_contract(self):
        data_dir = self.root / "datasets"
        data_dir.mkdir()
        upload = data_dir / "new.csv"
        upload.write_text("Ngày,MG95\n2026-09-01,10\n", encoding="utf-8")
        builtin = self.root / "base.csv"
        builtin.write_text("Ngày,MG95\n2026-01-01,9\n", encoding="utf-8")
        calls = {}

        def load_df(path):
            return pd.read_csv(path, parse_dates=["Ngày"])

        def run_upload_simulation(base_path, upload_files, start_date,
                                  sel_horizons=None, sel_models=None, log_fn=None):
            calls.update(base_path=base_path, upload_files=upload_files,
                         start_date=start_date, sel_horizons=sel_horizons,
                         sel_models=sel_models)
            return pd.DataFrame({"% Lệch": [5.0, 7.0], "Sai lệch": [1.0, 3.0]})

        fake_worker = SimpleNamespace(
            BUILTIN_CSV=builtin,
            load_df=load_df,
            run_upload_simulation=run_upload_simulation,
        )
        self.assertTrue(pe._acquire_pipeline_lock("pipe-test"))
        with mock.patch.dict(sys.modules, {"backtest_worker": fake_worker}), \
             mock.patch("project_io.write_cache"):
            pe.run_pipeline_task("pipe-test", "batch-test", 1)

        status = pe.get_pipeline_status()
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["backtest_result"]["mape"], 6.0)
        self.assertEqual(status["backtest_result"]["mae"], 2.0)
        self.assertEqual(calls["sel_models"], ["GUMNet"])
        self.assertEqual(calls["sel_horizons"], pe.HORIZONS)
        self.assertFalse(pe.LOCK_FILE.exists())

    def test_run_pipeline_task_marks_failed_when_backtest_raises(self):
        data_dir = self.root / "datasets"
        data_dir.mkdir()
        builtin = self.root / "base.csv"
        builtin.write_text("Ngày,MG95\n2026-01-01,9\n", encoding="utf-8")
        fake_worker = SimpleNamespace(
            BUILTIN_CSV=builtin,
            load_df=lambda path: pd.read_csv(path, parse_dates=["Ngày"]),
            run_upload_simulation=lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("contract error")),
        )
        self.assertTrue(pe._acquire_pipeline_lock("pipe-fail"))
        with mock.patch.dict(sys.modules, {"backtest_worker": fake_worker}), \
             mock.patch("project_io.write_cache"):
            pe.run_pipeline_task("pipe-fail", "batch-fail", 10)
        status = pe.get_pipeline_status()
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["is_running"])
        self.assertFalse(pe.LOCK_FILE.exists())


class PipelineSupervisorTests(unittest.TestCase):
    """Watchdog là lớp bảo vệ CUỐI CÙNG: nó là thứ duy nhất báo được 'worker đã chết' khi
    worker chết kiểu native. Nếu chính nó chết vì WinError 32 thì trạng thái kẹt vĩnh viễn."""

    def setUp(self):
        import pipeline_supervisor as ps
        self.ps = ps
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_status = ps.STATUS_FILE
        ps.STATUS_FILE = self.root / ".pipeline_status.json"

    def tearDown(self):
        self.ps.STATUS_FILE = self.old_status
        self.temp.cleanup()

    def test_write_status_recovers_from_transient_permission_error(self):
        calls = {"n": 0}
        real_replace = self.ps.os.replace

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("[WinError 32] gia lap xung dot file")
            return real_replace(src, dst)

        with mock.patch.object(self.ps.os, "replace", side_effect=flaky):
            self.ps._write_status({"status": "failed", "is_running": False}, delay=0)

        self.assertEqual(calls["n"], 3)
        saved = json.loads(self.ps.STATUS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")

    def test_write_status_falls_back_to_direct_write_when_always_blocked(self):
        def always_blocked(src, dst):
            raise PermissionError("[WinError 32] luon bi chan")

        with mock.patch.object(self.ps.os, "replace", side_effect=always_blocked):
            # Không được ném lỗi ra ngoài: thà mất tính nguyên tử còn hơn mất hẳn trạng thái.
            self.ps._write_status({"status": "failed", "is_running": False}, attempts=3, delay=0)

        saved = json.loads(self.ps.STATUS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(self.ps.STATUS_FILE.with_suffix(".json.tmp").exists(),
                         "Phải dọn file tạm sau khi ghi trực tiếp")


if __name__ == "__main__":
    unittest.main()
