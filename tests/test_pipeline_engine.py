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


if __name__ == "__main__":
    unittest.main()
