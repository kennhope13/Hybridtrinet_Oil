"""Test khóa huấn luyện bằng 2 TIẾN TRÌNH HỆ ĐIỀU HÀNH THẬT chạy đồng thời (không mô phỏng
tuần tự trong cùng 1 process) — xác nhận cơ chế acquire_or_takeover_lock() trong
train_all_horizons.py thực sự an toàn khi có 2 job cùng cố khởi động.

Chạy trong 1 thư mục tạm cô lập (copy riêng train_all_horizons.py + project_io.py sang đó) để
không đụng tới .training.lock thật của app đang chạy.
"""
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER_SCRIPT = '''
import sys, time
sys.path.insert(0, r"{workdir}")
import train_all_horizons as tah

job_id = sys.argv[1] if len(sys.argv) > 1 else None
hold_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
try:
    acquired_id = tah.acquire_or_takeover_lock(job_id, ["GUMNet"], [1])
    print(f"ACQUIRED:{{acquired_id}}", flush=True)
    time.sleep(hold_seconds)
    tah.release_own_lock(acquired_id)
    print("RELEASED", flush=True)
except SystemExit as e:
    print(f"REJECTED:{{e.code}}", flush=True)
'''


class LockConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="lock_race_")
        self.workdir = Path(self.tmpdir)
        shutil.copy(ROOT / "train_all_horizons.py", self.workdir / "train_all_horizons.py")
        shutil.copy(ROOT / "project_io.py", self.workdir / "project_io.py")
        self.worker_path = self.workdir / "worker.py"
        self.worker_path.write_text(WORKER_SCRIPT.format(workdir=str(self.workdir)), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_worker(self, job_id_arg, hold_seconds):
        import os
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.Popen(
            [sys.executable, str(self.worker_path), job_id_arg or "", str(hold_seconds)],
            cwd=str(self.workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )

    def test_two_different_jobs_race_only_one_wins(self):
        """2 job THẬT SỰ KHÁC NHAU (job_id khác nhau) cùng cố khởi động cùng lúc — chỉ 1 thắng."""
        # p1 phải giữ lock đủ lâu để CHẮC CHẮN còn đang giữ khi p2 khởi động xong (python +
        # import torch của p2 có thể mất vài giây) — nếu không, 2 job sẽ chỉ chạy NỐI TIẾP nhau
        # (mỗi job tự thành công riêng) thay vì THẬT SỰ TRANH CHẤP cùng lúc, làm test vô nghĩa.
        p1 = self._run_worker("", 8.0)
        lock_file = self.workdir / ".training.lock"
        for _ in range(100):
            if lock_file.exists():
                break
            time.sleep(0.1)
        p2 = self._run_worker("", 0.1)

        out2, _ = p2.communicate(timeout=15)
        out1, _ = p1.communicate(timeout=15)

        acquired = [o for o in (out1, out2) if "ACQUIRED:" in o]
        rejected = [o for o in (out1, out2) if "REJECTED:" in o]
        self.assertEqual(len(acquired), 1, f"Phải đúng 1 process giành được lock. out1={out1!r} out2={out2!r}")
        self.assertEqual(len(rejected), 1, f"Phải đúng 1 process bị từ chối. out1={out1!r} out2={out2!r}")
        self.assertFalse((self.workdir / ".training.lock").exists(), "Lock phải được dọn sau khi cả 2 process kết thúc")

    def test_same_job_id_takeover_not_treated_as_conflict(self):
        """Script con dùng ĐÚNG job_id mà 'app' đã giữ chỗ trước đó -> phải tiếp quản được, không
        bị coi là xung đột với chính nó."""
        placeholder = self.workdir / ".training.lock"
        import json
        placeholder.write_text(json.dumps({
            "job_id": "shared-job-001", "pid": 999999999,  # PID chắc chắn không tồn tại/không phải mình
            "models": ["GUMNet"], "horizons": [1], "started_at": "2026-01-01 00:00:00",
        }), encoding="utf-8")

        p = self._run_worker("shared-job-001", 0.3)
        out, _ = p.communicate(timeout=10)
        self.assertIn("ACQUIRED:shared-job-001", out, f"Phải tiếp quản đúng job_id placeholder. out={out!r}")

    def test_other_job_cannot_delete_active_lock(self):
        """Job B (job_id khác, PID khác đang sống) không được phép xóa lock của job A đang chạy."""
        p_a = self._run_worker("job-a", 8.0)
        lock_file = self.workdir / ".training.lock"
        for _ in range(100):  # đợi tới 10s cho job A import xong (torch) và tạo lock, thay vì
            if lock_file.exists():
                break
            time.sleep(0.1)
        self.assertTrue(lock_file.exists(), "Job A phải đã tạo lock")
        content_before = lock_file.read_text(encoding="utf-8")

        p_b = self._run_worker("job-b", 0.1)
        out_b, _ = p_b.communicate(timeout=10)
        self.assertIn("REJECTED", out_b, f"Job B phải bị từ chối vì job A còn sống. out_b={out_b!r}")

        # Lock của Job A phải còn nguyên, không bị Job B xóa/ghi đè trong lúc bị từ chối
        self.assertTrue(lock_file.exists(), "Lock của Job A không được biến mất do Job B")
        self.assertEqual(lock_file.read_text(encoding="utf-8"), content_before, "Nội dung lock của Job A không được đổi")

        out_a, _ = p_a.communicate(timeout=10)
        self.assertIn("ACQUIRED:job-a", out_a)


if __name__ == "__main__":
    unittest.main()
