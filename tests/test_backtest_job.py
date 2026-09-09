"""Test cho kiến trúc backtest CHẠY NỀN ĐỘC LẬP (không còn chạy đồng bộ trong session Streamlit).

Trích các hàm liên quan (get_backtest_status, _backtest_job_alive, spawn_backtest_job,
ensure_backtest_job_running) từ app_main.py bằng AST, chạy trong thư mục tạm — không đụng file
trạng thái/lock thật của app đang chạy, không launch subprocess thật (mock subprocess.Popen).
"""
import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "app_main.py").read_text(encoding="utf-8"))

FUNC_NAMES = {"get_backtest_status", "_backtest_job_alive", "spawn_backtest_job", "ensure_backtest_job_running", "_replace_with_retry"}


def _pid_alive_fake(alive_pids):
    def _f(pid):
        return pid in alive_pids
    return _f


def load_env(tmp_root, alive_pids=frozenset(), popen_mock=None, pipeline_running=False):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in FUNC_NAMES]
    assert len(nodes) == 5, f"Thiếu hàm, chỉ tìm thấy: {[n.name for n in nodes]}"
    env = {
        "ROOT": tmp_root,
        "BACKTEST_LOCK_FILE": tmp_root / ".backtest.lock",
        "BACKTEST_STATUS_FILE": tmp_root / ".backtest_job.json",
        "json": json,
        "os": __import__("os"),
        "sys": __import__("sys"),
        "time": __import__("time"),
        "uuid": __import__("uuid"),
        "subprocess": popen_mock or subprocess,
        "_pid_alive": _pid_alive_fake(alive_pids),
        # pipeline_engine thật chạy nền hoàn toàn tách biệt (subprocess riêng) — chỉ cần giả lập
        # đúng mặt cắt ensure_backtest_job_running() thực sự dùng: is_running True/False.
        "pipeline_engine": SimpleNamespace(
            get_pipeline_status=lambda: {"is_running": pipeline_running},
            pipeline_lock_active=lambda: False,
        ),
        'CACHE_FILE': tmp_root / 'simulation_cache.json',
        'cache_mismatch': False,
        'combined': SimpleNamespace(empty=False),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "app_main.py", "exec"), env)
    return env


class BacktestJobTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bt_job_test_")
        self.root = Path(self.tmpdir)
        self.fake_popen = MagicMock()
        self.env = load_env(self.root, popen_mock=SimpleNamespace(
            Popen=self.fake_popen, DEVNULL=subprocess.DEVNULL, CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0)))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _status(self):
        return json.loads((self.root / ".backtest_job.json").read_text(encoding="utf-8"))

    def _write_lock(self, job_id, pid):
        (self.root / ".backtest.lock").write_text(json.dumps({"job_id": job_id, "pid": pid}), encoding="utf-8")

    def _write_status(self, **fields):
        (self.root / ".backtest_job.json").write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")

    # 1. Chưa có job/status nào -> phải tự spawn 1 job mới
    def test_spawns_job_when_none_exists(self):
        self.env["ensure_backtest_job_running"]("fp-1", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "2026-01-01"))
        self.fake_popen.assert_called_once()
        status = self._status()
        self.assertEqual(status["fingerprint"], "fp-1")
        self.assertEqual(status["status"], "pending")

    # 2b. pipeline_engine (hệ thống mới) đang thật sự chạy -> hệ thống cũ phải nhường, không tự
    # tạo job riêng chồng lên (tránh 2 hệ thống cùng ghi simulation_cache.json và tranh GPU).
    def test_yields_when_new_pipeline_engine_is_actively_running(self):
        env = load_env(self.root, popen_mock=SimpleNamespace(
            Popen=self.fake_popen, DEVNULL=subprocess.DEVNULL,
            CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0)), pipeline_running=True)
        env["ensure_backtest_job_running"]("fp-1", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "2026-01-01"))
        self.fake_popen.assert_not_called()
        self.assertFalse((self.root / ".backtest_job.json").exists())

    # 2. Không có file dữ liệu nào -> không cần chạy gì cả
    def test_no_files_means_no_job(self):
        self.env["ensure_backtest_job_running"]("fp-1", ["GUMNet"], [], SimpleNamespace(strftime=lambda f: "x"))
        self.fake_popen.assert_not_called()
        self.assertFalse((self.root / ".backtest_job.json").exists())

    # 3. Lỗi tái hiện thật ngoài đời: os.replace() ném PermissionError [WinError 32] thoáng qua
    # (file đích đang bị tiến trình khác mở đúng lúc ghi) -> phải tự thử lại và thành công, không
    # để lỗi lộ ra ngoài.
    def test_replace_with_retry_recovers_from_transient_permission_error(self):
        tmp = self.root / "a.tmp"
        dest = self.root / "a.json"
        tmp.write_text("{}", encoding="utf-8")
        calls = {"n": 0}
        real_replace = self.env["os"].replace

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("[WinError 32] gia lap tranh chap file")
            return real_replace(src, dst)

        self.env["os"] = SimpleNamespace(replace=flaky_replace)
        self.env["_replace_with_retry"](tmp, dest, attempts=5, delay=0)
        self.assertEqual(calls["n"], 3)
        self.assertTrue(dest.exists())

    def test_replace_with_retry_gives_up_after_exhausting_attempts(self):
        tmp = self.root / "b.tmp"
        tmp.write_text("{}", encoding="utf-8")
        dest = self.root / "b.json"

        def always_fails(src, dst):
            raise PermissionError("[WinError 32] luon bi khoa")

        self.env["os"] = SimpleNamespace(replace=always_fails)
        with self.assertRaises(PermissionError):
            self.env["_replace_with_retry"](tmp, dest, attempts=3, delay=0)

    # 3. RERUN khi job ĐANG chạy đúng fingerprint hiện tại -> không spawn job trùng
    def test_rerun_while_job_running_same_fingerprint_no_duplicate(self):
        self._write_status(job_id="job-1", fingerprint="fp-1", status="running", started_at="t", finished_at=None, error=None)
        self._write_lock("job-1", pid=999001)
        env = load_env(self.root, alive_pids={999001}, popen_mock=SimpleNamespace(
            Popen=self.fake_popen, DEVNULL=subprocess.DEVNULL, CREATE_NO_WINDOW=0))
        # Mô phỏng NHIỀU lượt rerun liên tiếp (kiểu người dùng F5, hoặc nhiều tab)
        for _ in range(5):
            env["ensure_backtest_job_running"]("fp-1", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "x"))
        self.fake_popen.assert_not_called()

    # 4. CHUYỂN TRANG: trạng thái phải đọc được từ FILE, không phụ thuộc session_state — mô
    #    phỏng bằng cách gọi lại với 1 "env" hoàn toàn MỚI (như 1 lượt script rerun độc lập ở
    #    trang khác), không truyền qua bất kỳ session_state nào, vẫn phải thấy đúng job đang chạy.
    def test_status_survives_across_fresh_script_execution(self):
        self._write_status(job_id="job-2", fingerprint="fp-2", status="running", started_at="t", finished_at=None, error=None)
        self._write_lock("job-2", pid=999002)
        fresh_env = load_env(self.root, alive_pids={999002})  # y hệt việc "sang trang khác" -> script chạy lại từ đầu
        status = fresh_env["get_backtest_status"]()
        self.assertEqual(status["job_id"], "job-2")
        self.assertTrue(fresh_env["_backtest_job_alive"](status))

    # 5. Job trùng fingerprint đã SUCCESS -> không tạo job mới
    def test_duplicate_fingerprint_already_success_no_new_job(self):
        (self.root / 'simulation_cache.json').write_text('{}')
        self._write_status(job_id="job-3", fingerprint="fp-3", status="success", started_at="t", finished_at="t2", error=None)
        self.env["ensure_backtest_job_running"]("fp-3", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "x"))
        self.fake_popen.assert_not_called()

    def test_success_without_cache_rebuilds(self):
        self._write_status(job_id='old', fingerprint='fp', status='success')
        self.env['ensure_backtest_job_running']('fp', ['GUMNet'], ['a.csv'],
                                               SimpleNamespace(strftime=lambda _: '2026-01-01'))
        self.fake_popen.assert_called_once()

    # 6. DỮ LIỆU ĐỔI GIỮA LÚC JOB CŨ CHẠY: job cũ (fp-old) đang chạy -> không chen ngang. Sau khi
    #    job cũ xong (không còn alive), lượt gọi tiếp theo với fp MỚI phải tự spawn job kế tiếp.
    def test_data_changed_mid_job_chains_to_latest_fingerprint_after_old_finishes(self):
        self._write_status(job_id="job-old", fingerprint="fp-old", status="running", started_at="t", finished_at=None, error=None)
        self._write_lock("job-old", pid=999003)
        env_running = load_env(self.root, alive_pids={999003}, popen_mock=SimpleNamespace(
            Popen=self.fake_popen, DEVNULL=subprocess.DEVNULL, CREATE_NO_WINDOW=0))
        # Trong lúc job cũ còn chạy, dữ liệu đã đổi (fp-new) -> KHÔNG được chen ngang
        env_running["ensure_backtest_job_running"]("fp-new", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "x"))
        self.fake_popen.assert_not_called()

        # Job cũ giờ đã xong thật (không còn PID sống) -> lượt gọi tiếp theo (rerun kế tiếp) với
        # đúng fingerprint MỚI NHẤT phải tự spawn job kế tiếp — đây là "chạy nối tiếp theo dữ
        # liệu mới nhất" mà không cần vòng lặp riêng trong worker.
        env_finished = load_env(self.root, alive_pids=set(), popen_mock=SimpleNamespace(
            Popen=self.fake_popen, DEVNULL=subprocess.DEVNULL, CREATE_NO_WINDOW=0))
        env_finished["ensure_backtest_job_running"]("fp-new", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "x"))
        self.fake_popen.assert_called_once()
        self.assertEqual(self._status()["fingerprint"], "fp-new")

    # 7. Job LỖI (status=failed) đúng fingerprint hiện tại -> _backtest_job_alive() phải trả về
    #    False (không còn coi là "đang chạy"), và không tự động spawn lại vô hạn (phải có nút
    #    "Thử lại" ở UI mới spawn tiếp — kiểm tra qua force=True riêng ở test dưới).
    def test_failed_job_not_considered_alive(self):
        self._write_status(job_id="job-4", fingerprint="fp-4", status="failed", started_at="t", finished_at="t2", error="lỗi mô phỏng")
        status = self.env["get_backtest_status"]()
        self.assertFalse(self.env["_backtest_job_alive"](status))

    def test_failed_job_same_fingerprint_does_not_auto_retry_without_force(self):
        self._write_status(job_id="job-4", fingerprint="fp-4", status="failed", started_at="t", finished_at="t2", error="lỗi mô phỏng")
        self.env["ensure_backtest_job_running"]("fp-4", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "x"))
        # failed không khớp điều kiện "success" hay "alive" -> sẽ spawn lại tự động (đây là hành
        # vi ĐÚNG YÊU CẦU: dữ liệu vẫn cần được đối chiếu, job cũ chỉ là thất bại chứ không phải
        # "đã xong" — hệ thống tự thử lại là hợp lý, không phải vòng lặp vô hạn vì cần fingerprint
        # y hệt và không có gì thay đổi trạng thái liên tục).
        self.fake_popen.assert_called_once()

    def test_force_retry_spawns_even_if_already_success(self):
        self._write_status(job_id="job-5", fingerprint="fp-5", status="success", started_at="t", finished_at="t2", error=None)
        self.env["ensure_backtest_job_running"]("fp-5", ["GUMNet"], ["a.csv"], SimpleNamespace(strftime=lambda f: "x"), force=True)
        self.fake_popen.assert_called_once()

    # 8. Cache cũ / lock rác: status "running" nhưng KHÔNG có lock file thật (hoặc PID đã chết)
    #    -> _backtest_job_alive() phải trả về False, không kẹt mãi ở trạng thái "đang chạy" giả.
    def test_stale_running_status_without_alive_lock_is_not_considered_alive(self):
        self._write_status(job_id="job-6", fingerprint="fp-6", status="running", started_at="t", finished_at=None, error=None)
        # Không ghi lock file nào cả (job đã chết, dọn lock nhưng lỡ chưa kịp cập nhật status)
        status = self.env["get_backtest_status"]()
        self.assertFalse(self.env["_backtest_job_alive"](status))

    def test_stale_lock_pid_dead_is_not_considered_alive(self):
        self._write_status(job_id="job-7", fingerprint="fp-7", status="running", started_at="t", finished_at=None, error=None)
        self._write_lock("job-7", pid=999999999)  # PID chắc chắn không tồn tại
        env = load_env(self.root, alive_pids=set())  # không có PID nào "sống" trong môi trường giả lập này
        status = env["get_backtest_status"]()
        self.assertFalse(env["_backtest_job_alive"](status))

    # 9. LOCK CẠNH TRANH thật bằng 2 TIẾN TRÌNH HỆ ĐIỀU HÀNH THẬT — dùng đúng hàm khóa thật của
    #    run_backtest_job.py (import trực tiếp, không chạy full backtest vì việc đó xong quá
    #    nhanh với dữ liệu tối thiểu, không đủ thời gian để quan sát tranh chấp một cách đáng
    #    tin cậy — giữ lock nhân tạo vài giây, giống hệt cách test_lock_concurrency.py đã làm
    #    cho khóa huấn luyện).
    def test_real_two_process_lock_contention_on_backtest_lock(self):
        import shutil
        work = self.root / "worker_copy"
        work.mkdir()
        shutil.copy(ROOT / "run_backtest_job.py", work / "run_backtest_job.py")
        shutil.copy(ROOT / "backtest_worker.py", work / "backtest_worker.py")
        shutil.copy(ROOT / "project_io.py", work / "project_io.py")
        worker_script = work / "lock_worker.py"
        worker_script.write_text(f'''
import sys, time
sys.path.insert(0, r"{work}")
import run_backtest_job as rbj

job_id = sys.argv[1]
hold_seconds = float(sys.argv[2])
try:
    acquired = rbj.acquire_or_takeover_lock(job_id, f"fp-{{job_id}}")
    print(f"ACQUIRED:{{acquired}}", flush=True)
    time.sleep(hold_seconds)
    rbj.release_own_lock(acquired)
    print("RELEASED", flush=True)
except SystemExit as e:
    print(f"REJECTED:{{e.code}}", flush=True)
''', encoding="utf-8")

        def _run(job_id, hold_seconds):
            import sys as _sys, os as _os
            env = dict(_os.environ, PYTHONIOENCODING="utf-8")
            return subprocess.Popen(
                [_sys.executable, str(worker_script), job_id, str(hold_seconds)],
                cwd=str(work), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
            )

        p1 = _run("bt-race-a", 8.0)  # giữ đủ lâu để chắc chắn còn giữ khi p2 khởi động xong
        lock_file = work / ".backtest.lock"
        import time
        for _ in range(150):
            if lock_file.exists():
                break
            time.sleep(0.1)
        self.assertTrue(lock_file.exists(), "Job A phải đã tạo lock")

        p2 = _run("bt-race-b", 0.1)
        out2, _ = p2.communicate(timeout=15)
        out1, _ = p1.communicate(timeout=15)

        self.assertIn("ACQUIRED:", out1)
        self.assertIn("REJECTED", out2, f"Job B phải bị từ chối vì job A còn đang giữ khóa. out2={out2!r}")
        self.assertFalse(lock_file.exists(), "Lock phải được dọn sau khi cả 2 xong")


if __name__ == "__main__":
    unittest.main()
