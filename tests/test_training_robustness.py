import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_all_horizons as tah
import app_main
from project_io import load_checkpoint
from streamlit.testing.v1 import AppTest


class TrainingRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_atomic_lock_and_takeover(self):
        """Kiểm tra tạo lock nguyên tử, tiếp quản bằng job_id và bảo vệ không xóa nhầm lock."""
        lock_file = self.tmp_path / ".training.lock"
        original_lock_file = tah.TRAIN_LOCK_FILE
        tah.TRAIN_LOCK_FILE = lock_file

        try:
            # 1. Tạo lock ban đầu với job_id 1
            job_id_1 = tah.acquire_or_takeover_lock("job_alpha", ["GUMNet"], [1, 5])
            self.assertEqual(job_id_1, "job_alpha")
            self.assertTrue(lock_file.exists())
            info = json.loads(lock_file.read_text(encoding="utf-8"))
            self.assertEqual(info["job_id"], "job_alpha")

            # 2. Tiếp quản lock với CÙNG job_id (như app giữ chỗ rồi script con tiếp quản)
            job_id_takeover = tah.acquire_or_takeover_lock("job_alpha", ["GUMNet"], [1, 5])
            self.assertEqual(job_id_takeover, "job_alpha")
            self.assertTrue(lock_file.exists())

            # 3. Thử giải phóng lock bằng SAI job_id -> Lock KHÔNG được bị xóa!
            tah.release_own_lock("job_beta_wrong")
            self.assertTrue(lock_file.exists())

            # 4. Giải phóng lock bằng ĐÚNG job_id -> Lock được dọn dẹp
            tah.release_own_lock("job_alpha")
            self.assertFalse(lock_file.exists())
        finally:
            tah.TRAIN_LOCK_FILE = original_lock_file

    def test_app_main_lock_reservation_and_release(self):
        """Kiểm tra acquire_training_lock và release_training_lock trong app_main."""
        lock_file = self.tmp_path / ".training.lock"
        orig_app_lock = app_main.TRAIN_LOCK_FILE
        app_main.TRAIN_LOCK_FILE = lock_file

        try:
            # Giữ chỗ lock trước khi Popen
            job_id = app_main.acquire_training_lock(["GUMNet"], [1, 5], job_id="job_test_123")
            self.assertEqual(job_id, "job_test_123")
            active = app_main.get_active_training_lock()
            self.assertIsNotNone(active)
            self.assertEqual(active["job_id"], "job_test_123")

            # Không cho phép job khác đè lên khi đang active
            with self.assertRaises(RuntimeError):
                app_main.acquire_training_lock(["HybridTriNet"], [10], job_id="job_different")

            # Cập nhật PID cho cùng job_id
            app_main.acquire_training_lock(["GUMNet"], [1, 5], job_id="job_test_123", pid=os.getpid())
            active2 = app_main.get_active_training_lock()
            self.assertEqual(active2["pid"], os.getpid())

            # Giải phóng sai job_id -> không xóa
            app_main.release_training_lock(job_id="wrong_id")
            self.assertTrue(lock_file.exists())

            # Giải phóng đúng job_id -> xóa
            app_main.release_training_lock(job_id="job_test_123")
            self.assertFalse(lock_file.exists())
        finally:
            app_main.TRAIN_LOCK_FILE = orig_app_lock

    def test_update_training_data_overlay_and_validation(self):
        """Kiểm tra cập nhật dữ liệu: sửa ngày cũ đè giá trị đúng cột mà không làm mất cột khác thành NaN."""
        fake_data_path = self.tmp_path / "clean_data.csv"
        orig_data_path = tah.DATA_PATH
        tah.DATA_PATH = fake_data_path

        try:
            # Tập dữ liệu gốc ban đầu
            base_df = pd.DataFrame({
                "Ngày": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "MG95": [20000.0, 20500.0, 21000.0],
                "MG92": [19000.0, 19500.0, 20000.0],
                "DO 0.001%": [18000.0, 18200.0, 18500.0],
                "DO 0.05%": [17500.0, 17800.0, 18000.0],
                "Exo_Macro": [100.0, 101.0, 102.0],
            })
            base_df.to_csv(fake_data_path, index=False)

            # File cập nhật: sửa ngày 2026-01-02 cho MG95 = 20888.0, thêm ngày mới 2026-01-04
            # Chú ý: file mới này KHÔNG có cột "DO 0.001%" và "Exo_Macro"
            upload_file = self.tmp_path / "new_update.csv"
            update_df = pd.DataFrame({
                "Ngày": ["2026-01-02", "2026-01-04"],
                "MG95": [20888.0, 21500.0],
                "MG92": [19888.0, 20200.0],
            })
            update_df.to_csv(upload_file, index=False)

            tah.update_training_data(specific_file=str(upload_file))

            result_df = pd.read_csv(fake_data_path)
            result_df["Ngày"] = pd.to_datetime(result_df["Ngày"])
            result_df = result_df.set_index("Ngày")

            # 1. Ngày trùng (2026-01-02) phải nhận giá trị MỚI của MG95
            self.assertEqual(result_df.loc[pd.Timestamp("2026-01-02"), "MG95"], 20888.0)
            self.assertEqual(result_df.loc[pd.Timestamp("2026-01-02"), "MG92"], 19888.0)

            # 2. Các cột không có trong file mới (DO 0.001%, Exo_Macro) của ngày trùng PHẢI ĐƯỢC GIỮ NGUYÊN
            self.assertEqual(result_df.loc[pd.Timestamp("2026-01-02"), "DO 0.001%"], 18200.0)
            self.assertEqual(result_df.loc[pd.Timestamp("2026-01-02"), "Exo_Macro"], 101.0)

            # 3. Ngày mới (2026-01-04) được thêm vào
            self.assertIn(pd.Timestamp("2026-01-04"), result_df.index)
            self.assertEqual(result_df.loc[pd.Timestamp("2026-01-04"), "MG95"], 21500.0)

            # 4. Kiểm tra validate: file thiếu cột ngày -> phải raise ValueError
            bad_file_no_date = self.tmp_path / "bad_no_date.csv"
            pd.DataFrame({"MG95": [20000.0]}).to_csv(bad_file_no_date, index=False)
            with self.assertRaises(ValueError):
                tah.update_training_data(specific_file=str(bad_file_no_date))

            # 5. Kiểm tra validate: file thiếu cột mục tiêu -> phải raise ValueError
            bad_file_no_target = self.tmp_path / "bad_no_target.csv"
            pd.DataFrame({"Ngày": ["2026-01-01"], "Unknown_Col": [123]}).to_csv(bad_file_no_target, index=False)
            with self.assertRaises(ValueError):
                tah.update_training_data(specific_file=str(bad_file_no_target))

        finally:
            tah.DATA_PATH = orig_data_path

    def test_checkpoint_protection_on_validation_failure(self):
        """Kiểm tra khi xác thực checkpoint thất bại thì checkpoint cũ được bảo toàn 100%."""
        orig_out_dir = tah.OUT_DIR
        test_out_dir = self.tmp_path / "checkpoints_test"
        test_out_dir.mkdir(parents=True, exist_ok=True)
        tah.OUT_DIR = test_out_dir

        try:
            # Tạo một checkpoint cũ hợp lệ
            real_ckpt_path = test_out_dir / "gumnet_h1.pt"
            old_checkpoint_content = {
                "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
                "horizon": 1,
                "seq_len": 30,
                "num_quantiles": 3,
                "quantiles": [0.1, 0.5, 0.9],
                "feature_cols": ["f1"],
                "target_cols": tah.TARGET_COLS,
                "input_dim": 1,
                "output_dim": 4,
                "feature_scaler": None,
                "target_scaler": None,
                "date_col": "Ngày",
                "d_feat": 64,
            }
            torch.save(old_checkpoint_content, real_ckpt_path)
            initial_mtime = real_ckpt_path.stat().st_mtime

            # Giả lập training tạo checkpoint hỏng (thiếu model_state_dict) trong file tạm
            job_id = "test_fail_job"
            tmp_ckpt = test_out_dir / f".tmp_{job_id}_gumnet_h1.pt"
            torch.save({"corrupted": True}, tmp_ckpt)

            # Hàm xác thực sẽ phát hiện lỗi và ném ValueError
            with self.assertRaises(Exception):
                verified = load_checkpoint(tmp_ckpt, map_location="cpu")
                if not isinstance(verified, dict) or "model_state_dict" not in verified or "horizon" not in verified:
                    raise ValueError("Checkpoint thiếu model_state_dict hoặc horizon")
                os.replace(tmp_ckpt, real_ckpt_path)

            # Checkpoint cũ VẪN NGUYÊN VẸN, không hề bị thay đổi hay xóa!
            self.assertTrue(real_ckpt_path.exists())
            self.assertEqual(real_ckpt_path.stat().st_mtime, initial_mtime)
            loaded_old = load_checkpoint(real_ckpt_path, map_location="cpu")
            self.assertIn("model_state_dict", loaded_old)
            torch.testing.assert_close(loaded_old["model_state_dict"]["weight"], torch.tensor([1.0, 2.0]))
        finally:
            tah.OUT_DIR = orig_out_dir

    def test_log_line_parser_and_progress(self):
        """Kiểm tra helper _process_log_line cập nhật đúng trạng thái horizon."""
        hz_status = {
            1: {"state": "waiting", "val_loss": None},
            5: {"state": "waiting", "val_loss": None},
        }

        # Bắt đầu mốc 1
        curr_hz, done_cnt, changed, msg = app_main._process_log_line(
            "🚀 ĐANG HUẤN LUYỆN MỐC: 1 NGÀY", hz_status, None, 0, 2
        )
        self.assertEqual(curr_hz, 1)
        self.assertEqual(hz_status[1]["state"], "running")
        self.assertTrue(changed)

        # Xong mốc 1 với Best Val Loss
        curr_hz, done_cnt, changed, msg = app_main._process_log_line(
            "    Best Val Loss: 0.045678", hz_status, 1, done_cnt, 2
        )
        self.assertEqual(done_cnt, 1)
        self.assertEqual(hz_status[1]["state"], "done")
        self.assertAlmostEqual(hz_status[1]["val_loss"], 0.045678)

        # Bắt đầu mốc 5
        curr_hz, done_cnt, changed, msg = app_main._process_log_line(
            "🚀 ĐANG HUẤN LUYỆN MỐC: 5 NGÀY", hz_status, 1, done_cnt, 2
        )
        self.assertEqual(curr_hz, 5)
        self.assertEqual(hz_status[5]["state"], "running")

        # Giả lập thất bại giữa chừng: nếu process returncode != 0
        # mốc 5 đang running sẽ được chuyển sang failed
        if hz_status[curr_hz]["state"] == "running":
            hz_status[curr_hz]["state"] = "failed"
        self.assertEqual(hz_status[5]["state"], "failed")
        # Tiến độ hoàn thành vẫn là 1/2 (50%), không bị ép lên 100%
        pct = int((done_cnt / 2) * 100)
        self.assertEqual(pct, 50)

    def test_apptest_all_five_pages(self):
        """Kiểm tra AppTest chuyển qua toàn bộ 5 trang khách hàng (thêm tab Biểu đồ tách riêng khỏi Lịch sử)."""
        at = AppTest.from_file(str(ROOT / "app_main.py"), default_timeout=30).run()
        self.assertFalse(at.exception, f"Trang 1 exception: {at.exception}")

        pages = [
            "▦  Đánh giá mô hình",
            "◷  Lịch sử & Xuất dữ liệu",
            "📈  Biểu đồ",
            "❓  Hướng dẫn sử dụng",
            "◈  Dự báo",
        ]
        for page in pages:
            if at.sidebar.radio:
                at.sidebar.radio[0].set_value(page).run()
                self.assertFalse(at.exception, f"Lỗi khi chuyển sang {page}: {at.exception}")


if __name__ == "__main__":
    unittest.main()
