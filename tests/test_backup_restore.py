"""Test cơ chế sao lưu/khôi phục checkpoint (mục 6 — chuẩn bị vận hành) thêm trong app_main.py.

Trích riêng đúng 3 hàm liên quan bằng AST rồi chạy trong 1 namespace TRỎ VÀO THƯ MỤC TẠM
(không phải ROOT/checkpoints_multi thật) — tuyệt đối không đụng tới checkpoint production khi
chạy test này.
"""
import ast
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "app_main.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

FUNC_NAMES = {"backup_checkpoints_before_training", "list_checkpoint_backups", "restore_checkpoint_backup"}


def _load_functions(env):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in FUNC_NAMES]
    assert len(nodes) == 3, f"Thiếu hàm, chỉ tìm thấy: {[n.name for n in nodes]}"
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "app_main.py", "exec"), env)


class BackupRestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ckpt_backup_test_")
        self.root = Path(self.tmpdir)
        (self.root / "checkpoints_multi").mkdir()
        (self.root / "checkpoints_backup").mkdir()
        self.env = {
            "ROOT": self.root,
            "CKPT_BACKUP_DIR": self.root / "checkpoints_backup",
            "CKPT_BACKUP_KEEP": 2,
        }
        _load_functions(self.env)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_ckpt(self, name, content=b"fake-weights"):
        p = self.root / "checkpoints_multi" / name
        p.write_bytes(content)
        return p

    def test_backup_copies_only_selected_checkpoints(self):
        self._write_ckpt("gumnet_h1.pt", b"h1-v1")
        self._write_ckpt("gumnet_h5.pt", b"h5-v1")  # không được chọn -> không sao lưu

        saved = self.env["backup_checkpoints_before_training"]("job-1", ["GUMNet"], [1])
        self.assertIn("gumnet_h1.pt", saved)
        self.assertNotIn("gumnet_h5.pt", saved)

        backup_dir = self.root / "checkpoints_backup" / "job-1"
        self.assertTrue((backup_dir / "gumnet_h1.pt").exists())
        self.assertFalse((backup_dir / "gumnet_h5.pt").exists())
        self.assertEqual((backup_dir / "gumnet_h1.pt").read_bytes(), b"h1-v1")

    def test_restore_brings_back_exact_old_content(self):
        ckpt = self._write_ckpt("gumnet_h1.pt", b"old-version")
        self.env["backup_checkpoints_before_training"]("job-1", ["GUMNet"], [1])

        # Mô phỏng huấn luyện mới ghi đè checkpoint (kết quả "tệ hơn")
        ckpt.write_bytes(b"new-version-bad")
        self.assertEqual(ckpt.read_bytes(), b"new-version-bad")

        restored = self.env["restore_checkpoint_backup"]("job-1")
        self.assertIn("gumnet_h1.pt", restored)
        self.assertEqual(ckpt.read_bytes(), b"old-version", "Phải khôi phục đúng nội dung cũ")

    def test_hybrid_metadata_backed_up_and_restored_together(self):
        self._write_ckpt("hybrid_h1.pt", b"hybrid-v1")
        meta_dir = self.root / "checkpoints_multi" / "hybrid_h1_meta"
        meta_dir.mkdir()
        (meta_dir / "feature_cols.json").write_text('{"job_id": "job-1"}', encoding="utf-8")
        (meta_dir / "x_mu.npy").write_bytes(b"mu-data")

        self.env["backup_checkpoints_before_training"]("job-1", ["HybridTriNet"], [1])
        backup_meta = self.root / "checkpoints_backup" / "job-1" / "hybrid_h1_meta"
        self.assertTrue((backup_meta / "feature_cols.json").exists())
        self.assertTrue((backup_meta / "x_mu.npy").exists())

        # Hỏng metadata thật rồi khôi phục lại
        (meta_dir / "feature_cols.json").write_text("CORRUPTED", encoding="utf-8")
        self.env["restore_checkpoint_backup"]("job-1")
        self.assertEqual((meta_dir / "feature_cols.json").read_text(encoding="utf-8"), '{"job_id": "job-1"}')

    def test_old_backups_beyond_keep_limit_are_pruned(self):
        self._write_ckpt("gumnet_h1.pt")
        for i in range(4):  # CKPT_BACKUP_KEEP=2 -> chỉ được giữ tối đa 2 sau khi tạo 4 job
            self.env["backup_checkpoints_before_training"](f"job-{i}", ["GUMNet"], [1])
        remaining = list((self.root / "checkpoints_backup").iterdir())
        self.assertLessEqual(len(remaining), 2, f"Phải tự dọn bớt, chỉ còn tối đa 2 bản, còn thấy: {[p.name for p in remaining]}")

    def test_restore_missing_job_id_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            self.env["restore_checkpoint_backup"]("khong-ton-tai")

    def test_list_backups_never_touches_real_production_root(self):
        # Đảm bảo hàm list không hard-code đường dẫn thật nào ngoài CKPT_BACKUP_DIR truyền vào
        result = self.env["list_checkpoint_backups"]()
        self.assertEqual(result, [])  # thư mục tạm đang rỗng lúc này


if __name__ == "__main__":
    unittest.main()
