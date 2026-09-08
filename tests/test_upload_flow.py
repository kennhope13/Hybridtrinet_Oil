import io
import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_pipeline as dp


class Uploaded(io.BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name

    def getvalue(self):
        return super().getvalue()


class UploadFlowTests(unittest.TestCase):
    def upload(self, name, frame):
        return Uploaded(name, frame.to_csv(index=False).encode("utf-8"))

    def test_partial_upload_keeps_good_file(self):
        good = self.upload("good.csv", pd.DataFrame({"Ngày": ["2026-01-02"], "MG95": [10]}))
        bad = self.upload("bad.csv", pd.DataFrame({"other": [1]}))
        result = dp.preview_uploaded_files([good, bad], pd.Timestamp("2026-01-01"))
        self.assertTrue(result[0]["is_valid"])
        self.assertFalse(result[1]["is_valid"])

    def test_file_missing_date_column_does_not_crash(self):
        bad = self.upload("bad.csv", pd.DataFrame({"MG95": [10]}))
        result = dp.preview_uploaded_files([bad], pd.Timestamp("2026-01-01"))[0]
        self.assertFalse(result["is_valid"])
        self.assertIn("Thiếu cột 'Ngày'", result["error"])

    def test_same_size_content_changes_signature(self):
        self.assertNotEqual(dp.compute_sha256(b"100"), dp.compute_sha256(b"200"))

    def test_old_data_does_not_start_update(self):
        old = self.upload("old.csv", pd.DataFrame({"Ngày": ["2025-01-01"], "MG95": [10]}))
        result = dp.preview_uploaded_files([old], pd.Timestamp("2026-01-01"))[0]
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["has_new_data"])


if __name__ == "__main__":
    unittest.main()
