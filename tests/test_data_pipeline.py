import io
import tempfile
import unittest
from pathlib import Path
import sys

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


class DataPipelineTests(unittest.TestCase):
    def csv_file(self, name, frame):
        return Uploaded(name, frame.to_csv(index=False).encode("utf-8"))

    def test_date_alias_does_not_match_arbitrary_ng_substring(self):
        frame = pd.DataFrame({"Xang": [1], "MG95": [2]})
        _, error, _ = dp.parse_and_validate_dataframe(self.csv_file("bad.csv", frame), "bad.csv")
        self.assertIn("Thiếu cột 'Ngày'", error)

    def test_modified_rows_are_detected_with_existing_records(self):
        existing = pd.DataFrame({"Ngày": pd.to_datetime(["2026-01-01"]), "MG95": [10.0]})
        incoming = pd.DataFrame({"Ngày": ["2026-01-01", "2026-01-02"], "MG95": [12.0, 13.0]})
        preview = dp.preview_uploaded_files(
            [self.csv_file("new.csv", incoming)], pd.Timestamp("2026-01-01"), existing
        )[0]
        self.assertEqual(preview["modified_rows_count"], 1)
        self.assertEqual(preview["new_rows_count"], 1)

    def test_modified_details_reports_old_and_new_values(self):
        existing = pd.DataFrame({"Ngày": pd.to_datetime(["2026-01-01"]), "MG95": [10.0]})
        incoming = pd.DataFrame({"Ngày": ["2026-01-01"], "MG95": [12.0]})
        preview = dp.preview_uploaded_files(
            [self.csv_file("new.csv", incoming)], pd.Timestamp("2026-01-01"), existing
        )[0]
        self.assertEqual(len(preview["modified_details"]), 1)
        detail = preview["modified_details"][0]
        self.assertEqual(detail["column"], "MG95")
        self.assertEqual(detail["old_value"], 10.0)
        self.assertEqual(detail["new_value"], 12.0)

    def test_commit_ignores_modified_date_without_confirmation(self):
        existing = pd.DataFrame({"Ngày": pd.to_datetime(["2026-01-01"]), "MG95": [10.0]})
        incoming = pd.DataFrame({"Ngày": ["2026-01-01"], "MG95": [12.0]})
        upload = self.csv_file("mod.csv", incoming)
        previews = dp.preview_uploaded_files([upload], pd.Timestamp("2026-01-01"), existing)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            res = dp.commit_valid_files(previews, {"mod.csv": upload}, out)
            self.assertFalse(res["success"])
            self.assertEqual(res["no_new_data_files"][0]["filename"], "mod.csv")
            self.assertEqual(len(list(out.glob("*.csv"))), 0)

    def test_commit_overwrites_modified_date_when_confirmed(self):
        existing = pd.DataFrame({"Ngày": pd.to_datetime(["2026-01-01"]), "MG95": [10.0]})
        incoming = pd.DataFrame({"Ngày": ["2026-01-01"], "MG95": [12.0]})
        upload = self.csv_file("mod.csv", incoming)
        previews = dp.preview_uploaded_files([upload], pd.Timestamp("2026-01-01"), existing)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            res = dp.commit_valid_files(
                previews, {"mod.csv": upload}, out,
                overwrite_confirmed_files={previews[0]["sha256"]},
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["total_overwritten_rows"], 1)
            files = list(out.glob("*.csv"))
            self.assertEqual(len(files), 1)
            saved = pd.read_csv(files[0])
            self.assertEqual(len(saved), 1)
            self.assertEqual(float(saved.iloc[0]["MG95"]), 12.0)

    # Lỗi thật đã sửa: trước đây xác nhận ghi đè khớp theo TÊN FILE — 2 file trùng tên nhưng
    # khác nội dung, xác nhận đúng 1 file sẽ vô tình áp dụng luôn cho file kia (chưa xác nhận).
    # Giờ khớp theo sha256 (nội dung thật) nên phải phân biệt đúng, không lẫn lộn.
    def test_confirming_one_file_does_not_overwrite_another_file_with_same_name(self):
        existing = pd.DataFrame({"Ngày": pd.to_datetime(["2026-01-01"]), "MG95": [10.0]})
        incoming_a = pd.DataFrame({"Ngày": ["2026-01-01"], "MG95": [12.0]})
        incoming_b = pd.DataFrame({"Ngày": ["2026-01-01"], "MG95": [99.0]})
        upload_a = self.csv_file("dup.csv", incoming_a)
        upload_b = self.csv_file("dup.csv", incoming_b)
        previews = dp.preview_uploaded_files([upload_a, upload_b], pd.Timestamp("2026-01-01"), existing)
        self.assertNotEqual(previews[0]["sha256"], previews[1]["sha256"])

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            # Chỉ xác nhận ghi đè cho file A (theo sha256 của A), KHÔNG xác nhận file B.
            res = dp.commit_valid_files(
                previews,
                {"dup.csv": upload_b},  # raw_files_map chỉ giữ được 1 object cho 1 tên file
                out,
                overwrite_confirmed_files={previews[0]["sha256"]},
            )
            # File A (đã xác nhận) được ghi đè đúng 1 dòng của NÓ. File B (chưa xác nhận) phải
            # bị coi là "không có dữ liệu mới", không lây xác nhận từ file A dù trùng tên.
            self.assertEqual(res["total_overwritten_rows"], 1)
            self.assertEqual(len(res["saved_files"]), 1)
            self.assertTrue(any(f["filename"] == "dup.csv" for f in res["no_new_data_files"]))

    def test_commit_writes_only_validated_new_rows_and_unique_names(self):
        incoming = pd.DataFrame({"Ngày": ["2026-01-01", "2026-01-02"], "MG95": [12.0, 13.0]})
        upload = self.csv_file("same.csv", incoming)
        previews = dp.preview_uploaded_files([upload], pd.Timestamp("2026-01-01"))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            first = dp.commit_valid_files(previews, {"same.csv": upload}, out)
            second = dp.commit_valid_files(previews, {"same.csv": upload}, out)
            self.assertTrue(first["success"] and second["success"])
            files = list(out.glob("*.csv"))
            self.assertEqual(len(files), 2)
            for path in files:
                saved = pd.read_csv(path)
                self.assertEqual(len(saved), 1)
                self.assertEqual(str(saved.iloc[0]["Ngày"])[:10], "2026-01-02")


if __name__ == "__main__":
    unittest.main()
