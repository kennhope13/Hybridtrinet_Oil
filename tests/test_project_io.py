import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from project_io import load_checkpoint, read_cache, save_upload, write_cache


class ProjectIOTests(unittest.TestCase):
    def test_upload_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            first = save_upload(folder, "../../existing.csv", b"first")
            second = save_upload(folder, "C:\\existing.csv", b"second")
            self.assertEqual(first.parent, Path(folder).resolve())
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"first")
            with self.assertRaises(ValueError):
                save_upload(folder, "model.pt", b"invalid")

    def test_cache_roundtrip_preserves_dates_and_numbers(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "price": [12.5]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.json"
            write_cache(path, "version1", frame)
            fingerprint, actual = read_cache(path)
            self.assertEqual(fingerprint, "version1")
            pd.testing.assert_frame_equal(frame, actual, check_dtype=False)
            path.write_text("broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_cache(path)

    def test_legacy_scaler_checkpoint(self):
        scaler = StandardScaler().fit(np.array([[1., 2.], [3., 4.]]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"scaler": scaler, "weights": torch.ones(2)}, path)
            loaded = load_checkpoint(path)
            np.testing.assert_allclose(loaded["scaler"].mean_, scaler.mean_)

    def test_unlisted_object_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"object": complex(1, 2)}, path)
            with self.assertRaises(Exception):
                load_checkpoint(path)

    def test_existing_checkpoints(self):
        paths = list((ROOT / "checkpoints_multi").glob("*.pt"))
        for path in paths:
            with self.subTest(checkpoint=path.name):
                self.assertIsInstance(load_checkpoint(path), dict)

    def test_mape_rejects_zero_actual(self):
        spec = importlib.util.spec_from_file_location("metrics", ROOT / "oil_forecast_research_new-main/src/metrics.py")
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        self.assertAlmostEqual(metrics.mape([100], [110]), 10)
        with self.assertRaises(ValueError):
            metrics.mape([0], [1])

    def test_app_starts(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app_main.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))


if __name__ == "__main__":
    unittest.main()
