"""Release acceptance checks for local history, portable reports and settings."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from thesisui.data import training_requirements, write_job_status
from thesisui.sessions import export_session, list_sessions, read_session

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
try:
    from PySide6.QtCore import QPoint, QSettings
    from thesisui.window import MainWindow, create_application
    HAS_QT = True
except ImportError:
    HAS_QT = False


def save_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def create_prediction(parent):
    path = parent / "20261125_123456_predict_test"
    path.mkdir(parents=True)
    save_json(path / "prediction.json", {"input_file": "K006.mat", "window_count": 2,
              "mean_faulty_score": .2, "dataset": "paderborn", "predicted_class": "healthy",
              "classes": ["healthy", "faulty"]})
    np.savez(path / "window_predictions.npz", probabilities=np.array([[.7, .3], [.9, .1]]))
    write_job_status(path, "predict", "complete")
    return path


def create_training(parent, status="complete"):
    path = parent / "20261125_123500_train_test"
    path.mkdir(parents=True)
    save_json(path / "completed.json", {"target": "N09_M07_F10", "seed": 7, "epochs": 10,
              "status": "complete", "target_labels_used": False, "updates": 730,
              "final": {"acceptance": .71, "source_val_teacher_macro_f1": .44}})
    save_json(path / "run_manifest.json", {"config": {"adaptation_epochs": 10}})
    (path / "history.csv").write_text("epoch,acceptance\n10,0.71\n", encoding="utf-8")
    (path / "teacher.pt").write_bytes(b"never deserialize in history or reporting")
    write_job_status(path, "train", status)
    return path


class SessionReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sessions = self.root / "sessions"
        self.export = self.root / "exports"

    def test_prediction_export_contains_scores_and_preserves_inputs(self):
        folder = create_prediction(self.sessions)
        before = (folder / "prediction.json").read_bytes()
        out = export_session(read_session(folder), self.export)
        with ZipFile(out) as z:
            self.assertEqual(set(z.namelist()), {"summary.txt", "prediction.json", "window_scores.csv"})
            self.assertIn("not calibrated", z.read("summary.txt").decode())
            rows = list(csv.DictReader(z.read("window_scores.csv").decode().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["fault_class_score"]), .3)
            self.assertEqual(before, z.read("prediction.json"))
        self.assertEqual(before, (folder / "prediction.json").read_bytes())

    def test_training_report_excludes_models_and_raw_data(self):
        folder = create_training(self.sessions)
        session = list_sessions(self.sessions)[0]
        self.assertEqual(session.teacher, folder / "teacher.pt")
        with ZipFile(export_session(session, self.export)) as z:
            self.assertEqual(set(z.namelist()), {"summary.txt", "completed.json", "run_manifest.json", "history.csv"})
            self.assertIn("Acceptance is not accuracy", z.read("summary.txt").decode())

    def test_non_committed_or_missing_files_cannot_publish_a_teacher(self):
        folder = create_training(self.sessions, "cancelled")
        for status in ("cancelled", "failed", "started"):
            write_job_status(folder, "train", status)
            session = read_session(folder)
            self.assertIsNone(session.teacher)
            with self.assertRaises(ValueError):
                export_session(session, self.export)
        write_job_status(folder, "train", "complete")
        (folder / "teacher.pt").unlink()
        self.assertEqual(read_session(folder).status, "incomplete")

    def test_malformed_json_and_bad_predictions_fail_safely(self):
        folder = create_prediction(self.sessions)
        model_report = folder / "prediction.json"
        original = model_report.read_bytes()
        model_report.write_text("{bad json", encoding="utf-8")
        self.assertEqual(read_session(folder).status, "incomplete")
        model_report.write_bytes(original)
        np.savez(folder / "window_predictions.npz", probabilities=np.array([[.5, .5]]))
        with self.assertRaises(ValueError):
            export_session(read_session(folder), self.export)
        self.assertEqual(list(self.export.iterdir()), [])

    def test_preflight_points_to_source_and_selected_target_files(self):
        missing = training_requirements("N09_M07_F10", 7, self.root)
        self.assertTrue(any("source_fit.npz" in p for p in missing))
        self.assertTrue(any("N09_M07_F10_unlabeled.npz" in p for p in missing))
        self.assertFalse(any("evaluator" in p or p.endswith("_labels.npz") for p in missing))


@unittest.skipUnless(HAS_QT, "Optional native Qt dependency is not installed")
class ReleaseWindowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_application()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.history = self.root / "sessions"
        self.ini = QSettings(str(self.root / "test.ini"), QSettings.Format.IniFormat)
        self.window = self.build_window()

    def build_window(self):
        window = MainWindow(auto_health=False, settings=self.ini, session_root=self.history, report_directory=self.root / "reports")
        window.gpu_timer.stop()
        self.addCleanup(window.close)
        return window

    def test_restarting_restores_preferences_and_selects_completed_teacher(self):
        run = create_training(self.history)
        self.window.training_page.epochs.setValue(9)
        self.window.signal_page.device.setCurrentIndex(1)
        self.window.close()
        restarted = self.build_window()
        self.assertEqual(restarted.training_page.epochs.value(), 9)
        self.assertEqual(restarted.signal_page.device.currentData(), "cpu")
        restarted.navigate(4)
        restarted.session_page.teacher_button.click()
        self.assertEqual(restarted.stack.currentIndex(), 1)
        self.assertEqual(restarted.signal_page.model.currentData(), str(run / "teacher.pt"))

    def test_missing_recording_does_not_launch_worker(self):
        self.window.signal_page.file.setText(str(self.root / "no.mat"))
        with patch.object(self.window.controller, "start") as start:
            self.window.signal_page.request("predict")
        start.assert_not_called()
        self.assertIn("existing .mat", self.window.signal_page.score.text())

    def test_full_score_chart_fits_compact_visible_page(self):
        self.window.resize(1024, 670)
        self.window.show()
        self.window.navigate(1)
        page = self.window.signal_page
        page.tabs.setCurrentWidget(page.window_scores)
        self.app.processEvents()
        self.assertLessEqual(self.window.width(), 1024)
        self.assertTrue(page.rect().contains(page.window_scores.mapTo(page, QPoint(0, 0))))
        self.assertTrue(page.rect().contains(page.window_scores.mapTo(page, page.window_scores.rect().bottomRight())))
        self.assertFalse(self.window.console.isVisible())
