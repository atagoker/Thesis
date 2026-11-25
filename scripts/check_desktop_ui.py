"""Exercise the real native UI and local worker; save screenshots and checks."""

import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
from zipfile import ZipFile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox
from thesisui.data import ROOT, read_json
from thesisui.window import MainWindow, create_application


def main():
    app = create_application()
    window = MainWindow(auto_health=False, settings=False)
    window.resize(1460, 980)
    window.show()
    directory = ROOT / "outputs/desktop_ui_checks"
    directory.mkdir(parents=True, exist_ok=True)
    events = []
    window.controller.event.connect(events.append)

    def settle(milliseconds=150):
        app.processEvents()
        QTest.qWait(milliseconds)

    def wait_job(timeout=90000, expect_error=False):
        started = time.monotonic()
        while window.controller.busy:
            if time.monotonic() - started > timeout / 1000:
                window.controller.cancel()
                raise TimeoutError("Native UI job timed out.")
            settle(30)
        settle(100)
        if window.controller.had_error != expect_error:
            raise AssertionError("Unexpected worker state: " + str([e for e in events if e['event'] == 'error']))

    exported_pngs = []

    def capture(name, settle_first=True):
        if settle_first:
            settle(100)
        previous = window.last_png_export
        QTest.mouseClick(window.save_png_button, Qt.MouseButton.LeftButton)
        path = window.last_png_export
        assert path is not None and path != previous and path.is_file()
        image = QImage(str(path))
        assert not image.isNull()
        assert (image.width(), image.height()) == (window.width() * 2, window.height() * 2)
        shutil.copy2(path, directory / name)
        exported_pngs.append(str(path))

    window.start_job("health", [])
    wait_job()
    assert any(e["event"] == "health" and e["cuda"] for e in events)
    capture("overview.png")
    window.demo_buttons[0].click()
    QTest.mouseClick(window.signal_page.predict, Qt.MouseButton.LeftButton)
    wait_job()
    healthy = [e for e in events if e["event"] == "result" and e.get("kind") == "predict"][-1]
    assert healthy["predicted_class"] == "healthy"
    assert healthy["device"] == "cuda"
    assert healthy["window_count"] == 31
    assert window.signal_page.info is not None
    window.signal_page.toggle_play()
    settle(150)
    window.signal_page.toggle_play()
    window.signal_page.example(True)
    QTest.mouseClick(window.signal_page.predict, Qt.MouseButton.LeftButton)
    wait_job()
    faulty = [e for e in events if e["event"] == "result" and e.get("kind") == "predict"][-1]
    assert faulty["predicted_class"] == "faulty"
    capture("prediction.png")
    window.signal_page.tabs.setCurrentIndex(0)
    capture("prediction_waveform.png")
    window.signal_page.tabs.setCurrentIndex(1)
    window.resize(1024, 670)
    settle()
    page = window.signal_page
    assert page.rect().contains(page.window_scores.mapTo(page, QPoint(0, 0)))
    assert page.rect().contains(page.window_scores.mapTo(page, page.window_scores.rect().bottomRight()))
    capture("prediction_compact.png")
    window.resize(1460, 980)
    if "--quick" in sys.argv:
        QTest.mouseClick(window.signal_page.inspect, Qt.MouseButton.LeftButton)
        wait_job()
        assert window.signal_page.decision.text() == "Signal ready"
        assert "model decision" in window.signal_page.score.text()
        window.signal_page.dataset.setCurrentIndex(1)
        window.signal_page.example(True)
        QTest.mouseClick(window.signal_page.predict, Qt.MouseButton.LeftButton)
        wait_job()
        cwru = [e for e in events if e["event"] == "result" and e.get("dataset") == "cwru"][-1]
        assert cwru["predicted_class"] == "outer_race"
        capture("cwru_prediction.png")
        window.signal_page.model.setCurrentIndex(0)
        assert window.signal_page.decision.text() == "Awaiting prediction"
        window.navigate(0)
        window.resize(1100, 760)
        capture("overview_compact.png")
        window.resize(1460, 980)
        window.navigate(3)
        capture("experiments.png")
        print("QUICK UI CHECK PASSED: Paderborn/CWRU prediction, inspect/reset states and compact layout.", flush=True)
        window.close()
        settle(50)
        return
    window.signal_page.open_result.click()
    assert window.stack.currentIndex() == 4
    assert str(window.session_page.selected().folder / "prediction.json") == faulty["report"]
    window.session_page.export_button.click()
    with ZipFile(window.last_report_export) as exported:
        assert "not calibrated" in exported.read("summary.txt").decode()
        prediction_scores = list(csv.DictReader(exported.read("window_scores.csv").decode().splitlines()))
        assert len(prediction_scores) == faulty["window_count"]
        assert abs(sum(float(r["fault_class_score"]) for r in prediction_scores) / len(prediction_scores)
                   - faulty["mean_faulty_score"]) < 1e-6
    window.navigate(1)
    window.signal_page.device.setCurrentIndex(1)
    window.signal_page.predict.click()
    wait_job()
    cpu = [e for e in events if e["event"] == "result" and e.get("kind") == "predict"][-1]
    assert cpu["device"] == "cpu" and cpu["predicted_class"] == "faulty" and cpu["window_count"] == 31
    checkpoint = window.signal_page.model.currentData()
    cwru_path = ROOT / "data/raw/cwru48/138.mat"
    window.signal_page.dataset.setCurrentIndex(1)
    window.signal_page.file.setText(str(cwru_path))
    wrong = window.signal_page.add_model("Wrong format test", checkpoint, "Test only")
    window.signal_page.model.setCurrentIndex(wrong)
    window.signal_page.predict.click()
    wait_job(expect_error=True)
    assert "four-class CWRU" in window.signal_page.score.text()
    assert read_json(window.controller.session / "job.json")["status"] == "failed"
    malformed = directory / "invalid_test_recording.mat"
    malformed.write_bytes(b"This is not a valid MATLAB file.")
    window.signal_page.file.setText(str(malformed))
    window.signal_page.inspect.click()
    wait_job(expect_error=True)
    assert "Cannot read this MATLAB recording" in window.signal_page.score.text()
    window.navigate(2)
    window.training_page.epochs.setValue(10)
    saved = {"training": False, "exported_while_busy": False}

    def capture_training():
        saved["exported_while_busy"] = window.controller.busy
        capture("training_live.png", settle_first=False)

    def live_capture(event):
        if event["event"] == "epoch" and event["epoch"] >= 3 and not saved["training"]:
            saved["training"] = True
            QTimer.singleShot(0, capture_training)

    window.controller.event.connect(live_capture)
    QTest.mouseClick(window.training_page.start, Qt.MouseButton.LeftButton)
    wait_job()
    training_session = window.controller.session
    completed = read_json(training_session / "completed.json")
    assert completed["updates"] == 730
    assert completed["target_labels_used"] is False
    assert len(window.training_page.rows) == 10
    assert any(e["event"] == "batch" for e in events)
    assert saved["exported_while_busy"], "PNG was not exported during real training."
    for epoch in range(1, 11):
        steps = [e["step"] for e in events if e["event"] == "batch" and e["epoch"] == epoch]
        assert steps == [1, *range(5, 71, 5), 73], (epoch, steps)
    capture("training_complete.png")
    window.training_page.open_result.click()
    assert window.session_page.selected().teacher == training_session / "teacher.pt"
    window.session_page.export_button.click()
    with ZipFile(window.last_report_export) as exported:
        assert "Acceptance is not accuracy" in exported.read("summary.txt").decode()
        assert len(list(csv.DictReader(exported.read("history.csv").decode().splitlines()))) == 10
    capture("saved_sessions.png")
    reopened = MainWindow(auto_health=False, settings=False)
    reopened.gpu_timer.stop()
    reopened.show_session(training_session)
    reopened.session_page.teacher_button.click()
    assert reopened.signal_page.model.currentData() == str(training_session / "teacher.pt")
    reopened.close()
    # Prove progress observation preserved the frozen algorithm's numerical result.
    import torch
    reference = torch.load(ROOT / "outputs/paderborn_commissioning_v1/seed_7/N09_M07_F10/teacher_student.pt", weights_only=True, map_location="cpu")
    observed = torch.load(training_session / "teacher.pt", weights_only=True, map_location="cpu")
    exact = all(torch.equal(value, observed["state_dict"][key]) for key, value in reference["state_dict"].items())
    assert exact, "Progress observation changed the teacher tensors."
    frozen = read_json(ROOT / "outputs/paderborn_commissioning_v1/training_complete.json")
    reference_control = next(r for r in frozen["runs"] if r["seed"] == 7 and r["condition"] == "N09_M07_F10")
    assert completed["source_sequence_sha256"] == reference_control["source_sequence_sha256"]
    # Cancellation is tested on another real run, not by pretending a finished task stopped.
    cancel_on_batch = {"enabled": True}

    def request_stop(event):
        if cancel_on_batch["enabled"] and event["event"] == "batch":
            cancel_on_batch["enabled"] = False
            with patch("thesisui.window.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
                window.close()

    window.controller.event.connect(request_stop)
    window.navigate(2)
    window.training_page.epochs.setValue(30)
    QTest.mouseClick(window.training_page.start, Qt.MouseButton.LeftButton)
    wait_job()
    cancelled_job = read_json(window.controller.session / "job.json")
    assert cancelled_job["status"] == "cancelled"
    assert not (window.controller.session / "completed.json").exists()
    assert window.training_page.last_checkpoint is None
    assert not window.training_page.open_result.isEnabled()
    assert not window.isVisible()
    window.show()  # Only to capture the final experiments view, after closing was verified.
    window.navigate(3)
    capture("experiments.png")
    report = {"native_qt_widgets": True, "browser_or_server_used": False,
              "cuda_health_passed": True, "healthy_prediction": healthy["predicted_class"],
              "faulty_prediction": faulty["predicted_class"], "real_training_epochs": 10,
              "real_training_updates": 730, "live_batch_events": sum(e['event'] == 'batch' for e in events),
              "exact_teacher_tensors_match_frozen_run": exact, "source_sequence_match": True,
              "one_click_png_exports": exported_pngs, "png_export_scale": 2,
              "png_exported_during_training": saved["exported_while_busy"],
              "cancellation_passed": True, "close_during_training_passed": True,
              "cpu_prediction_passed": True, "compact_window_scores_fit": True,
              "incompatible_checkpoint_rejected": True, "malformed_recording_rejected": True,
              "session_reopened_and_teacher_selected": True, "report_exports_verified": True,
              "screenshots": [p.name for p in directory.glob("*.png")],
              "training_session": str(training_session)}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    window.close()
    settle(50)


if __name__ == "__main__":
    main()
