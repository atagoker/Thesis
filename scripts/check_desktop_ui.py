"""Exercise the real native UI and local worker; save screenshots and checks."""

import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from thesisui.data import ROOT, read_json
from thesisui.window import MainWindow, create_application


def main():
    app = create_application()
    window = MainWindow(auto_health=False)
    window.resize(1460, 980)
    window.show()
    directory = ROOT / "outputs/desktop_ui_checks"
    directory.mkdir(parents=True, exist_ok=True)
    events = []
    window.controller.event.connect(events.append)

    def settle(milliseconds=150):
        app.processEvents()
        QTest.qWait(milliseconds)

    def wait_job(timeout=90000):
        started = time.monotonic()
        while window.controller.busy:
            if time.monotonic() - started > timeout / 1000:
                window.controller.cancel()
                raise TimeoutError("Native UI job timed out.")
            settle(30)
        settle(100)
        if window.controller.had_error:
            raise AssertionError("Worker reported an error: " + str([e for e in events if e['event'] == 'error']))

    def capture(name):
        settle(100)
        if not window.grab().save(str(directory / name)):
            raise AssertionError("Could not save native UI screenshot.")

    window.start_job("health", [])
    wait_job()
    assert any(e["event"] == "health" and e["cuda"] for e in events)
    capture("overview.png")
    window.navigate(1)
    window.signal_page.example(False)
    QTest.mouseClick(window.signal_page.predict, Qt.MouseButton.LeftButton)
    wait_job()
    healthy = [e for e in events if e["event"] == "result" and e.get("kind") == "predict"][-1]
    assert healthy["predicted_class"] == "healthy"
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
    window.navigate(2)
    window.training_page.epochs.setValue(10)
    saved = {"training": False}

    def live_capture(event):
        if event["event"] == "epoch" and event["epoch"] >= 3 and not saved["training"]:
            saved["training"] = True
            QTimer.singleShot(0, lambda: window.grab().save(str(directory / "training_live.png")))

    window.controller.event.connect(live_capture)
    QTest.mouseClick(window.training_page.start, Qt.MouseButton.LeftButton)
    wait_job()
    training_session = window.controller.session
    completed = read_json(training_session / "completed.json")
    assert completed["updates"] == 730
    assert completed["target_labels_used"] is False
    assert len(window.training_page.rows) == 10
    assert any(e["event"] == "batch" for e in events)
    for epoch in range(1, 11):
        steps = [e["step"] for e in events if e["event"] == "batch" and e["epoch"] == epoch]
        assert steps == [1, *range(5, 71, 5), 73], (epoch, steps)
    capture("training_complete.png")
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
            window.controller.cancel()

    window.controller.event.connect(request_stop)
    window.training_page.epochs.setValue(30)
    QTest.mouseClick(window.training_page.start, Qt.MouseButton.LeftButton)
    wait_job()
    cancelled_job = read_json(window.controller.session / "job.json")
    assert cancelled_job["status"] == "cancelled"
    assert not (window.controller.session / "completed.json").exists()
    window.navigate(3)
    capture("experiments.png")
    report = {"native_qt_widgets": True, "browser_or_server_used": False,
              "cuda_health_passed": True, "healthy_prediction": healthy["predicted_class"],
              "faulty_prediction": faulty["predicted_class"], "real_training_epochs": 10,
              "real_training_updates": 730, "live_batch_events": sum(e['event'] == 'batch' for e in events),
              "exact_teacher_tensors_match_frozen_run": exact, "source_sequence_match": True,
              "cancellation_passed": True, "screenshots": [p.name for p in directory.glob("*.png")],
              "training_session": str(training_session)}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    window.close()
    settle(50)


if __name__ == "__main__":
    main()
