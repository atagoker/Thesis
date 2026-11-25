"""Responsive Qt process controller for one active local job at a time."""

from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from thesisui.data import ROOT, decode_event, make_session, write_job_status


class JobController(QObject):
    event = Signal(dict)
    log = Signal(str)
    busy_changed = Signal(bool)
    completed = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed_to_start)
        self.buffer = b""
        self.session = None
        self.action = ""
        self.busy = False
        self.had_error = False
        self.cancel_requested = False
        self.received_done = False

    def start(self, action, arguments=()):
        if self.busy:
            raise RuntimeError("A local task is already running.")
        self.action = action
        self.session = None if action == "health" else make_session(action)
        args = ["-u", str(ROOT / "scripts/desktop_worker.py"), action]
        if self.session:
            args.extend(["--session", str(self.session)])
        args.extend(str(a) for a in arguments)
        self.buffer = b""
        self.had_error = self.cancel_requested = self.received_done = False
        self.busy = True
        self.busy_changed.emit(True)
        self.log.emit(f"Starting {action}" + (f" · {self.session.name}" if self.session else ""))
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        self.process.start(str(python), args)

    def read_stdout(self):
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if line.strip():
                self.deliver(line.decode("utf-8", errors="replace"))

    def deliver(self, line):
        item = decode_event(line)
        if item["event"] == "done":
            self.received_done = True
        if item["event"] == "error":
            self.had_error = True
        if item["event"] in {"stage", "error", "cancelled", "log"}:
            self.log.emit(item.get("message", line))
        if item["event"] == "epoch":
            self.log.emit(f"Epoch {item['epoch']} complete · source loss {item['source_loss']:.4f} · "
                          f"accepted {item['acceptance']:.1%} · source-validation F1 {item['source_val_teacher_macro_f1']:.4f}")
        self.event.emit(item)

    def read_stderr(self):
        value = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if value:
            self.log.emit(value)

    def failed_to_start(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.had_error = True
            self.log.emit("Could not start the Python worker: " + self.process.errorString())
            self.record_failure("failed")
            self.busy = False
            self.busy_changed.emit(False)
            self.completed.emit(False, "Could not start Python worker")

    def record_failure(self, status):
        if self.session:
            try:
                write_job_status(self.session, self.action, status)
            except OSError as error:
                self.log.emit(f"Could not record session status: {error}")

    def finished(self, exit_code, exit_status):
        self.read_stdout()
        self.read_stderr()
        if self.buffer.strip():
            self.deliver(self.buffer.decode("utf-8", errors="replace"))
        self.buffer = b""
        success = (exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
                   and self.received_done and not self.had_error)
        message = "Complete" if success else ("Stopped by user" if self.cancel_requested else f"Task failed (exit {exit_code})")
        if not success:
            self.record_failure("cancelled" if self.cancel_requested else "failed")
        self.busy = False
        self.busy_changed.emit(False)
        self.log.emit(message)
        self.completed.emit(success, message)

    def cancel(self):
        if not self.busy or self.cancel_requested or self.received_done:
            return
        self.cancel_requested = True
        self.log.emit("Stop requested…")
        if self.session:
            try:
                (self.session / "cancel.flag").touch()
            except OSError as error:
                self.log.emit(f"Graceful stop unavailable: {error}")
                self.process.terminate()
        if self.action != "train":
            self.process.terminate()
        # If imports or a CUDA operation delay graceful cancellation, stop this job only.
        process_id = self.process.processId()
        QTimer.singleShot(5000, lambda: self.process.kill()
                          if self.busy and self.process.processId() == process_id else None)
