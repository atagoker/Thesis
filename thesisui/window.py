"""Native Windows/Qt workspace. No HTTP server, HTML or browser components."""

from datetime import datetime
from pathlib import Path
import shutil
import time

from PySide6.QtCore import QProcess, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QGridLayout, QHBoxLayout,
                             QMainWindow, QMessageBox, QPlainTextEdit, QScrollArea, QSizePolicy,
                             QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from thesisui import theme as t
from thesisui.branding import APP_NAME
from thesisui.data import ROOT, dataset_summary, summary
from thesisui.jobs import JobController
from thesisui.plots import Plot
from thesisui.signal_page import SignalPage
from thesisui.study_pages import ExperimentsPage, TrainingPage
from thesisui.widgets import Metric, WaveMark, button, label, panel

PAGES = [("Overview", "Your diagnostic workspace", "Models, data and the latest measured findings."),
         ("Signal & prediction", "Inspect a bearing recording", "View real vibration signals and run your local PyTorch models."),
         ("Live commissioning", "Watch the model learn", "Real batch progress, pseudo-label acceptance and training curves."),
         ("Experiments", "Compare the measured results", "Explore all four methods on the same held-out bearings.")]


class MainWindow(QMainWindow):
    def __init__(self, auto_health=True):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Bearing Diagnosis · Native Desktop")
        self.setMinimumSize(1024, 680)
        self.last_update = None
        self.started = None
        self.active_action = ""
        self.latest_session = None
        self.closing_after_stop = False
        self.event_count = 0
        self.controller = JobController(self)
        self.controller.event.connect(self.handle_event)
        self.controller.log.connect(self.append_log)
        self.controller.busy_changed.connect(self.set_busy)
        self.controller.completed.connect(self.job_finished)
        root = QWidget()
        root.setObjectName("Root")
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        self.setCentralWidget(root)
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(208)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 28, 18, 20)
        side.setSpacing(9)
        brand = QHBoxLayout()
        mark = WaveMark()
        mark.setFixedSize(44, 44)
        brand.addWidget(mark)
        brand_text = label(APP_NAME)
        brand_text.setStyleSheet(f"font-size:{24 if len(APP_NAME) <= 8 else 20}px;font-weight:600;")
        brand_text.setToolTip(APP_NAME)
        brand.addWidget(brand_text)
        brand.addStretch()
        side.addLayout(brand)
        side.addWidget(label("BEARING DIAGNOSIS", "Eyebrow"))
        side.addSpacing(26)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav = []
        for i, (name, _, _) in enumerate(PAGES):
            item = button(name)
            item.setObjectName("Nav")
            item.setCheckable(True)
            item.clicked.connect(lambda checked=False, index=i: self.navigate(index))
            self.nav_group.addButton(item, i)
            side.addWidget(item)
            self.nav.append(item)
        side.addStretch()
        side.addWidget(label("COMPUTE DEVICE", "MetricLabel"))
        self.device_label = label("Checking PyTorch…", "Subtitle", True)
        side.addWidget(self.device_label)
        self.gpu_label = label("Waiting for GPU telemetry", "Subtitle", True)
        self.gpu_label.setStyleSheet(f"font-size:11px;color:{t.CYAN};")
        side.addWidget(self.gpu_label)
        side.addSpacing(16)
        self.session_button = button("Open session folder")
        self.session_button.clicked.connect(self.open_session)
        side.addWidget(self.session_button)
        side.addSpacing(13)
        side.addWidget(label("NATIVE DESKTOP", "Eyebrow"))
        side.addWidget(label("Runs locally on your laptop\nPython · PyTorch · Qt", "Subtitle", True))
        main.addWidget(sidebar)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(26, 22, 26, 15)
        content_layout.setSpacing(14)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(5)
        heading.addWidget(label("THESIS / COMMISSIONING LAB", "Eyebrow"))
        self.page_title = label("", "Title")
        self.page_subtitle = label("", "Subtitle", True)
        heading.addWidget(self.page_title)
        heading.addWidget(self.page_subtitle)
        header.addLayout(heading, 1)
        status = QVBoxLayout()
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.badge = label("LOCAL · READY", "Badge")
        self.live_clock = label("No active job", "Subtitle")
        self.live_clock.setStyleSheet(f"font-size:11px;color:{t.MUTED};")
        status.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignRight)
        status.addWidget(self.live_clock, 0, Qt.AlignmentFlag.AlignRight)
        header.addLayout(status)
        self.stop = button("Stop task")
        self.stop.setObjectName("Danger")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.controller.cancel)
        header.addWidget(self.stop)
        content_layout.addLayout(header)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.stack = QStackedWidget()
        self.signal_page = SignalPage()
        self.training_page = TrainingPage()
        self.experiments_page = ExperimentsPage()
        self.overview = self.build_overview()
        for page in [self.overview, self.signal_page, self.training_page, self.experiments_page]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.splitter.addWidget(self.stack)
        activity, activity_layout = panel()
        activity_layout.setContentsMargins(15, 11, 15, 11)
        activity_layout.setSpacing(7)
        activity_title = QHBoxLayout()
        activity_title.addWidget(label("Activity stream", "SectionTitle"))
        self.stage_label = label("Ready", "Subtitle")
        activity_title.addStretch()
        activity_title.addWidget(self.stage_label)
        activity_layout.addLayout(activity_title)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.document().setMaximumBlockCount(500)
        self.console.setMinimumHeight(66)
        activity_layout.addWidget(self.console)
        self.splitter.addWidget(activity)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([650, 135])
        content_layout.addWidget(self.splitter, 1)
        main.addWidget(content, 1)
        self.statusBar().showMessage(str(ROOT))
        self.signal_page.run_requested.connect(self.start_job)
        self.training_page.run_requested.connect(self.start_job)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.gpu_process = QProcess(self)
        self.gpu_process.finished.connect(self.gpu_ready)
        self.gpu_timer = QTimer(self)
        self.gpu_timer.setInterval(3000)
        self.gpu_timer.timeout.connect(self.poll_gpu)
        self.gpu_timer.start()
        self.navigate(0)
        self.append_log("Desktop workspace ready. Charts show measured data; active tasks stream real worker events.")
        self.set_icon()
        if auto_health:
            QTimer.singleShot(300, lambda: self.start_job("health", []))

    def set_icon(self):
        mark = WaveMark()
        mark.resize(44, 44)
        pixmap = QPixmap(44, 44)
        pixmap.fill(Qt.GlobalColor.transparent)
        mark.render(pixmap)
        self.setWindowIcon(QIcon(pixmap))

    def build_overview(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.run_state = label("Ready for your next recording", "SectionTitle")
        welcome, body = panel()
        top = QHBoxLayout()
        text = QVBoxLayout()
        text.addWidget(self.run_state)
        text.addWidget(label("Inspect a signal, run a model or watch teacher–student commissioning on your GPU.", "Subtitle", True))
        top.addLayout(text, 1)
        open_signal = button("Open a recording", True)
        open_signal.clicked.connect(lambda: self.navigate(1))
        open_training = button("Start a live run")
        open_training.clicked.connect(lambda: self.navigate(2))
        top.addWidget(open_signal)
        top.addWidget(open_training)
        body.addLayout(top)
        layout.addWidget(welcome)
        metrics = QHBoxLayout()
        prepared = dataset_summary()
        evaluation = summary()
        models = evaluation.get("overall_target_macro_f1", {})
        self.device_metric = Metric("Compute", "Checking", "Local PyTorch environment", t.CYAN)
        data = Metric("Paderborn data", f"{prepared.get('raw_recordings_audit_attempted', 0):,}", "Recordings audited · 32 bearings")
        teacher = Metric("Teacher–student", f"{models['teacher_student']:.2%}" if "teacher_student" in models else "—",
                         "Saved mean recording macro F1", t.PURPLE)
        control = Metric("Source EMA", f"{models['source_ema']:.2%}" if "source_ema" in models else "—",
                         "Matched control · saved result", t.GREEN)
        for widget in [self.device_metric, data, teacher, control]:
            metrics.addWidget(widget, 1)
        layout.addLayout(metrics)
        row = QHBoxLayout()
        chart_panel, chart_layout = panel("Measured experiment results", "Saved Paderborn study · three targets × three seeds")
        comparison = Plot("", "", "Recording macro F1 (%)", 255)
        comparison.comparison(evaluation)
        chart_layout.addWidget(comparison)
        row.addWidget(chart_panel, 3)
        flow, flow_layout = panel("How commissioning works")
        for title, subtitle, color in [
            ("01   Labeled source", "Learn healthy and faulty vibration patterns.", t.CYAN),
            ("02   Unlabeled target", "Teacher selects confident provisional labels.", t.PURPLE),
            ("03   Student + EMA teacher", "Student learns; teacher follows slowly.", t.GREEN),
        ]:
            title_label = label(title, "SectionTitle")
            title_label.setStyleSheet(f"font-size:13px;color:{color};")
            flow_layout.addWidget(title_label)
            flow_layout.addWidget(label(subtitle, "Subtitle", True))
        flow_layout.addStretch()
        flow_layout.addWidget(label("Current finding: target adaptation has not outperformed source EMA on average.", "Subtitle", True))
        row.addWidget(flow, 2)
        layout.addLayout(row, 1)
        layout.addWidget(label("Research prototype · scores are model outputs · a new desktop run does not replace the frozen thesis study.", "Subtitle", True))
        return page

    def navigate(self, index):
        self.stack.setCurrentIndex(index)
        self.nav[index].setChecked(True)
        _, title, subtitle = PAGES[index]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)

    def start_job(self, action, arguments):
        if self.controller.busy:
            self.append_log("Finish or stop the current task before starting another.")
            return
        if action in {"inspect", "predict"}:
            arg_map = dict(zip(arguments[::2], arguments[1::2]))
            if not Path(arg_map.get("--file", "")).is_file():
                self.append_log("Choose an existing .mat recording before starting.")
                self.signal_page.handle({"event": "error", "message": "The selected recording does not exist."})
                return
            if action == "predict" and not Path(arg_map.get("--checkpoint", "")).is_file():
                self.append_log("Choose an existing model checkpoint.")
                self.signal_page.handle({"event": "error", "message": "The model checkpoint is missing."})
                return
        self.active_action = action
        self.started = time.monotonic()
        self.last_update = self.started
        self.controller.start(action, arguments)
        self.latest_session = self.controller.session or self.latest_session

    def set_busy(self, busy):
        self.stop.setEnabled(busy)
        self.signal_page.set_busy(busy)
        self.training_page.set_busy(busy)
        self.badge.setText("LOCAL · WORKING" if busy else "LOCAL · READY")
        self.badge.setStyleSheet(f"background:{t.ACTIVE_BG if busy else t.SUCCESS_BG};color:{t.CYAN if busy else t.GREEN};")
        self.run_state.setText(f"Active task: {self.active_action}" if busy else "Ready for your next recording")

    def handle_event(self, item):
        self.last_update = time.monotonic()
        event = item["event"]
        if event == "health":
            self.device_label.setText(item["device"])
            self.device_metric.update_value("CUDA" if item["cuda"] else "CPU", f"PyTorch {item['torch']}")
            self.append_log(f"Compute ready · {item['device']} · Python {item['python']} · PyTorch {item['torch']}")
            self.poll_gpu()
        elif event == "stage":
            self.stage_label.setText(item["message"])
        elif event == "batch":
            self.stage_label.setText(f"Epoch {item['epoch']}/{item['total_epochs']} · batch {item['step']}/{item['steps']}")
        elif event == "prediction_progress":
            self.stage_label.setText(f"Predicting · {item['completed']} / {item['total']} windows")
        elif event == "result":
            self.append_log(item.get("message", "Result saved: " + item.get("report", "")))
            self.latest_session = self.controller.session
            if item.get("kind") == "train" and self.signal_page.dataset.currentData() == "paderborn":
                self.signal_page.model.addItem("Latest interactive teacher", item["checkpoint"])
                self.signal_page.model.setCurrentIndex(self.signal_page.model.count() - 1)
        elif event == "error":
            self.append_log(item.get("detail", item["message"]))
        if self.active_action in {"inspect", "predict"}:
            self.signal_page.handle(item)
        if self.active_action == "train":
            self.training_page.handle(item)

    def append_log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{stamp}]  {message}")
        bar = self.console.verticalScrollBar()
        bar.setValue(bar.maximum())

    def job_finished(self, success, message):
        self.stage_label.setText(message)
        if not success and not self.controller.cancel_requested:
            self.badge.setText("LOCAL · ERROR")
            self.badge.setStyleSheet(f"background:{t.ERROR_BG};color:{t.RED};")
        self.tick()
        if self.closing_after_stop:
            QTimer.singleShot(0, self.close)

    def tick(self):
        if self.controller.busy and self.started is not None:
            elapsed = int(time.monotonic() - self.started)
            age = time.monotonic() - self.last_update if self.last_update else 0
            self.live_clock.setText(f"Elapsed {elapsed//60:02d}:{elapsed%60:02d} · last event {age:.0f}s ago")
        elif self.started is not None:
            self.live_clock.setText("Idle · " + datetime.now().strftime("%H:%M:%S"))
        else:
            self.live_clock.setText(datetime.now().strftime("%d %b %Y · %H:%M:%S"))

    def poll_gpu(self):
        executable = shutil.which("nvidia-smi")
        if not executable:
            self.gpu_label.setText("GPU telemetry unavailable")
            return
        if self.gpu_process.state() == QProcess.ProcessState.NotRunning:
            self.gpu_process.start(executable, ["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])

    def gpu_ready(self, exit_code, status):
        text = bytes(self.gpu_process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        if exit_code == 0 and text:
            try:
                util, used, total = [int(v.strip()) for v in text.splitlines()[0].split(",")]
                self.gpu_label.setText(f"GPU {util}% · VRAM {used/1024:.1f} / {total/1024:.1f} GB\nUpdated {datetime.now():%H:%M:%S}")
            except ValueError:
                self.gpu_label.setText("GPU telemetry unavailable")

    def open_session(self):
        path = self.latest_session or ROOT / "outputs/desktop_sessions"
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event):
        if self.controller.busy:
            answer = QMessageBox.question(self, "A task is still running", f"Stop the active task and close {APP_NAME}?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                self.closing_after_stop = True
                self.controller.cancel()
            event.ignore()
            return
        self.gpu_timer.stop()
        if self.gpu_process.state() != QProcess.ProcessState.NotRunning:
            self.gpu_process.kill()
            self.gpu_process.waitForFinished(1000)
        event.accept()


def create_application():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Ata Goker Thesis")
    app.setStyle("Fusion")
    # The offscreen platform used for automated screenshots does not discover
    # Windows fonts by itself. Register the same native fonts for both paths.
    for filename in ["segoeui.ttf", "segoeuib.ttf", "seguisb.ttf", "consola.ttf"]:
        font_path = Path("C:/Windows/Fonts") / filename
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    palette = QPalette()
    for role, color in [(QPalette.ColorRole.Window, t.BG), (QPalette.ColorRole.WindowText, t.TEXT),
                        (QPalette.ColorRole.Base, t.PANEL_ALT), (QPalette.ColorRole.AlternateBase, t.PANEL),
                        (QPalette.ColorRole.Text, t.TEXT), (QPalette.ColorRole.Button, t.PANEL),
                        (QPalette.ColorRole.ButtonText, t.TEXT), (QPalette.ColorRole.Highlight, "#173f60"),
                        (QPalette.ColorRole.HighlightedText, t.TEXT), (QPalette.ColorRole.ToolTipBase, t.PANEL),
                        (QPalette.ColorRole.ToolTipText, t.TEXT)]:
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(t.STYLE)
    return app
