"""Native Windows/Qt workspace. No HTTP server, HTML or browser components."""

from datetime import datetime
from pathlib import Path
import shutil
import time

from PySide6.QtCore import QProcess, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QGridLayout, QHBoxLayout,
                             QMainWindow, QMessageBox, QPlainTextEdit, QScrollArea, QSizePolicy,
                             QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from thesisui import theme as t
from thesisui.branding import APP_NAME, APP_VERSION
from thesisui.data import ROOT, dataset_summary, summary, training_requirements
from thesisui.exporting import save_widget_png, snapshot_path
from thesisui.jobs import JobController
from thesisui.plots import Plot
from thesisui.session_page import SessionPage
from thesisui.sessions import SESSION_ROOT, EXPORT_ROOT, export_session, read_session
from thesisui.signal_page import SignalPage
from thesisui.study_pages import ExperimentsPage, TrainingPage
from thesisui.widgets import Metric, WaveMark, button, label, panel

PAGES = [("Overview", "Your diagnostic workspace", "Models, data and the latest measured findings."),
         ("Signal & prediction", "Inspect a bearing recording", "View real vibration signals and run your local PyTorch models."),
         ("Live commissioning", "Watch the model learn", "Real batch progress, pseudo-label acceptance and training curves."),
         ("Experiments", "Compare the measured results", "Explore all four methods on the same held-out bearings."),
         ("Saved sessions", "Reopen your results", "Review past runs, reuse a teacher model, or export a report for sharing.")]


class MainWindow(QMainWindow):
    def __init__(self, auto_health=True, export_directory=None, settings=None,
                 session_root=SESSION_ROOT, report_directory=EXPORT_ROOT):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} — Bearing Diagnosis")
        self.setMinimumSize(940, 600)
        self.settings = QSettings() if settings is None else settings
        self.report_directory = Path(report_directory)
        self.session_root = Path(session_root)
        self.last_update = None
        self.started = None
        self.active_action = ""
        self.latest_session = None
        self.latest_signal_session = None
        self.latest_training_session = None
        self.last_report_export = None
        self.closing_after_stop = False
        self.event_count = 0
        self.png_export_directory = Path(export_directory) if export_directory is not None else ROOT / "outputs/presentation_exports"
        self.last_png_export = None
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
        side.setContentsMargins(18, 22, 18, 16)
        side.setSpacing(7)
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
        side.addSpacing(14)
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
        side.addSpacing(6)
        self.session_button = button("Open session folder")
        self.session_button.clicked.connect(self.open_session)
        side.addWidget(self.session_button)
        self.help_button = button("Demo & help")
        self.help_button.clicked.connect(self.show_help)
        side.addWidget(self.help_button)
        side.addWidget(label(f"NATIVE DESKTOP / {APP_VERSION}", "Eyebrow"))
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
        self.signal_page = SignalPage(self.session_root)
        self.training_page = TrainingPage()
        self.experiments_page = ExperimentsPage()
        self.session_page = SessionPage(self.session_root)
        self.overview = self.build_overview()
        for page in [self.overview, self.signal_page, self.training_page, self.experiments_page, self.session_page]:
            if page is self.signal_page:
                self.stack.addWidget(page)
                continue
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.splitter.addWidget(self.stack)
        self.activity, activity_layout = panel()
        activity_layout.setContentsMargins(15, 11, 15, 11)
        activity_layout.setSpacing(7)
        activity_title = QHBoxLayout()
        activity_title.addWidget(label("Activity stream", "SectionTitle"))
        self.stage_label = label("Ready", "Subtitle", True)
        activity_title.addStretch()
        activity_title.addWidget(self.stage_label, 2)
        self.activity_button = button("Show log")
        self.activity_button.setCheckable(True)
        self.activity_button.toggled.connect(self.toggle_activity)
        activity_title.addWidget(self.activity_button)
        activity_layout.addLayout(activity_title)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.document().setMaximumBlockCount(500)
        self.console.setMinimumHeight(66)
        activity_layout.addWidget(self.console)
        self.splitter.addWidget(self.activity)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.toggle_activity(False)
        content_layout.addWidget(self.splitter, 1)
        main.addWidget(content, 1)
        self.statusBar().showMessage(str(ROOT))
        self.save_png_button = button("Save PNG", True)
        self.save_png_button.setToolTip("Save this app view at 2× resolution, including during training. Ctrl+Shift+S")
        self.save_png_button.setShortcut("Ctrl+Shift+S")
        self.save_png_button.clicked.connect(self.save_current_png)
        self.open_png_folder_button = button("Open PNG folder")
        self.open_png_folder_button.setToolTip(str(self.png_export_directory))
        self.open_png_folder_button.clicked.connect(self.open_png_folder)
        self.statusBar().addPermanentWidget(self.save_png_button)
        self.statusBar().addPermanentWidget(self.open_png_folder_button)
        self.signal_page.run_requested.connect(self.start_job)
        self.training_page.run_requested.connect(self.start_job)
        self.signal_page.results_requested.connect(lambda: self.show_session(self.latest_signal_session))
        self.training_page.results_requested.connect(lambda: self.show_session(self.latest_training_session))
        self.session_page.open_requested.connect(lambda session: self.open_directory(session.folder) if session else None)
        self.session_page.export_requested.connect(self.export_report)
        self.session_page.teacher_requested.connect(self.reuse_teacher)
        self.session_page.exports_requested.connect(lambda: self.open_directory(self.report_directory))
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
        self.restore_preferences()
        self.append_log("Desktop workspace ready. Charts show measured data; active tasks stream real worker events.")
        self.set_icon()
        if auto_health:
            QTimer.singleShot(300, lambda: None if self.controller.busy else self.start_job("health", []))

    def toggle_activity(self, expanded):
        self.console.setVisible(expanded)
        self.activity_button.setText("Hide log" if expanded else "Show log")
        self.activity.setMaximumHeight(320 if expanded else 76)
        if expanded:
            self.splitter.setSizes([max(300, self.splitter.height() - 160), 160])

    def show_help(self):
        QMessageBox.information(self, f"{APP_NAME} · Demo & help",
            "1. On Overview select Healthy demo, then Run prediction.\n"
            "2. Select Faulty example and run again with the same model. Switch between Waveform and Window scores.\n"
            "3. In Live commissioning choose reduced speed, seed 7 and 10 epochs. Start and watch real training events.\n"
            "4. In Saved sessions reuse the teacher or export its report. Compare the saved methods on Experiments.\n\n"
            "Save PNG captures the visible app screen. Scores are not calibrated failure risks. "
            "The demo requires the project's local examples and prepared checkpoints. See reports/DESKTOP_UI.md for setup.")

    def preference_widgets(self):
        p, train = self.signal_page, self.training_page
        return {"prediction/dataset": p.dataset, "prediction/condition": p.target, "prediction/seed": p.seed,
                "prediction/device": p.device, "training/condition": train.target, "training/seed": train.seed}

    def restore_preferences(self):
        if self.settings is False:
            return
        for key, widget in self.preference_widgets().items():
            value = self.settings.value(key)
            index = next((i for i in range(widget.count()) if str(widget.itemData(i)) == str(value)), -1)
            if index >= 0:
                widget.setCurrentIndex(index)
        p = self.signal_page
        p.last_recording_folder = self.settings.value("folders/recording", p.last_recording_folder, type=str)
        p.last_model_folder = self.settings.value("folders/model", p.last_model_folder, type=str)
        p.channel.setText(self.settings.value("prediction/channel", p.channel.text(), type=str))
        for key, widget in [("prediction/hz", p.sampling), ("training/epochs", self.training_page.epochs)]:
            try:
                widget.setValue(int(self.settings.value(key, widget.value())))
            except (ValueError, TypeError):
                pass
        recording = self.settings.value("prediction/recording", "", type=str)
        p.file.setText(recording if Path(recording).is_file() else "")
        if not p.file.text() and p.binary:
            p.example(False)
        path = self.settings.value("prediction/model", "", type=str)
        index = p.model.findData(path)
        if index >= 0:
            p.model.setCurrentIndex(index)

    def save_preferences(self):
        if self.settings is False:
            return
        p = self.signal_page
        for key, widget in self.preference_widgets().items():
            self.settings.setValue(key, widget.currentData())
        for key, value in {"folders/recording": p.last_recording_folder, "folders/model": p.last_model_folder,
                           "prediction/recording": p.file.text(), "prediction/channel": p.channel.text(),
                           "prediction/hz": p.sampling.value(), "prediction/model": p.model.currentData() or "",
                           "training/epochs": self.training_page.epochs.value()}.items():
            self.settings.setValue(key, value)
        self.settings.sync()

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
        demo = QHBoxLayout()
        demo.addWidget(label("Teacher demo · Paderborn, reduced speed, seed 7", "Subtitle", True), 1)
        self.demo_buttons = []
        for title, faulty in [("Healthy demo", False), ("Faulty demo", True)]:
            entry = button(title)
            entry.clicked.connect(lambda checked=False, f=faulty: self.start_demo(f))
            demo.addWidget(entry)
            self.demo_buttons.append(entry)
        body.addLayout(demo)
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
        if index == 4:
            self.session_page.refresh()

    def start_demo(self, faulty):
        if not self.controller.busy:
            self.signal_page.load_demo(faulty)
            self.navigate(1)

    def start_job(self, action, arguments):
        if self.controller.busy:
            self.append_log("Finish or stop the current task before starting another.")
            return
        arg_map = dict(zip(arguments[::2], arguments[1::2]))
        error = None
        if action in {"inspect", "predict"}:
            path = Path(arg_map.get("--file", ""))
            if not path.is_file() or path.suffix.lower() != ".mat":
                error = "Browse to an existing .mat recording, or choose an installed demo recording."
            elif arg_map.get("--dataset") == "cwru" and not arg_map.get("--channel"):
                error = "Enter the MATLAB vibration channel and original sampling rate for CWRU."
            elif action == "predict" and not Path(arg_map.get("--checkpoint", "")).is_file():
                error = "Select an installed model or restore the published checkpoints."
        elif action == "train":
            missing = training_requirements(arg_map.get("--target"), arg_map.get("--seed"))
            if missing:
                error = f"Training inputs are missing ({len(missing)} files). First missing: {missing[0]}. See the setup guide."
        if error:
            self.start_error(action, error)
            return
        self.active_action = action
        self.started = time.monotonic()
        self.last_update = self.started
        try:
            self.controller.start(action, arguments)
        except OSError as error:
            self.start_error(action, f"Cannot create the session folder: {error}")
            return
        self.latest_session = self.controller.session or self.latest_session

    def start_error(self, action, message):
        self.stage_label.setText("Could not start · see page for details")
        self.append_log(message)
        if action in {"inspect", "predict"}:
            self.signal_page.handle({"event": "error", "message": message})
        elif action == "train":
            self.training_page.handle({"event": "error", "message": message})

    def set_busy(self, busy):
        self.stop.setEnabled(busy)
        self.signal_page.set_busy(busy)
        self.training_page.set_busy(busy)
        self.session_page.set_busy(busy)
        for widget in self.demo_buttons:
            widget.setEnabled(not busy)
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
        if not success:
            cancelled = self.controller.cancel_requested
            self.badge.setText("LOCAL · CANCELLED" if cancelled else "LOCAL · ERROR")
            self.badge.setStyleSheet(f"background:{t.ERROR_BG};color:{t.AMBER if cancelled else t.RED};")
            if cancelled or not self.controller.had_error:
                if cancelled:
                    self.handle_event({"event": "cancelled", "message": message})
                else:
                    self.start_error(self.active_action, message)
        elif self.controller.session:
            session = read_session(self.controller.session)
            if session.status == "complete":
                if session.teacher:
                    self.latest_training_session = session.folder
                    self.training_page.open_result.setEnabled(True)
                    if self.signal_page.dataset.currentData() == "paderborn":
                        self.signal_page.add_model("Run · " + session.folder.name[:15], session.teacher, session.description)
                elif session.action in {"predict", "inspect"}:
                    self.latest_signal_session = session.folder
                    self.signal_page.open_result.setEnabled(True)
        if self.stack.currentIndex() == 4:
            self.session_page.refresh(self.controller.session)
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
        self.open_directory(self.latest_session or self.session_root)

    def open_directory(self, path):
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                self.statusBar().showMessage(str(path), 15000)
        except OSError as error:
            QMessageBox.warning(self, "Could not open folder", str(error))

    def show_session(self, path):
        self.navigate(4)
        self.session_page.refresh(path)

    def reuse_teacher(self, session):
        if self.controller.busy or not session:
            return
        session = read_session(session.folder)
        if session.teacher:
            self.signal_page.use_teacher(session)
            self.navigate(1)
            self.statusBar().showMessage("Saved teacher selected. Choose a Paderborn recording to predict.", 15000)
        else:
            self.session_page.refresh()
            QMessageBox.warning(self, "Teacher unavailable", "The run is incomplete or its saved model is missing.")

    def export_report(self, session):
        if session is None:
            return None
        try:
            path = export_session(session, self.report_directory)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            QMessageBox.warning(self, "Could not export report", str(error))
            return None
        self.last_report_export = path
        self.append_log(f"Report exported · {path}")
        self.statusBar().showMessage(f"Saved {path.name} · use Report folder", 15000)
        return path

    def save_current_png(self):
        view = ["overview", "signal_prediction", "live_commissioning", "experiments", "saved_sessions"][self.stack.currentIndex()]
        path = snapshot_path(self.png_export_directory, view)
        try:
            save_widget_png(self, path)
        except (OSError, RuntimeError, ValueError) as error:
            self.statusBar().showMessage("PNG export failed. Your active task continues.", 15000)
            QMessageBox.warning(self, "Could not save PNG", f"The image could not be saved to:\n{path}\n\n{error}")
            return None
        self.last_png_export = path
        self.append_log(f"PNG saved · {path}")
        self.statusBar().showMessage(f"Saved {path.name} · use Open PNG folder", 15000)
        return path

    def open_png_folder(self):
        try:
            self.png_export_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.warning(self, "Could not open PNG folder", str(error))
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.png_export_directory))):
            self.statusBar().showMessage(f"PNG folder: {self.png_export_directory}", 15000)

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
        self.save_preferences()
        if self.gpu_process.state() != QProcess.ProcessState.NotRunning:
            self.gpu_process.kill()
            self.gpu_process.waitForFinished(1000)
        event.accept()


def create_application():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
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
