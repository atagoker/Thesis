"""Native signal browser, embedded plots and recording-level inference."""

from pathlib import Path
import time

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QScrollArea,
                             QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from thesisui import theme as t
from thesisui.data import ROOT, TARGETS, TARGET_NAMES, checkpoint_choices
from thesisui.plots import Plot
from thesisui.sessions import list_sessions, SESSION_ROOT
from thesisui.widgets import button, form_field, label, panel

QtModelDetails = Qt.ItemDataRole.UserRole + 1

class SignalPage(QWidget):
    run_requested = Signal(str, list)
    results_requested = Signal()

    def __init__(self, session_root=SESSION_ROOT):
        super().__init__()
        self.session_root = session_root
        self.last_recording_folder = str(ROOT / "data/raw")
        self.last_model_folder = str(ROOT / "outputs")
        self.info = None
        self.binary = True
        self.playing = False
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)
        controls, left = panel("Prediction controls", "Choose a dataset, model and recording. Inspect the signal or run a prediction directly.")
        controls.setMinimumWidth(285)
        left.setSpacing(8)
        self.dataset = QComboBox()
        self.dataset.addItem("Paderborn · healthy / faulty", "paderborn")
        self.dataset.addItem("CWRU · four fault classes", "cwru")
        form_field("Dataset / input format", self.dataset, left)
        self.file = QLineEdit()
        self.file.setPlaceholderText("Choose a .mat recording")
        self.file.setToolTip("Full path to your vibration recording")
        form_field("Recording", self.file, left)
        self.browse = button("Browse .mat file…")
        left.addWidget(self.browse)
        self.target = QComboBox()
        for target in TARGETS:
            self.target.addItem(TARGET_NAMES[target], target)
        form_field("Model adaptation condition", self.target, left)
        self.seed = QComboBox()
        for seed in [7, 17, 27]:
            self.seed.addItem(f"Seed {seed}", seed)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed)
        self.custom = button("Custom model…")
        seed_row.addWidget(self.custom)
        left.addLayout(seed_row)
        self.model = QComboBox()
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(19)
        form_field("Saved model · seed above", self.model, left)
        self.model_note = label("", "Subtitle", True)
        left.addWidget(self.model_note)
        quick = QHBoxLayout()
        self.quick_healthy, self.quick_faulty = button("Healthy example"), button("Faulty example")
        quick.addWidget(self.quick_healthy)
        quick.addWidget(self.quick_faulty)
        self.channel = QLineEdit("X138_DE_time")
        self.sampling = QSpinBox()
        self.sampling.setRange(1, 1000000)
        self.sampling.setValue(48000)
        self.sampling.setSuffix(" Hz")
        self.cwru_fields = QWidget()
        cwru_layout = QFormLayout(self.cwru_fields)
        cwru_layout.setContentsMargins(0, 0, 0, 0)
        cwru_layout.addRow("MAT channel", self.channel)
        cwru_layout.addRow("Original rate", self.sampling)
        self.cwru_fields.hide()
        left.addWidget(self.cwru_fields)
        actions = QHBoxLayout()
        self.inspect = button("Inspect signal")
        self.predict = button("Run prediction", True)
        actions.addWidget(self.inspect)
        actions.addWidget(self.predict)
        self.device = QComboBox()
        self.device.addItem("Device: Auto (prefer CUDA)", "auto")
        self.device.addItem("Device: CPU", "cpu")
        self.device.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device.setMinimumContentsLength(16)
        left.addWidget(self.device)
        left.addStretch()
        self.decision = label("Ready to inspect", "SectionTitle", True)
        self.decision.setStyleSheet(f"font-size:23px;color:{t.MUTED};")
        self.score = label("Choose a recording and model.", "Subtitle", True)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setFixedWidth(322)
        control_scroll.setWidget(controls)
        fixed_controls = QVBoxLayout()
        fixed_controls.addWidget(control_scroll, 1)
        fixed_controls.addLayout(quick)
        fixed_controls.addLayout(actions)
        outer.addLayout(fixed_controls)
        right = QVBoxLayout()
        result_panel, result_layout = panel()
        result_layout.setContentsMargins(14, 10, 14, 10)
        result_layout.setSpacing(4)
        result_row = QHBoxLayout()
        result_row.addWidget(self.decision)
        result_row.addStretch()
        result_row.addWidget(self.score)
        self.open_result = button("Saved result")
        self.open_result.setEnabled(False)
        self.open_result.clicked.connect(self.results_requested)
        result_row.addWidget(self.open_result)
        result_layout.addLayout(result_row)
        result_layout.addWidget(label("Scores are model outputs, not calibrated failure risks.", "Subtitle", True))
        right.addWidget(result_panel)
        self.signal_summary = label("No recording loaded · measurements stay on this computer", "Subtitle", True)
        right.addWidget(self.signal_summary)
        self.condition_warning = label("", "Subtitle", True)
        self.condition_warning.setStyleSheet(f"color:{t.AMBER};")
        self.condition_warning.hide()
        right.addWidget(self.condition_warning)
        self.tabs = QTabWidget()
        waveform = QWidget()
        waveform_layout = QVBoxLayout(waveform)
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        self.wave = Plot("Vibration waveform", "Time (s, nominal sampling rate)", "Recorded amplitude", 170)
        self.wave.placeholder("Load a recording to see its real vibration signal")
        waveform_layout.addWidget(self.wave, 1)
        play_row = QHBoxLayout()
        self.play = button("Play recording")
        self.play.setEnabled(False)
        self.play_info = label("Recorded-data playback", "Subtitle")
        play_row.addWidget(self.play)
        play_row.addWidget(self.play_info)
        play_row.addStretch()
        waveform_layout.addLayout(play_row)
        self.window_scores = Plot("Fault-class score by window", "Window number", "Model score", 195)
        self.window_scores.placeholder("Run prediction for window-by-window scores")
        self.spectrum = Plot("First standardized window · spectrum", "Frequency (Hz, nominal)", "Magnitude", 195)
        self.spectrum.placeholder("Spectrum appears after loading the signal")
        self.tabs.addTab(waveform, "Waveform")
        self.tabs.addTab(self.window_scores, "Window scores")
        self.tabs.addTab(self.spectrum, "Spectrum")
        right.addWidget(self.tabs, 1)
        outer.addLayout(right, 1)
        self.dataset.currentIndexChanged.connect(self.refresh_models)
        self.seed.currentIndexChanged.connect(self.refresh_models)
        self.target.currentIndexChanged.connect(self.refresh_models)
        self.custom.clicked.connect(self.choose_model)
        self.browse.clicked.connect(self.choose_file)
        self.quick_healthy.clicked.connect(lambda: self.example(False))
        self.quick_faulty.clicked.connect(lambda: self.example(True))
        self.inspect.clicked.connect(lambda: self.request("inspect"))
        self.predict.clicked.connect(lambda: self.request("predict"))
        self.play.clicked.connect(self.toggle_play)
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.play_tick)
        self.file.textChanged.connect(self.clear_signal)
        self.channel.textChanged.connect(self.clear_signal)
        self.sampling.valueChanged.connect(self.clear_signal)
        self.dataset.currentIndexChanged.connect(self.clear_signal)
        self.model.currentIndexChanged.connect(self.clear_result)
        self.model.currentIndexChanged.connect(self.describe_model)
        self.refresh_models()
        self.example(False)

    def clear_result(self, *_):
        self.decision.setText("Awaiting prediction")
        self.decision.setStyleSheet(f"font-size:23px;color:{t.MUTED};")
        self.score.setText("Inspect the recording or run the selected model.")
        self.score.setToolTip("")
        self.condition_warning.hide()
        self.open_result.setEnabled(False)
        self.window_scores.placeholder("Run prediction for the selected file and model")

    def clear_signal(self, *_):
        self.timer.stop()
        self.playing = False
        self.info = None
        self.play.setText("Play recording")
        self.play.setEnabled(False)
        self.play_info.setText("Recorded-data playback")
        self.signal_summary.setText("No recording loaded · measurements stay on this computer")
        self.wave.placeholder("Load a recording to see its real vibration signal")
        self.spectrum.placeholder("Spectrum appears after loading the signal")
        self.clear_result()

    def describe_model(self, *_):
        meta = self.model.currentData(QtModelDetails) or "No saved model for this selection. Restore results or select a trusted checkpoint."
        self.model_note.setText(meta)
        self.model.setToolTip(self.model.currentData() or "")

    def load_demo(self, faulty):
        self.dataset.setCurrentIndex(0)
        self.target.setCurrentIndex(0)
        self.seed.setCurrentIndex(0)
        self.device.setCurrentIndex(0)
        self.refresh_models()
        index = next((i for i in range(self.model.count())
                      if Path(self.model.itemData(i)).name == "teacher_student.pt"), -1)
        self.model.setCurrentIndex(index)
        self.example(faulty)

    def add_model(self, name, path, details):
        path = str(path)
        index = self.model.findData(path)
        if index < 0:
            self.model.addItem(name, path)
            index = self.model.count() - 1
        self.model.setItemData(index, details, QtModelDetails)
        return index

    def refresh_models(self):
        self.binary = self.dataset.currentData() == "paderborn"
        self.cwru_fields.setVisible(not self.binary)
        self.target.setEnabled(self.binary)
        self.model.clear()
        if self.binary:
            options = checkpoint_choices(self.seed.currentData(), self.target.currentData())
        else:
            seed = self.seed.currentData()
            options = []
            for title, filename in [("Initial source", "source_best.pt"), ("Extra source", "continued_source.pt"),
                                    ("Source EMA", "source_ema_control.pt"), ("Teacher · target load 1 hp", "teacher_target1.pt")]:
                for parent in ["outputs", "results"]:
                    path = ROOT / parent / f"pilot_v1/seed_{seed}" / filename
                    if path.is_file():
                        options.append((title, path))
                        break
        for title, path in options:
            base = "Paderborn · 2 classes" if self.binary else "CWRU · 4 classes"
            detail = "Source condition only"
            if path.name == "teacher_student.pt":
                detail = TARGET_NAMES[self.target.currentData()]
            elif path.name == "teacher_target1.pt":
                detail = "Target load: 1 hp"
            self.add_model(title, path, f"{base} · {detail}")
        if options:
            self.model.setCurrentIndex(len(options) - 1)
        if self.binary:
            try:
                for session in list_sessions(self.session_root):
                    if session.teacher:
                        self.add_model("Run · " + session.folder.name[:15], session.teacher, session.description)
            except OSError:
                pass
        self.describe_model()

    def use_teacher(self, session):
        self.dataset.setCurrentIndex(self.dataset.findData("paderborn"))
        for widget, value in [(self.target, session.result.get("target")), (self.seed, session.result.get("seed"))]:
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        self.model.setCurrentIndex(self.add_model("Run · " + session.folder.name[:15], session.teacher, session.description))
        self.describe_model()

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a trusted project checkpoint", self.last_model_folder, "PyTorch models (*.pt)")
        if path:
            self.last_model_folder = str(Path(path).parent)
            self.model.setCurrentIndex(self.add_model(Path(path).name, path, "Custom checkpoint · validated before inference"))
            self.describe_model()

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a vibration recording", self.last_recording_folder, "MATLAB recordings (*.mat)")
        if path:
            self.last_recording_folder = str(Path(path).parent)
            self.file.setText(path)

    def example(self, faulty):
        if self.dataset.currentData() == "paderborn":
            bearing = "KA22" if faulty else "K006"
            path = ROOT / "data/raw/paderborn" / bearing / f"{self.target.currentData()}_{bearing}_1.mat"
        else:
            index = 138 if faulty else 97
            path = ROOT / "data/raw/cwru48" / f"{index}.mat"
            self.channel.setText(f"X{index:03d}_DE_time")
        self.file.setText(str(path))

    def request(self, action):
        self.timer.stop()
        self.playing = False
        self.play.setText("Play recording")
        args = ["--file", self.file.text().strip(), "--dataset", self.dataset.currentData(),
                "--channel", self.channel.text().strip(), "--sampling-rate", str(self.sampling.value())]
        self.clear_result()
        if action == "predict":
            args += ["--checkpoint", self.model.currentData() or "", "--device", self.device.currentData()]
            self.window_scores.placeholder("Waiting for real model outputs…")
            self.decision.setText("Processing…")
            self.decision.setStyleSheet(f"font-size:23px;color:{t.CYAN};")
            self.score.setText("Preprocessing and inference run in a separate process.")
        else:
            self.decision.setText("Inspecting signal…")
            self.tabs.setCurrentIndex(0)
        self.run_requested.emit(action, args)

    def handle(self, item):
        event = item["event"]
        if event == "signal":
            self.info = item
            self.wave.waveform(item)
            self.signal_summary.setText(f"{Path(item['file']).name}  ·  {item['raw_samples']:,} samples  ·  "
                                        f"{item['windows']} windows  ·  {item['duration']:.2f} s nominal")
            self.spectrum.series([("Spectrum", item["fft_hz"], item["fft_amplitude"], t.CYAN)])
            self.play.setEnabled(True)
        elif event == "prediction_progress":
            self.window_scores.scores(item["faulty_scores"], self.binary)
            self.score.setText(f"Processed {item['completed']} / {item['total']} windows")
        elif event == "result" and item.get("kind") == "predict":
            decision = item["predicted_class"]
            self.decision.setText(decision.replace("_", " ").upper())
            self.decision.setStyleSheet(f"font-size:23px;color:{t.GREEN if decision == 'healthy' else t.RED};")
            self.score.setText(f"Fault-class model score: {item['mean_faulty_score']:.4f}\n"
                               f"{item['window_count']} windows · {item['device'].upper()}")
            self.score.setToolTip("\n".join(f"{key}: {value:.4f}" for key, value in item['mean_probabilities'].items()))
            self.tabs.setCurrentWidget(self.window_scores)
            model_target, record_target = item.get("adaptation_target_condition"), item.get("recording_condition_hint")
            if model_target and record_target and model_target != record_target:
                self.condition_warning.setText(f"Model adapted to {model_target}; filename indicates {record_target}.")
                self.condition_warning.show()
        elif event == "result" and item.get("kind") == "inspect":
            self.decision.setText("Signal ready")
            self.score.setText("Run prediction to obtain a model decision.")
        elif event in {"error", "cancelled"}:
            self.decision.setText("Task cancelled" if event == "cancelled" else "Could not finish")
            self.decision.setStyleSheet(f"font-size:23px;color:{t.RED};")
            self.score.setText(item["message"])
            self.window_scores.placeholder("No completed prediction for this task")

    def set_busy(self, busy):
        for control in [self.dataset, self.target, self.seed, self.model, self.custom, self.file, self.browse,
                        self.quick_healthy, self.quick_faulty, self.channel, self.sampling, self.inspect, self.predict, self.device]:
            control.setEnabled(not busy)
        if not busy:
            self.target.setEnabled(self.binary)

    def toggle_play(self):
        if not self.info:
            return
        self.playing = not self.playing
        self.play.setText("Stop playback" if self.playing else "Play recording")
        if self.playing:
            self.play_started = time.monotonic()
            self.timer.start()
        else:
            self.timer.stop()

    def play_tick(self):
        position = time.monotonic() - self.play_started
        if position >= self.info["duration"]:
            self.playing = False
            self.timer.stop()
            self.play.setText("Play recording")
            self.wave.waveform(self.info)
            self.play_info.setText("Playback complete · recorded data")
        else:
            self.wave.waveform(self.info, position)
            self.play_info.setText(f"{position:.1f} / {self.info['duration']:.1f} s · recorded data")
