"""Native signal browser, embedded plots and recording-level inference."""

from pathlib import Path
import time

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
                             QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from thesisui import theme as t
from thesisui.data import ROOT, TARGETS, TARGET_NAMES, checkpoint_choices
from thesisui.plots import Plot
from thesisui.widgets import button, form_field, label, panel


class SignalPage(QWidget):
    run_requested = Signal(str, list)

    def __init__(self):
        super().__init__()
        self.info = None
        self.binary = True
        self.playing = False
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)
        controls, left = panel("Recording controls", "Load a real MATLAB recording, inspect it, then run a saved model.")
        controls.setMinimumWidth(275)
        controls.setMaximumWidth(330)
        left.setSpacing(8)
        self.dataset = QComboBox()
        self.dataset.addItem("Paderborn · healthy / faulty", "paderborn")
        self.dataset.addItem("CWRU · four fault classes", "cwru")
        form_field("Dataset / input format", self.dataset, left)
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
        self.model.setMinimumWidth(230)
        form_field("Saved model", self.model, left)
        self.file = QLineEdit()
        self.file.setPlaceholderText("Choose a .mat recording")
        self.file.setToolTip("Full path to your vibration recording")
        form_field("Recording", self.file, left)
        self.browse = button("Browse .mat file…")
        left.addWidget(self.browse)
        quick = QHBoxLayout()
        self.quick_healthy, self.quick_faulty = button("Healthy example"), button("Faulty example")
        quick.addWidget(self.quick_healthy)
        quick.addWidget(self.quick_faulty)
        left.addLayout(quick)
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
        left.addLayout(actions)
        left.addStretch()
        self.decision = label("Ready to inspect", "SectionTitle", True)
        self.decision.setStyleSheet(f"font-size:23px;color:{t.MUTED};")
        self.score = label("Choose a recording and model.", "Subtitle", True)
        outer.addWidget(controls)
        right = QVBoxLayout()
        result_panel, result_layout = panel()
        result_layout.setContentsMargins(14, 10, 14, 10)
        result_layout.setSpacing(4)
        result_row = QHBoxLayout()
        result_row.addWidget(self.decision)
        result_row.addStretch()
        result_row.addWidget(self.score)
        result_layout.addLayout(result_row)
        result_layout.addWidget(label("Scores are model outputs, not calibrated failure risks.", "Subtitle"))
        right.addWidget(result_panel)
        self.signal_summary = label("No recording loaded · measurements stay on this computer", "Subtitle", True)
        right.addWidget(self.signal_summary)
        self.wave = Plot("Vibration waveform", "Time (s, nominal sampling rate)", "Recorded amplitude", 210)
        self.wave.placeholder("Load a recording to see its real vibration signal")
        right.addWidget(self.wave, 1)
        play_row = QHBoxLayout()
        self.play = button("Play recording")
        self.play.setEnabled(False)
        self.play_info = label("Recorded-data playback", "Subtitle")
        play_row.addWidget(self.play)
        play_row.addWidget(self.play_info)
        play_row.addStretch()
        right.addLayout(play_row)
        tabs = QTabWidget()
        self.window_scores = Plot("Fault-class score by window", "Window number", "Model score", 195)
        self.window_scores.placeholder("Run prediction for window-by-window scores")
        self.spectrum = Plot("First standardized window · spectrum", "Frequency (Hz, nominal)", "Magnitude", 195)
        self.spectrum.placeholder("Spectrum appears after loading the signal")
        tabs.addTab(self.window_scores, "Window predictions")
        tabs.addTab(self.spectrum, "Signal spectrum")
        right.addWidget(tabs, 1)
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
        self.file.textChanged.connect(self.clear_result)
        self.model.currentIndexChanged.connect(self.clear_result)
        self.refresh_models()
        self.example(False)

    def clear_result(self, *_):
        self.decision.setText("Awaiting prediction")
        self.decision.setStyleSheet(f"font-size:23px;color:{t.MUTED};")
        self.score.setText("Inspect the recording or run the selected model.")
        self.window_scores.placeholder("Run prediction for the selected file and model")

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
            self.model.addItem(title, str(path))
        if options:
            self.model.setCurrentIndex(len(options) - 1)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a trusted project checkpoint", str(ROOT / "outputs"), "PyTorch models (*.pt)")
        if path:
            self.model.addItem(Path(path).name, path)
            self.model.setCurrentIndex(self.model.count() - 1)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a vibration recording", str(ROOT / "data/raw"), "MATLAB recordings (*.mat)")
        if path:
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
            args += ["--checkpoint", self.model.currentData() or ""]
            self.window_scores.placeholder("Waiting for real model outputs…")
            self.decision.setText("Processing…")
            self.decision.setStyleSheet(f"font-size:23px;color:{t.CYAN};")
            self.score.setText("Preprocessing and inference run in a separate process.")
        else:
            self.decision.setText("Inspecting signal…")
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
        elif event == "result" and item.get("kind") == "inspect":
            self.decision.setText("Signal ready")
            self.score.setText("Run prediction to obtain a model decision.")
        elif event == "error":
            self.decision.setText("Check the input")
            self.decision.setStyleSheet(f"font-size:23px;color:{t.RED};")
            self.score.setText(item["message"])

    def set_busy(self, busy):
        for control in [self.dataset, self.target, self.seed, self.model, self.custom, self.file, self.browse,
                        self.quick_healthy, self.quick_faulty, self.channel, self.sampling, self.inspect, self.predict]:
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
