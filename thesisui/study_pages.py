"""Native pages for live commissioning and measured experiment comparisons."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QProgressBar,
                             QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from thesisui import theme as t
from thesisui.data import METHOD_NAMES, TARGETS, TARGET_NAMES, summary
from thesisui.plots import Plot
from thesisui.widgets import Metric, button, label, panel


class TrainingPage(QWidget):
    run_requested = Signal(str, list)

    def __init__(self):
        super().__init__()
        self.rows = []
        self.last_checkpoint = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        controls, body = panel("Live teacher–student commissioning", "A new local training session using source labels and unlabeled target signals.")
        row = QHBoxLayout()
        self.target = QComboBox()
        for code in TARGETS:
            self.target.addItem(TARGET_NAMES[code], code)
        self.seed = QComboBox()
        for seed in [7, 17, 27]:
            self.seed.addItem(f"Seed {seed}", seed)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 30)
        self.epochs.setValue(10)
        self.epochs.setSuffix(" epochs")
        self.start = button("Start live training", True)
        row.addWidget(self.target, 2)
        row.addWidget(self.seed)
        row.addWidget(self.epochs)
        row.addWidget(self.start)
        body.addLayout(row)
        body.addWidget(label("EMA 0.99  ·  confidence ≥ 0.90  ·  source + target objective  ·  fixed final epoch", "Subtitle", True))
        layout.addWidget(controls)
        self.phase = label("Ready · models and prepared data will be verified when you start", "Subtitle", True)
        layout.addWidget(self.phase)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        metrics = QHBoxLayout()
        self.epoch_metric = Metric("Epoch / batch", "—", "No active training session", t.CYAN)
        self.source_metric = Metric("Source loss", "—", "Supervised objective")
        self.acceptance_metric = Metric("Accepted pseudo-labels", "—", "True target labels remain hidden", t.PURPLE)
        self.validation_metric = Metric("Source validation F1", "—", "Diagnostic only", t.GREEN)
        for widget in [self.epoch_metric, self.source_metric, self.acceptance_metric, self.validation_metric]:
            metrics.addWidget(widget, 1)
        layout.addLayout(metrics)
        charts = QHBoxLayout()
        self.loss = Plot("Training objectives", "Completed epoch", "Loss", 270)
        self.loss.placeholder("Source and target losses appear as epochs finish")
        self.acceptance = Plot("Teacher and student behaviour", "Completed epoch", "Fraction", 270)
        self.acceptance.placeholder("Acceptance, confidence and disagreement appear here")
        charts.addWidget(self.loss, 1)
        charts.addWidget(self.acceptance, 1)
        layout.addLayout(charts, 1)
        self.explanation = label("Interactive runs save in outputs/desktop_sessions. Your published experiments remain available on the Experiments screen.", "Subtitle", True)
        layout.addWidget(self.explanation)
        self.start.clicked.connect(self.request)

    def request(self):
        self.rows = []
        self.last_checkpoint = None
        self.progress.setValue(0)
        self.loss.placeholder("Waiting for the first completed epoch…")
        self.acceptance.placeholder("Waiting for the first completed epoch…")
        self.epoch_metric.update_value("Starting", "Loading and checking data")
        self.source_metric.update_value("—")
        self.acceptance_metric.update_value("—")
        self.validation_metric.update_value("—")
        self.run_requested.emit("train", ["--target", self.target.currentData(), "--seed", str(self.seed.currentData()),
                                          "--epochs", str(self.epochs.value())])

    def handle(self, item):
        event = item["event"]
        if event == "stage":
            self.phase.setText(item["message"])
        elif event == "training_started":
            self.phase.setText(f"{item['device'].upper()} · {item['source_windows']:,} labeled source windows + "
                               f"{item['target_windows']:,} unlabeled target windows")
        elif event == "batch":
            progress = ((item["epoch"] - 1) + item["step"] / item["steps"]) / item["total_epochs"]
            self.progress.setValue(round(progress * 1000))
            self.epoch_metric.update_value(f"{item['epoch']} / {item['total_epochs']}",
                                           f"Batch {item['step']} / {item['steps']} completed")
            self.source_metric.update_value(f"{item['source_loss']:.4f}", "Latest completed source batch")
        elif event == "epoch":
            self.rows.append(item)
            epochs = [r["epoch"] for r in self.rows]
            self.source_metric.update_value(f"{item['source_loss']:.4f}", "Mean over completed epoch")
            self.acceptance_metric.update_value(f"{item['acceptance']:.1%}",
                                                f"{item['accepted']:,} / {item['target_exposures']:,} exposures")
            self.validation_metric.update_value(f"{item['source_val_teacher_macro_f1']:.1%}", "Source bearing validation; no target truth")
            self.loss.series([("Source", epochs, [r["source_loss"] for r in self.rows], t.CYAN),
                              ("Target", epochs, [r["target_loss_full_batch"] for r in self.rows], t.PURPLE)])
            self.acceptance.series([("Accepted", epochs, [r["acceptance"] for r in self.rows], t.GREEN),
                                    ("Confidence", epochs, [r["mean_confidence"] for r in self.rows], t.PURPLE),
                                    ("Disagreement", epochs, [r["teacher_student_disagreement"] for r in self.rows], t.AMBER)],
                                   ylim=(-.03, 1.04), percent=True)
            self.explanation.setText(f"Accepted pseudo-labels this epoch: healthy {item['accepted_healthy']:,} · faulty {item['accepted_faulty']:,}. "
                                     "These are model guesses, not reference labels.")
        elif event == "result" and item.get("kind") == "train":
            self.progress.setValue(1000)
            self.last_checkpoint = item["checkpoint"]
            self.phase.setText(f"Complete · {item['updates']} updates · teacher checkpoint saved")
            self.explanation.setText("Saved: " + item["checkpoint"])
        elif event in {"error", "cancelled"}:
            self.phase.setText(item["message"])

    def set_busy(self, busy):
        for widget in [self.start, self.target, self.seed, self.epochs]:
            widget.setEnabled(not busy)


class ExperimentsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        controls = QHBoxLayout()
        controls.addWidget(label("Paderborn · published commissioning study", "SectionTitle"))
        controls.addStretch()
        self.target = QComboBox()
        for code in TARGETS:
            self.target.addItem(TARGET_NAMES[code], code)
        controls.addWidget(self.target)
        layout.addLayout(controls)
        layout.addWidget(label("SAVED RESULTS · 3 seeds · fixed bearing split · recording-level metrics", "Eyebrow"))
        self.chart = Plot("Macro F1 by method", "", "Macro F1 (%)", 275)
        layout.addWidget(self.chart, 1)
        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels(["Method", "Macro F1", "Balanced accuracy", "Fault recall", "Healthy FPR"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(200)
        layout.addWidget(self.table)
        self.finding = label("", "Subtitle", True)
        self.finding.setStyleSheet(f"color:{t.AMBER};")
        layout.addWidget(self.finding)
        layout.addWidget(label("Exploratory evidence: one held-out healthy bearing. Seed variability does not measure population uncertainty. "
                               "These saved results are distinct from a live training session.", "Subtitle", True))
        self.target.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        data = summary()
        target = self.target.currentData()
        self.chart.comparison(data, target)
        rows = {r["method"]: r for r in data.get("summary", []) if r["scope"] == "target" and r["adaptation_target"] == target}
        for row_index, (method, name) in enumerate(METHOD_NAMES.items()):
            values = [name]
            r = rows.get(method)
            for key in ["macro_f1", "balanced_accuracy", "fault_recall", "healthy_false_positive_rate"]:
                values.append(f"{r[key+'_mean']*100:.2f} ± {r[key+'_sd']*100:.2f}%" if r else "Unavailable")
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if method == "teacher_student":
                    item.setForeground(QColor(t.PURPLE))
                self.table.setItem(row_index, col, item)
            self.table.setRowHeight(row_index, 38)
        self.finding.setText("Mean across all targets: teacher–student 66.38% vs source EMA 66.84%. "
                             "The current method does not demonstrate an average benefit beyond the matched EMA control.")
