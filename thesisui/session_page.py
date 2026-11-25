"""Browse completed, failed and cancelled runs without loading model weights."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QHeaderView, QPlainTextEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from thesisui.sessions import list_sessions, report_text
from thesisui.widgets import button, label


class SessionPage(QWidget):
    open_requested = Signal(object)
    export_requested = Signal(object)
    teacher_requested = Signal(object)
    exports_requested = Signal()

    def __init__(self, session_root):
        super().__init__()
        self.session_root = session_root
        self.sessions = []
        self.busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self.refresh_button = button("Refresh")
        self.open_button = button("Open folder")
        self.export_button = button("Export report ZIP", True)
        self.exports_button = button("Report folder")
        self.teacher_button = button("Use teacher model")
        for widget in [self.refresh_button, self.open_button, self.teacher_button,
                       self.export_button, self.exports_button]:
            actions.addWidget(widget)
        actions.addStretch()
        layout.addLayout(actions)
        self.note = label("", "Subtitle", True)
        layout.addWidget(self.note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Session ID (UTC)", "Action", "Status", "Recording or run"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 2)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(160)
        self.details.setPlaceholderText("Select a session to read its results.")
        layout.addWidget(self.details, 3)
        self.refresh_button.clicked.connect(lambda: self.refresh())
        self.table.itemSelectionChanged.connect(self.show_selection)
        self.open_button.clicked.connect(lambda: self.open_requested.emit(self.selected()))
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self.selected()))
        self.teacher_button.clicked.connect(lambda: self.teacher_requested.emit(self.selected()))
        self.exports_button.clicked.connect(self.exports_requested)
        self.show_selection()

    def refresh(self, preferred=None):
        selected = self.selected()
        preferred = preferred or (selected.folder if selected else None)
        if preferred:
            preferred = preferred.resolve()
        try:
            self.sessions = list_sessions(self.session_root)
            self.note.setText(f"{len(self.sessions)} local sessions · separate from the saved thesis experiments"
                              if self.sessions else "No sessions yet. Run a prediction or start commissioning first.")
        except OSError as error:
            self.sessions = []
            self.note.setText(f"Could not read the session folder: {error}")
        self.table.setRowCount(len(self.sessions))
        position = 0
        for index, session in enumerate(self.sessions):
            if session.folder == preferred:
                position = index
            for column, value in enumerate([session.folder.name, session.action.title(),
                                             session.status.title(), session.description]):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column in (1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(index, column, item)
        if self.sessions:
            self.table.selectRow(position)
        self.show_selection()

    def selected(self):
        index = self.table.currentRow()
        return self.sessions[index] if 0 <= index < len(self.sessions) else None

    def show_selection(self):
        session = self.selected()
        self.open_button.setEnabled(session is not None)
        self.export_button.setEnabled(bool(session and session.status == "complete"))
        self.teacher_button.setEnabled(bool(session and session.teacher and not self.busy))
        self.details.setPlainText(report_text(session) if session else "")

    def set_busy(self, busy):
        self.busy = busy
        self.show_selection()
