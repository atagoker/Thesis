"""Small native Qt building blocks for the desktop workspace."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from thesisui import theme as t


def label(text, name=None, wrap=False):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                         QSizePolicy.Policy.Preferred if wrap else QSizePolicy.Policy.Fixed)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


def button(text, primary=False):
    widget = QPushButton(text.replace("&", "&&"))
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        widget.setObjectName("Primary")
    return widget


def panel(title=None, subtitle=None):
    frame = QFrame()
    frame.setObjectName("Panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    if title:
        layout.addWidget(label(title, "SectionTitle"))
    if subtitle:
        layout.addWidget(label(subtitle, "Subtitle", True))
    return frame, layout


class Metric(QFrame):
    def __init__(self, title, value="—", detail="", color=t.TEXT):
        super().__init__()
        self.setObjectName("Panel")
        self.setProperty("accent", {t.CYAN: "cyan", t.GREEN: "green", t.PURPLE: "purple"}.get(color, "blue"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 13, 17, 13)
        layout.setSpacing(4)
        layout.addWidget(label(title.upper(), "MetricLabel"))
        self.value = label(value, "MetricValue")
        self.value.setStyleSheet(f"color:{color};")
        layout.addWidget(self.value)
        self.detail = label(detail, "Subtitle", True)
        self.detail.setStyleSheet(f"font-size:11px;color:{t.MUTED};")
        layout.addWidget(self.detail)

    def update_value(self, value, detail=None):
        self.value.setText(str(value))
        if detail is not None:
            self.detail.setText(detail)


class WaveMark(QWidget):
    def sizeHint(self):
        return QSize(44, 44)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(t.ACTIVE_BG))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 11, 11)
        painter.setPen(QPen(QColor(t.CYAN), 2.1))
        path = QPainterPath()
        points = [(7, 24), (13, 24), (17, 14), (22, 32), (27, 10), (31, 24), (38, 24)]
        path.moveTo(*points[0])
        for x, y in points[1:]:
            path.lineTo(x, y)
        painter.drawPath(path)


def form_field(text, widget, layout):
    layout.addWidget(label(text.upper(), "MetricLabel"))
    layout.addWidget(widget)
