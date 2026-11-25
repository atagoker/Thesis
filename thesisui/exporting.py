"""Native PNG exports of the current application view, independent of training."""

from datetime import datetime
from pathlib import Path
import uuid

from PySide6.QtCore import QIODevice, QSaveFile, Qt
from PySide6.QtGui import QPixmap

from thesisui.branding import APP_NAME


def snapshot_path(directory, view):
    """Give every click a separate, readable filename without replacing images."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(directory) / f"{APP_NAME}_{view}_{stamp}_{uuid.uuid4().hex[:8]}.png"


def save_widget_png(widget, path, scale=2):
    """Render the visible native window at 2x size and commit the PNG atomically.

    This renders only our widgets, not the screen or other applications. It does
    not resize the window, navigate pages, advance playback or touch any worker.
    Scrollable areas retain their currently visible position.
    """
    if not isinstance(scale, int) or not 1 <= scale <= 4:
        raise ValueError("PNG scale must be an integer between 1 and 4.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = QPixmap(widget.width() * scale, widget.height() * scale)
    if pixmap.isNull():
        raise OSError("Could not allocate the PNG image.")
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    widget.render(pixmap)
    output = QSaveFile(str(path))
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError(output.errorString())
    try:
        if not pixmap.save(output, "PNG"):
            raise OSError("Qt could not encode the PNG image.")
        if not output.commit():
            raise OSError(output.errorString())
    except Exception:
        output.cancelWriting()
        raise
    return path
