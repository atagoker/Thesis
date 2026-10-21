"""Launch the native Qt Widgets desktop application."""

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    try:
        from thesisui.window import MainWindow, create_application
        app = create_application()
        window = MainWindow()
        screen = app.primaryScreen().availableGeometry()
        window.resize(min(1460, screen.width() - 40), min(980, screen.height() - 70))
        window.show()
        raise SystemExit(app.exec())
    except Exception:
        import traceback
        path = Path(__file__).resolve().parents[1] / "outputs/desktop_startup_error.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback.format_exc(), encoding="utf-8")
        if sys.stderr:
            traceback.print_exc()
        # pythonw has no terminal; show a native message if Qt is available.
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication([])
            from thesisui.branding import APP_NAME
            QMessageBox.critical(None, f"{APP_NAME} could not start", f"Details saved to:\n{path}\n\nRun pip install -r requirements-desktop.txt in the thesis environment.")
        except Exception:
            pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()
