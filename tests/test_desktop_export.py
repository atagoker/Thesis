"""Native PNG export checks without GPU training or a visible desktop."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from thesisui.exporting import save_widget_png, snapshot_path
    from thesisui.window import MainWindow, create_application
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "Optional native Qt desktop dependency is not installed")
class DesktopExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_application()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "saved_pngs"
        self.window = MainWindow(auto_health=False, export_directory=self.folder, settings=False,
                                 session_root=Path(self.temp.name) / "sessions")
        self.window.gpu_timer.stop()
        self.window.resize(1100, 760)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_one_click_exports_every_page_as_real_double_resolution_png(self):
        names = ["overview", "signal_prediction", "live_commissioning", "experiments", "saved_sessions"]
        for index, name in enumerate(names):
            with self.subTest(page=name):
                self.window.navigate(index)
                self.app.processEvents()
                QTest.mouseClick(self.window.save_png_button, Qt.MouseButton.LeftButton)
                path = self.window.last_png_export
                self.assertTrue(path.is_file())
                self.assertIn(name, path.name)
                self.assertEqual(path.parent, self.folder)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                image = QImage(str(path))
                self.assertFalse(image.isNull())
                self.assertEqual(image.width(), self.window.width() * 2)
                self.assertEqual(image.height(), self.window.height() * 2)
                self.assertEqual(self.window.stack.currentIndex(), index)
                self.assertEqual(self.window.size().width(), 1100)
        self.assertEqual(len(list(self.folder.glob("*.png"))), 5)

    def test_repeated_saves_keep_both_images(self):
        first = self.window.save_current_png()
        original = first.read_bytes()
        second = self.window.save_current_png()
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        self.assertTrue(second.is_file())
        self.assertEqual(len(list(self.folder.glob("*.png"))), 2)

    def test_export_remains_enabled_while_task_controls_are_busy(self):
        self.window.set_busy(True)
        self.window.navigate(2)
        self.assertTrue(self.window.save_png_button.isEnabled())
        with patch.object(self.window.controller, "cancel") as cancel:
            path = self.window.save_current_png()
            self.assertTrue(path.is_file())
            cancel.assert_not_called()
        self.assertEqual(self.window.badge.text(), "LOCAL · WORKING")
        self.window.set_busy(False)

    def test_write_failure_reports_error_without_claiming_success(self):
        self.folder.write_text("This path is a file, not a directory.", encoding="utf-8")
        with patch("thesisui.window.QMessageBox.warning") as warning:
            self.assertIsNone(self.window.save_current_png())
            warning.assert_called_once()
        self.assertIsNone(self.window.last_png_export)
        self.assertIn("failed", self.window.statusBar().currentMessage())

    def test_open_folder_uses_native_directory_link(self):
        with patch("thesisui.window.QDesktopServices.openUrl", return_value=True) as open_url:
            QTest.mouseClick(self.window.open_png_folder_button, Qt.MouseButton.LeftButton)
            self.assertTrue(self.folder.is_dir())
            self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), self.folder)

    def test_filename_and_scale_guard(self):
        self.assertNotEqual(snapshot_path(self.folder, "overview"), snapshot_path(self.folder, "overview"))
        with self.assertRaises(ValueError):
            save_widget_png(self.window, self.folder / "invalid.png", scale=0)
