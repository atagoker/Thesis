"""Native Qt desktop palette and widgets."""

BG = "#05070d"
PANEL = "#101624"
PANEL_ALT = "#090e19"
BORDER = "#263550"
TEXT = "#f1f5ff"
MUTED = "#9eafcc"
CYAN = "#00e5ff"
GREEN = "#53ffa8"
AMBER = "#ffca45"
RED = "#ff4f7c"
PURPLE = "#b66bff"
BLUE = "#589bff"

ACTIVE_BG = "#073044"
SUCCESS_BG = "#0a3027"
ERROR_BG = "#351424"

STYLE = f"""
QMainWindow, QWidget#Root {{ background: {BG}; color: {TEXT}; }}
QWidget {{ color: {TEXT}; font-family: 'Segoe UI'; font-size: 13px; }}
QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: 28px; font-weight: 600; }}
QLabel#Subtitle {{ color: {MUTED}; font-size: 13px; }}
QLabel#Eyebrow {{ color: {CYAN}; font-size: 10px; font-weight: 600; letter-spacing: 2px; }}
QLabel#MetricValue {{ font-size: 28px; font-weight: 600; }}
QLabel#MetricLabel {{ color: {MUTED}; font-size: 11px; }}
QLabel#SectionTitle {{ font-size: 16px; font-weight: 600; }}
QFrame#Sidebar {{ background: {PANEL_ALT}; border-right: 1px solid {BORDER}; }}
QFrame#Panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#Panel[accent="cyan"] {{ border-top: 2px solid {CYAN}; }}
QFrame#Panel[accent="green"] {{ border-top: 2px solid {GREEN}; }}
QFrame#Panel[accent="purple"] {{ border-top: 2px solid {PURPLE}; }}
QFrame#Panel[accent="blue"] {{ border-top: 2px solid {BLUE}; }}
QFrame#Divider {{ background: {BORDER}; max-height: 1px; border: none; }}
QLabel#Badge {{ background: {SUCCESS_BG}; color: {GREEN}; border-radius: 6px; padding: 6px 10px; font-size: 11px; font-weight: 600; }}
QPushButton {{ background: #18233a; border: 1px solid #344b70; border-radius: 7px; padding: 9px 16px; font-weight: 600; }}
QPushButton:hover {{ background: #243858; border-color: {BLUE}; }}
QPushButton:pressed {{ background: #304f78; }}
QPushButton:disabled {{ color: #687487; background: #121923; border-color: #202b38; }}
QPushButton#Primary {{ background: {CYAN}; border-color: {CYAN}; color: #03151d; }}
QPushButton#Primary:hover {{ background: #74f3ff; border-color: #74f3ff; }}
QPushButton#Primary:disabled {{ background: #24434d; border-color: #24434d; color: #708f98; }}
QPushButton#Danger {{ color: {RED}; background: {ERROR_BG}; border-color: #92304f; }}
QPushButton#Danger:disabled {{ color: #6d5360; background: #19131b; border-color: #342630; }}
QPushButton#Nav {{ text-align: left; background: transparent; border: none; padding: 13px 14px; color: {MUTED}; font-weight: 500; }}
QPushButton#Nav:hover {{ background: #17263e; color: {TEXT}; }}
QPushButton#Nav:checked {{ background: {ACTIVE_BG}; color: {CYAN}; border-left: 3px solid {CYAN}; padding-left: 11px; }}
QPushButton#Nav {{ qproperty-autoDefault: false; }}
QLineEdit, QComboBox, QSpinBox {{ background: {PANEL_ALT}; border: 1px solid #344563; border-radius: 6px; padding: 8px 10px; min-height: 19px; selection-background-color: #1d5475; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {CYAN}; }}
QComboBox::drop-down {{ border: none; width: 25px; }}
QComboBox QAbstractItemView {{ background: #131f33; selection-background-color: #1b4b68; border: 1px solid #405f87; }}
QSpinBox::up-button, QSpinBox::down-button {{ background: #263d60; border: none; width: 19px; }}
QPlainTextEdit {{ background: #060b14; color: #a8cee9; border: 1px solid {BORDER}; border-radius: 7px; font-family: 'Consolas'; font-size: 11px; padding: 7px; }}
QProgressBar {{ background: #1b2633; border: none; border-radius: 4px; text-align: center; color: {TEXT}; min-height: 8px; max-height: 8px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {BLUE},stop:0.65 {CYAN},stop:1 {GREEN}); border-radius: 4px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 9px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #304154; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 9px; }}
QScrollBar::handle:horizontal {{ background: #304154; border-radius: 4px; min-width: 30px; }}
QTableWidget {{ background: {PANEL}; alternate-background-color: #141f33; border: none; gridline-color: {BORDER}; selection-background-color: #173f60; }}
QTableWidget::item {{ padding: 7px; }}
QHeaderView::section {{ background: #182332; color: {MUTED}; border: none; padding: 9px; font-size: 11px; font-weight: 600; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: #17212e; color: {MUTED}; padding: 9px 15px; margin-right: 4px; border-radius: 5px; }}
QTabBar::tab:selected {{ background: #12334b; color: {CYAN}; border-bottom: 2px solid {CYAN}; }}
QToolTip {{ background: #233447; color: {TEXT}; border: 1px solid #476079; padding: 6px; }}
QCheckBox {{ spacing: 9px; color: {MUTED}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QStatusBar {{ background: {PANEL_ALT}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QMenuBar, QMenu {{ background: {PANEL_ALT}; }}
QMessageBox {{ background: {PANEL}; }}
"""
