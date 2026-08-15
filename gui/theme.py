"""主题管理：跟随系统深浅色，统一调色板 + QSS。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

DARK = {
    "window": "#1E1F24",
    "panel": "#26282E",
    "card": "#2A2C33",
    "card_alt": "#23252B",
    "border": "#3A3D46",
    "border_hover": "#4A4D56",
    "text": "#E1E3E8",
    "subtext": "#9AA0AA",
    "accent": "#4F8CFF",
    "accent_hover": "#6BA0FF",
    "accent_pressed": "#3B74D8",
    "success": "#3FB950",
    "warn": "#D29922",
    "error": "#F85149",
    "log_bg": "#16171B",
    "input": "#23252B",
    "input_hover": "#2C2F37",
    "input_pressed": "#2E3037",
    "input_border": "#3A3D46",
    "nav_bg": "#1A1B1F",
    "nav_active": "#2A3A55",
    "nav_text": "#9AA0AA",
    "nav_active_text": "#FFFFFF",
    "chart_bg": "#1E1F24",
    "chart_grid": "#33353D",
}

LIGHT = {
    "window": "#F4F5F7",
    "panel": "#FFFFFF",
    "card": "#FFFFFF",
    "card_alt": "#F6F7F9",
    "border": "#D8DBE1",
    "border_hover": "#C6CBD4",
    "text": "#24292F",
    "subtext": "#6B7280",
    "accent": "#2F6FEB",
    "accent_hover": "#245BC8",
    "accent_pressed": "#1F4FAD",
    "success": "#1A7F37",
    "warn": "#9A6700",
    "error": "#CF222E",
    "log_bg": "#FAFBFC",
    "input": "#FFFFFF",
    "input_hover": "#EFF1F4",
    "input_pressed": "#E4E7EB",
    "input_border": "#D0D3D9",
    "nav_bg": "#E8EAEE",
    "nav_active": "#D7E2F8",
    "nav_text": "#5B6472",
    "nav_active_text": "#0D2C66",
    "chart_bg": "#FFFFFF",
    "chart_grid": "#E3E6EB",
}

# 图表系列配色（深浅主题通用）
SERIES_COLORS = ["#4F8CFF", "#3FB950", "#D29922", "#F85149", "#B392F0", "#56D4DD"]


def is_dark_mode() -> bool:
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        return False


def palette(dark: bool | None = None) -> dict:
    if dark is None:
        dark = is_dark_mode()
    return DARK if dark else LIGHT


def build_qss(p: dict) -> str:
    css = """
* { outline: none; }
QWidget { background-color: @window@; color: @text@; font-size: 13px; }
QMainWindow, QDialog { background-color: @window@; }

QLabel { background: transparent; }
QLabel#title { font-size: 21px; font-weight: 700; color: @text@; }
QLabel#subtitle { font-size: 13px; color: @subtext@; }
QLabel#section { font-size: 12px; font-weight: 700; color: @accent@; padding: 6px 2px; }
QLabel#cardTitle { font-size: 12px; color: @subtext@; }
QLabel#cardValue { font-size: 24px; font-weight: 700; color: @text@; }
QLabel#statValue { font-size: 30px; font-weight: 800; }
QLabel#help { font-size: 11px; color: @subtext@; }

QFrame#card {
    background-color: @card@;
    border: 1px solid @border@;
    border-radius: 10px;
}

QPushButton {
    background-color: @input@;
    color: @text@;
    border: 1px solid @input_border@;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover { background-color: @input_hover@; border-color: @border_hover@; }
QPushButton:pressed { background-color: @input_pressed@; }
QPushButton:disabled { color: @subtext@; background-color: @input@; border-color: @input_border@; }

QPushButton#primary {
    background-color: @accent@; border: none; color: #FFFFFF; font-weight: 600; padding: 8px 22px;
}
QPushButton#primary:hover { background-color: @accent_hover@; }
QPushButton#primary:pressed { background-color: @accent_pressed@; }
QPushButton#primary:disabled { background-color: @input@; color: @subtext@; }

QPushButton#danger {
    background-color: transparent; border: 1px solid @error@; color: @error@; padding: 8px 22px; font-weight: 600;
}
QPushButton#danger:hover { background-color: @input_hover@; }
QPushButton#danger:disabled { color: @subtext@; border-color: @input_border@; }

QPushButton#ghost { background-color: transparent; border: 1px solid @input_border@; color: @text@; }
QPushButton#ghost:hover { background-color: @input_hover@; }

QPushButton#groupHeader {
    background: transparent; border: none; text-align: left;
    font-size: 13px; font-weight: 700; color: @text@; padding: 4px 2px;
}
QPushButton#groupHeader:hover { color: @accent@; }

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: @input@;
    color: @text@;
    border: 1px solid @input_border@;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: @accent@;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid @accent@; }
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: @subtext@; }

QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: @input@; color: @text@; border: 1px solid @input_border@;
    selection-background-color: @accent@; selection-color: #FFFFFF; border-radius: 4px;
}

QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid @input_border@; background: @input@; }
QCheckBox::indicator:checked { background-color: @accent@; border-color: @accent@; }

QListWidget#nav {
    background-color: @nav_bg@; border: none; font-size: 14px;
}
QListWidget#nav::item {
    padding: 10px 14px; margin: 2px 8px; border-radius: 7px; color: @nav_text@;
}
QListWidget#nav::item:hover { background-color: @input@; color: @text@; }
QListWidget#nav::item:selected { background-color: @nav_active@; color: @nav_active_text@; }

QTabWidget::pane { border: 1px solid @border@; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: transparent; color: @subtext@; padding: 8px 20px; border: none;
    border-bottom: 2px solid transparent; font-size: 13px;
}
QTabBar::tab:selected { color: @accent@; border-bottom: 2px solid @accent@; }
QTabBar::tab:hover { color: @text@; }

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: @border@; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: @subtext@; }
QScrollBar:horizontal { background: transparent; height: 10px; }
QScrollBar::handle:horizontal { background: @border@; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSplitter::handle { background: @border@; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }

QStatusBar { background-color: @panel@; color: @subtext@; border-top: 1px solid @border@; }
QStatusBar::item { border: none; }

QToolTip { background-color: @panel@; color: @text@; border: 1px solid @border@; }

QProgressBar {
    background-color: @input@; border: 1px solid @input_border@; border-radius: 5px;
    text-align: center; color: @text@; height: 16px;
}
QProgressBar::chunk { background-color: @accent@; border-radius: 4px; }

QTableWidget, QTableView {
    background-color: @card@; color: @text@; border: 1px solid @border@; border-radius: 6px; gridline-color: @border@;
}
QHeaderView::section {
    background-color: @card_alt@; color: @subtext@; border: none;
    border-bottom: 1px solid @border@; padding: 6px; font-weight: 600;
}
QTableWidget::item:selected { background-color: @nav_active@; color: @text@; }

QMenu { background-color: @panel@; color: @text@; border: 1px solid @border@; border-radius: 6px; }
QMenu::item { padding: 6px 20px; }
QMenu::item:selected { background-color: @nav_active@; color: @text@; }

QPlainTextEdit#logPane {
    background-color: @log_bg@; border: 1px solid @border@; border-radius: 8px;
    font-family: "Cascadia Mono", Consolas, "Courier New", monospace; font-size: 12px;
    padding: 6px;
}
QPlainTextEdit#resultPane {
    background-color: @log_bg@; border: 1px solid @border@; border-radius: 8px; font-size: 13px; padding: 8px;
}
"""
    for k, v in p.items():
        css = css.replace("@" + k + "@", v)
    return css
