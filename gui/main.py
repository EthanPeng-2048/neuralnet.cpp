"""GUI 入口。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m gui.main
"""
from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from .app import MainWindow
from .theme import build_qss, palette


def main() -> int:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setApplicationName("neuralnet.cpp 控制台")
    app.setOrganizationName("neuralnet.cpp")
    app.setStyleSheet(build_qss(palette()))

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
