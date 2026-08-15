"""日志面板：终端风格只读日志 + 头部工具条（自动滚动/清空/保存）。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        head = QWidget()
        hh = QHBoxLayout(head)
        hh.setContentsMargins(0, 0, 0, 0)
        title = QLabel("运行日志")
        title.setObjectName("section")
        hh.addWidget(title)
        hh.addStretch(1)
        self._autoscroll = QCheckBox("自动滚动")
        self._autoscroll.setChecked(True)
        hh.addWidget(self._autoscroll)
        self._clear_btn = QPushButton("清空")
        self._clear_btn.setObjectName("ghost")
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hh.addWidget(self._clear_btn)
        self._save_btn = QPushButton("保存日志")
        self._save_btn.setObjectName("ghost")
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hh.addWidget(self._save_btn)
        v.addWidget(head)

        self._pane = QPlainTextEdit()
        self._pane.setObjectName("logPane")
        self._pane.setReadOnly(True)
        self._pane.setMaximumBlockCount(5000)
        font = QFont("Cascadia Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._pane.setFont(font)

        self._clear_btn.clicked.connect(self.clear)
        self._save_btn.clicked.connect(self._save_to_file)
        v.addWidget(self._pane, 1)

    def append_line(self, text: str) -> None:
        if self._autoscroll.isChecked():
            self._pane.appendPlainText(text)
        else:
            cur = self._pane.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.End)
            self._pane.setTextCursor(cur)
            self._pane.insertPlainText(text + "\n")

    def replace_last(self, text: str) -> None:
        """用 \r 覆盖行替换当前最后一行（终端风格）。"""
        doc = self._pane.document()
        if doc.blockCount() == 0:
            self.append_line(text)
            return
        cur = self._pane.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.movePosition(
            QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor
        )
        cur.insertText(text)

    def clear(self) -> None:
        self._pane.clear()

    def to_text(self) -> str:
        return self._pane.toPlainText()

    def _save_to_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "run_log.txt", "文本文件 (*.txt)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._pane.toPlainText())
