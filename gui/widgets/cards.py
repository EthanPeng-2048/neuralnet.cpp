"""通用卡片与统计组件。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class StatCard(QFrame):
    """带图标色的大数字统计卡。"""

    def __init__(self, title: str, accent: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        v.addWidget(t)
        self._value = QLabel("—")
        self._value.setObjectName("statValue")
        self._value.setStyleSheet(f"color: {accent};")
        v.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class Card(QFrame):
    """带标题的内容卡。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(8)
        t = QLabel(title)
        t.setObjectName("section")
        v.addWidget(t)
        self._body = QVBoxLayout()
        v.addLayout(self._body)

    def add_widget(self, w) -> None:
        if isinstance(w, QWidget):
            self._body.addWidget(w)
        else:  # QLayout
            self._body.addLayout(w)


class Row(QWidget):
    """水平排列小工具。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)

    def add(self, w: QWidget, stretch: int = 0) -> None:
        self._lay.addWidget(w, stretch)

    def add_stretch(self) -> None:
        self._lay.addStretch(1)
