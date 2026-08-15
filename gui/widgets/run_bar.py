"""运行控制栏：开始/停止 + 状态指示点 + 计时。"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

STATE_COLORS = {
    "idle": "#9AA0AA",
    "running": "#4F8CFF",
    "done": "#3FB950",
    "error": "#F85149",
}
STATE_TEXT = {
    "idle": "空闲",
    "running": "运行中",
    "done": "完成",
    "error": "失败",
}


class RunBar(QWidget):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, start_text: str = "▶  开始运行", parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)

        self.start_btn = QPushButton(start_text)
        self.start_btn.setObjectName("primary")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn = QPushButton("■  停止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        h.addWidget(self.start_btn)
        h.addWidget(self.stop_btn)
        h.addSpacing(8)

        self.dot = QLabel()
        self.dot.setFixedSize(12, 12)
        h.addWidget(self.dot)
        self.status = QLabel("空闲")
        h.addWidget(self.status)
        h.addStretch(1)
        self.elapsed = QLabel("00:00")
        h.addWidget(self.elapsed)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._elapsed_s = 0
        self.set_state("idle")

    def set_state(self, state: str) -> None:
        color = STATE_COLORS.get(state, "#9AA0AA")
        self.dot.setStyleSheet(
            f"background-color: {color}; border-radius: 6px;"
        )
        self.status.setText(STATE_TEXT.get(state, state))
        if state == "running":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def set_status_text(self, text: str) -> None:
        self.status.setText(text)

    def start_elapsed(self) -> None:
        self._elapsed_s = 0
        self.elapsed.setText("00:00")
        self._timer.start()

    def stop_elapsed(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        self._elapsed_s += 1
        m, s = divmod(self._elapsed_s, 60)
        h, m = divmod(m, 60)
        self.elapsed.setText(f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")
