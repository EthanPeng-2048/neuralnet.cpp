"""构建页：CMake 配置 / 构建 / 清理，支持构建选项选择。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..paths import PROJECT_ROOT
from ..runner import ProcessRunner
from ..widgets.cards import Card
from ..widgets.log_pane import LogPanel

PHASE = {"idle": "空闲", "configure": "配置中…", "build": "构建中…", "clean": "清理中…"}


class BuildPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner = ProcessRunner()
        self._phase = "idle"
        self._queue: list[list[str]] = []  # 顺序执行队列

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        left = QWidget()
        left.setFixedWidth(340)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        card = Card("构建选项")
        self.build_dir = QLineEdit("build")
        card.add_widget(self._field("构建目录", self.build_dir))

        self.build_type = QComboBox()
        self.build_type.addItems(["Release", "Debug"])
        card.add_widget(self._field("构建类型", self.build_type))

        self.cuda_cb = QCheckBox("启用 CUDA 加速（NN_ENABLE_CUDA）")
        self.tests_cb = QCheckBox("构建单元测试（NN_BUILD_TESTS）")
        card.add_widget(self.cuda_cb)
        card.add_widget(self.tests_cb)

        note = QLabel("注：Vulkan 由 CMake 自动检测（需要 Vulkan SDK + glslc）。")
        note.setObjectName("help")
        note.setWordWrap(True)
        card.add_widget(note)

        row = QWidget()
        rh = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        self.configure_btn = QPushButton("配置")
        self.build_btn = QPushButton("构建")
        self.both_btn = QPushButton("配置 + 构建")
        self.clean_btn = QPushButton("清理")
        for b in (self.configure_btn, self.build_btn, self.both_btn, self.clean_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.configure_btn.clicked.connect(lambda: self._run_phase("configure"))
        self.build_btn.clicked.connect(lambda: self._run_phase("build"))
        self.both_btn.clicked.connect(lambda: self._run_phase("both"))
        self.clean_btn.clicked.connect(lambda: self._run_phase("clean"))
        rh.addWidget(self.configure_btn)
        rh.addWidget(self.build_btn)
        rh.addWidget(self.clean_btn)
        card.add_widget(row)
        lv.addWidget(card)

        self.both_btn.setObjectName("primary")
        self.status = QLabel("空闲")
        lv.addWidget(self.status)
        lv.addStretch(1)
        outer.addWidget(left)

        self.log = LogPanel()
        outer.addWidget(self.log, 1)

        self.runner.line.connect(self.log.append_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)

    def _field(self, label: str, w: QWidget) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lab = QLabel(label)
        lab.setObjectName("cardTitle")
        lay.addWidget(lab)
        lay.addWidget(w)
        return wrap

    # ── 命令构建 ──────────────────────────────────────────────────
    def _build_dir(self) -> str:
        return self.build_dir.text().strip() or "build"

    def _configure_cmd(self) -> list[str]:
        cmd = ["-B", self._build_dir(), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=" + self.build_type.currentText()]
        if self.cuda_cb.isChecked():
            cmd.append("-DNN_ENABLE_CUDA=ON")
        if self.tests_cb.isChecked():
            cmd.append("-DNN_BUILD_TESTS=ON")
        return cmd

    def _run_phase(self, phase: str) -> None:
        if self.runner.running:
            return
        if phase == "configure":
            self._queue = [["cmake"] + self._configure_cmd()]
        elif phase == "build":
            self._queue = [["cmake", "--build", self._build_dir(), "--parallel"]]
        elif phase == "clean":
            self._queue = [["cmake", "--build", self._build_dir(), "--target", "clean"]]
        else:  # both
            self._queue = [
                ["cmake"] + self._configure_cmd(),
                ["cmake", "--build", self._build_dir(), "--parallel"],
            ]
        self.log.clear()
        self._run_next()

    def _run_next(self) -> None:
        if not self._queue:
            self._set_phase("idle")
            return
        args = self._queue.pop(0)
        self._set_phase("build" if args[0] == "cmake" and args[1] == "--build" else "configure")
        self.log.append_line("$ " + " ".join(args))
        self.runner.start("cmake", args[1:], str(PROJECT_ROOT))

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self.status.setText(PHASE.get(phase, phase))
        busy = phase != "idle"
        for b in (self.configure_btn, self.build_btn, self.both_btn, self.clean_btn):
            b.setEnabled(not busy)

    def _on_finished(self, code: int) -> None:
        if code == 0:
            self.log.append_line("✔ 完成")
        else:
            self.log.append_line(f"✘ 失败（退出码 {code}）")
            self._queue.clear()
            self._set_phase("idle")
            self.status.setText("构建失败")
            return
        self._run_next()

    def _on_error(self, msg: str) -> None:
        self._queue.clear()
        self._set_phase("idle")
        self.status.setText("启动失败")
        self.log.append_line(f"✘ {msg}")
