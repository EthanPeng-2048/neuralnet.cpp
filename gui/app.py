"""主窗口：左侧导航 + 右侧页面堆栈 + 跟随系统主题。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from .pages.bench_page import BenchPage
from .pages.build_page import BuildPage
from .pages.gpu_page import GpuPage
from .pages.home import HomePage
from .pages.mnist_infer_page import MnistInferPage
from .pages.mnist_train_page import MnistTrainPage
from .pages.text_infer_page import TextInferPage
from .pages.text_train_page import TextTrainPage
from .pages.tokenizer_page import TokenizerPage
from .paths import BUILD_DIR, available_executables
from .theme import build_qss, palette

PAGE_SPECS = [
    ("🏠  首页", HomePage),
    ("🔢  MNIST 训练", MnistTrainPage),
    ("✍️  MNIST 推理", MnistInferPage),
    ("📝  文本训练", TextTrainPage),
    ("💬  文本推理", TextInferPage),
    ("🔠  分词器", TokenizerPage),
    ("⚡  性能基准", BenchPage),
    ("🖥  GPU / 验证", GpuPage),
    ("🔨  构建", BuildPage),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neuralnet.cpp 控制台")
        self.resize(1280, 840)

        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(210)
        for text, _ in PAGE_SPECS:
            self.nav.addItem(text)
        h.addWidget(self.nav)

        self.stack = QStackedWidget()
        self.pages: list[QWidget] = []
        for _, cls in PAGE_SPECS:
            page = cls()
            self.pages.append(page)
            self.stack.addWidget(page)
        h.addWidget(self.stack, 1)

        self.nav.currentRowChanged.connect(self._on_page_changed)
        home = self.pages[0]
        if isinstance(home, HomePage):
            home.page_requested.connect(self.nav.setCurrentRow)
        self.nav.setCurrentRow(0)

        # 主题跟随系统
        QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_color_scheme)

        self.statusBar().showMessage(self._status_text())

    def _status_text(self) -> str:
        exes = available_executables()
        built = "已配置" if (BUILD_DIR / "build.ninja").exists() else "未配置"
        return (
            f"项目: neuralnet.cpp    构建: {built}    "
            f"可执行文件: {len(exes)} 个    "
            f"后端: CPU + Vulkan（CUDA 可选）"
        )

    def _on_page_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        page = self.pages[idx]
        if isinstance(page, HomePage):
            page.refresh()

    def _on_color_scheme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(palette()))
        for p in self.pages:
            if hasattr(p, "apply_palette"):
                p.apply_palette(palette())
