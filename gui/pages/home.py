"""首页：项目总览、统计卡片、快速入口、模型/数据集列表。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..paths import (
    BUILD_DIR,
    DATASETS_DIR,
    PRETRAINED_DIR,
    available_executables,
    scan_datasets,
    scan_models,
)
from ..theme import palette
from ..widgets.cards import Card, Row, StatCard

# (显示文本, 页面索引) 与 app.py 的导航顺序一致
QUICK_LINKS = [
    ("🔢  MNIST 训练", 1),
    ("✍️  MNIST 推理", 2),
    ("📝  文本训练", 3),
    ("💬  文本推理", 4),
    ("🔠  分词器", 5),
    ("⚡  性能基准", 6),
    ("🖥  GPU / 验证", 7),
    ("🔨  构建", 8),
]


class HomePage(QWidget):
    page_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._p = palette()
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setContentsMargins(4, 0, 4, 0)
        bl = QVBoxLayout(body)
        bl.setSpacing(14)

        # 标题
        title = QLabel("neuralnet.cpp 控制台")
        title.setObjectName("title")
        bl.addWidget(title)
        sub = QLabel("基于 PySide6 的 CLI 图形化控制台 —— 训练 / 推理 / 分词器 / 基准 / 构建")
        sub.setObjectName("subtitle")
        bl.addWidget(sub)

        # 统计卡片
        exes = available_executables()
        models = scan_models()
        datasets = scan_datasets()
        stats = Row()
        c1 = StatCard("已构建工具", self._p["accent"])
        c1.set_value(str(len(exes)))
        c2 = StatCard("模型文件", self._p["success"])
        c2.set_value(str(len(models)))
        c3 = StatCard("数据集", self._p["warn"])
        c3.set_value(str(len(datasets)))
        stats.add(c1, 1)
        stats.add(c2, 1)
        stats.add(c3, 1)
        bl.addWidget(stats)

        # 快速入口
        quick = Card("快速入口")
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, (text, idx) in enumerate(QUICK_LINKS):
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, i=idx: self.page_requested.emit(i))
            grid.addWidget(btn, i // 4, i % 4)
        quick.add_widget(grid)
        bl.addWidget(quick)

        # 模型 / 数据集
        two = Row()
        model_card = Card("可用模型（.bin）")
        self._model_list = QListWidget()
        self._model_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for path, label in models.items():
            self._model_list.addItem(f"{label}  →  {path}")
        model_card.add_widget(self._model_list)
        two.add(model_card, 1)

        ds_card = Card("可用数据集")
        self._ds_list = QListWidget()
        self._ds_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for p in datasets:
            self._ds_list.addItem(str(p))
        ds_card.add_widget(self._ds_list)
        two.add(ds_card, 1)
        bl.addWidget(two)

        # 构建状态
        status_card = Card("构建状态")
        info = QLabel(
            f"构建目录: {BUILD_DIR}\n"
            f"是否已配置: {'✔ 是' if (BUILD_DIR / 'build.ninja').exists() else '✘ 否'}\n"
            f"可执行文件: {', '.join(exes) if exes else '（无，请到「构建」页构建）'}"
        )
        info.setWordWrap(True)
        info.setObjectName("subtitle")
        status_card.add_widget(info)
        bl.addWidget(status_card)

        bl.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)

    def refresh(self) -> None:
        """切回首页时刷新统计。"""
        exes = available_executables()
        models = scan_models()
        datasets = scan_datasets()
        cards = self.findChildren(StatCard)
        if len(cards) >= 3:
            cards[0].set_value(str(len(exes)))
            cards[1].set_value(str(len(models)))
            cards[2].set_value(str(len(datasets)))
        self._model_list.clear()
        for path, label in models.items():
            self._model_list.addItem(f"{label}  →  {path}")
        self._ds_list.clear()
        for p in datasets:
            self._ds_list.addItem(str(p))
