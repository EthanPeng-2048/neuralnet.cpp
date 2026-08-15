"""MNIST 推理页：图片预览 + Top-K 概率条 + 日志。"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..cli import TOOLS, build_command, coerce_values
from ..paths import PROJECT_ROOT
from ..runner import ProcessRunner
from ..theme import palette
from ..widgets.cards import Card
from ..widgets.log_pane import LogPanel
from ..widgets.param_form import ParamForm
from ..widgets.run_bar import RunBar

_TOPK_ITEM = re.compile(r"(\d)\s*\(([\d.]+)%\)")


class TopKBars(QWidget):
    """Top-K 预测概率横条。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)

    def set_results(self, results: list[tuple[int, float]]) -> None:
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        p = palette()
        for digit, conf in results:
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.setSpacing(8)
            lab = QLabel(f"数字 {digit}")
            lab.setObjectName("cardTitle")
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(conf * 10))
            bar.setTextVisible(False)
            val = QLabel(f"{conf:.1f}%")
            val.setObjectName("help")
            rh.addWidget(lab)
            rh.addWidget(bar, 1)
            rh.addWidget(val)
            self._lay.addWidget(row)
        self._lay.addStretch(1)


class MnistInferPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tool = TOOLS["mnist_infer"]
        self.runner = ProcessRunner()
        self._settings = QSettings("neuralnet.cpp", "neuralnet_gui")
        self._p = palette()

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        left = QWidget()
        left.setFixedWidth(330)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.form = ParamForm(self.tool, settings_group="form/mnist_infer")
        lv.addWidget(self.form, 1)
        self.run_bar = RunBar("▶  运行推理")
        lv.addWidget(self.run_bar)
        outer.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Vertical)

        top = QWidget()
        tv = QHBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(10)

        preview_card = Card("输入图片预览")
        self.preview_label = QLabel("选择一张 CSV 图片后自动预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setWordWrap(True)
        preview_card.add_widget(self.preview_label)
        tv.addWidget(preview_card, 1)

        result_card = Card("Top-K 预测")
        self.topk = TopKBars()
        result_card.add_widget(self.topk)
        tv.addWidget(result_card, 1)
        top.setLayout(tv)
        splitter.addWidget(top)

        self.log = LogPanel()
        splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        rv.addWidget(splitter, 1)
        outer.addWidget(right, 1)

        # 事件
        self.run_bar.start_requested.connect(self._on_start)
        self.run_bar.stop_requested.connect(self._on_stop)
        self.runner.line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)

        self._input_edit = self.form.widget("input")
        if self._input_edit is not None:
            self._input_edit.textChanged.connect(self._maybe_preview)
        self._load_settings()

    # ── 运行 ──────────────────────────────────────────────────────
    def _on_start(self) -> None:
        errors = self.form.validate()
        if errors:
            self.run_bar.set_status_text("；".join(errors))
            return
        values = self.form.values()
        cmd = build_command(self.tool, values)
        self.log.clear()
        self.topk.set_results([])
        self.run_bar.set_state("running")
        self.run_bar.set_status_text("推理中…")
        self.run_bar.start_elapsed()
        self.form.set_enabled_all(False)
        self._settings.beginGroup("form/mnist_infer")
        for k, v in values.items():
            self._settings.setValue(k, v)
        self._settings.endGroup()
        self.log.append_line("$ " + " ".join(str(x) for x in cmd))
        self.runner.start(str(cmd[0]), [str(x) for x in cmd[1:]], str(PROJECT_ROOT))

    def _on_stop(self) -> None:
        self.runner.stop()
        self.run_bar.set_state("idle")
        self.run_bar.set_status_text("已停止")
        self.run_bar.stop_elapsed()

    def _on_line(self, text: str) -> None:
        self.log.append_line(text)
        if "->" in text:
            results = parse_topk(text)
            if results:
                self.topk.set_results(results)

    def _on_finished(self, code: int) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.stop_elapsed()
        if code == 0:
            self.run_bar.set_state("done")
            self.run_bar.set_status_text("推理完成")
        else:
            self.run_bar.set_state("error")
            self.run_bar.set_status_text(f"推理失败（退出码 {code}）")

    def _on_error(self, msg: str) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.set_state("error")
        self.run_bar.set_status_text("启动失败")
        self.log.append_line(f"✘ {msg}")

    # ── 预览 ──────────────────────────────────────────────────────
    def _maybe_preview(self, text: str) -> None:
        path = Path(text.strip())
        if path.exists() and path.is_file() and path.suffix.lower() == ".csv":
            self._render_preview(path)
        else:
            self.preview_label.setText("选择一张 CSV 图片后自动预览")
            self.preview_label.setPixmap(QPixmap())

    def _render_preview(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                line = f.readline()
            parts = [float(x) for x in line.strip().split(",") if x.strip()]
            label = None
            vals = parts
            if len(parts) == 785:
                label = int(parts[0])
                vals = parts[1:]
            if len(vals) < 784:
                raise ValueError("CSV 像素不足 784")
            vals = vals[:784]
            if max(vals) > 1.0:
                pixels = [int(max(0, min(255, v))) for v in vals]
            else:
                pixels = [int(max(0, min(255, v * 255))) for v in vals]
            img = Image.new("L", (28, 28))
            img.putdata(pixels)
            target = 260
            img = img.resize((target, target), Image.Resampling.NEAREST)
            data = img.tobytes()
            qimg = QImage(data, target, target, target, QImage.Format.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            title = f"数字: {label}\n" if label is not None else ""
            self.preview_label.setText(title + path.name)
            self.preview_label.setPixmap(pix)
        except Exception as e:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(f"预览失败: {e}")

    # ── 持久化 ────────────────────────────────────────────────────
    def _load_settings(self) -> None:
        self._settings.beginGroup("form/mnist_infer")
        saved = {k: self._settings.value(k) for k in self._settings.childKeys()}
        self._settings.endGroup()
        if saved:
            self.form.set_values(coerce_values(self.tool, saved))


def parse_topk(line: str) -> list[tuple[int, float]]:
    return [(int(m.group(1)), float(m.group(2))) for m in _TOPK_ITEM.finditer(line)]
