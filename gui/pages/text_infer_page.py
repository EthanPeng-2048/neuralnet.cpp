"""文本推理（GPT 生成）页：提示词输入 + 生成结果展示。"""
from __future__ import annotations

import re
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..cli import TOOLS, build_command, coerce_values
from ..paths import PROJECT_ROOT
from ..runner import ProcessRunner
from ..widgets.cards import Card
from ..widgets.log_pane import LogPanel
from ..widgets.param_form import ParamForm
from ..widgets.run_bar import RunBar

_SEPARATOR = re.compile(r"^-{5,}$")
_PROMPT_TOKENS = re.compile(r"Prompt tokens:\s*(\d+)\s*个")


class TextInferPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tool = TOOLS["text_infer"]
        self.runner = ProcessRunner()
        self._settings = QSettings("neuralnet.cpp", "neuralnet_gui")
        self._out_lines: list[str] = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        left = QWidget()
        left.setFixedWidth(330)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.form = ParamForm(self.tool, settings_group="form/text_infer")
        lv.addWidget(self.form, 1)
        self.run_bar = RunBar("▶  生成")
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

        prompt_card = Card("提示词 (Prompt)")
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("输入提示文本，例如：Once upon a time")
        self.prompt.setFixedHeight(110)
        prompt_card.add_widget(self.prompt)
        tv.addWidget(prompt_card, 1)

        result_card = Card("生成结果")
        self.result = QPlainTextEdit()
        self.result.setObjectName("resultPane")
        self.result.setReadOnly(True)
        result_card.add_widget(self.result)
        tv.addWidget(result_card, 2)
        top.setLayout(tv)
        splitter.addWidget(top)

        self.log = LogPanel()
        splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        rv.addWidget(splitter, 1)
        outer.addWidget(right, 1)

        self.run_bar.start_requested.connect(self._on_start)
        self.run_bar.stop_requested.connect(self._on_stop)
        self.runner.line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)
        self._load_settings()

    # ── 运行 ──────────────────────────────────────────────────────
    def _on_start(self) -> None:
        errors = self.form.validate()
        if errors:
            self.run_bar.set_status_text("；".join(errors))
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            self.run_bar.set_status_text("请输入提示词")
            return
        values = self.form.values()
        cmd = build_command(self.tool, values)
        cmd += ["--prompt", prompt]
        self.log.clear()
        self.result.clear()
        self._out_lines = []
        self.run_bar.set_state("running")
        self.run_bar.set_status_text("生成中…")
        self.run_bar.start_elapsed()
        self.form.set_enabled_all(False)
        self._settings.beginGroup("form/text_infer")
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
        self._out_lines.append(text)

    def _on_finished(self, code: int) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.stop_elapsed()
        if code == 0:
            self.run_bar.set_state("done")
            self.run_bar.set_status_text("生成完成")
            self._extract_result()
        else:
            self.run_bar.set_state("error")
            self.run_bar.set_status_text(f"生成失败（退出码 {code}）")

    def _on_error(self, msg: str) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.set_state("error")
        self.run_bar.set_status_text("启动失败")
        self.log.append_line(f"✘ {msg}")

    def _extract_result(self) -> None:
        """提取分隔线（----）之后的生成文本。"""
        lines = self._out_lines
        sep_idx = None
        for i, ln in enumerate(lines):
            if _SEPARATOR.match(ln.strip()):
                sep_idx = i
                break
        if sep_idx is None:
            self.result.setPlainText("\n".join(lines))
            return
        body = lines[sep_idx + 1:]
        prompt = self.prompt.toPlainText().strip()
        text = "\n".join(body).strip()
        if prompt and text.startswith(prompt):
            text = text[len(prompt):]
        self.result.setPlainText(text if text else "（无输出）")

    # ── 持久化 ────────────────────────────────────────────────────
    def _load_settings(self) -> None:
        self._settings.beginGroup("form/text_infer")
        saved = {k: self._settings.value(k) for k in self._settings.childKeys()}
        self._settings.endGroup()
        if saved:
            self.form.set_values(coerce_values(self.tool, saved))
