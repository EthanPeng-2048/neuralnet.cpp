"""页面基类：运行页（参数表单 + 图表 + 日志）与多工具页。"""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..cli import TOOLS, Tool, build_command, coerce_values
from ..parser import MetricParser
from ..paths import PROJECT_ROOT
from ..runner import ProcessRunner
from ..widgets.chart import ChartStack
from ..widgets.log_pane import LogPanel
from ..widgets.param_form import ParamForm
from ..widgets.run_bar import RunBar


class RunPageBase(QWidget):
    """训练类页面基类：左侧参数表单，右侧运行控制 + 实时图表 + 日志。"""

    tool_key: str = ""
    metric_config: list[dict] = []
    chart_panels: list[dict] = []

    def __init__(self, parent=None):
        self.tool: Tool = TOOLS[self.tool_key]
        self.parser = MetricParser(self.metric_config)
        self.runner = ProcessRunner()
        self._settings = QSettings("neuralnet.cpp", "neuralnet_gui")
        super().__init__(parent)
        self._build_ui()
        self._connect()
        self._load_settings()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        self.form = ParamForm(self.tool, settings_group=f"form/{self.tool_key}")
        self.form.setFixedWidth(330)
        outer.addWidget(self.form)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        self.run_bar = RunBar()
        rv.addWidget(self.run_bar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        if self.chart_panels:
            self.chart = ChartStack(self.chart_panels)
            splitter.addWidget(self.chart)
        else:
            self.chart = None
        self.log = LogPanel()
        splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        rv.addWidget(splitter, 1)
        outer.addWidget(right, 1)

    def _connect(self) -> None:
        self.run_bar.start_requested.connect(self._on_start)
        self.run_bar.stop_requested.connect(self._on_stop)
        self.runner.line.connect(self._on_line)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)

    # ── 事件 ──────────────────────────────────────────────────────
    def _on_start(self) -> None:
        errors = self.form.validate()
        if errors:
            self.run_bar.set_state("idle")
            self.run_bar.set_status_text("；".join(errors))
            return
        values = self.form.values()
        cmd = build_command(self.tool, values)
        self.log.clear()
        self.parser.clear()
        if self.chart is not None:
            self.chart.clear_all()
        self.run_bar.set_state("running")
        self.run_bar.set_status_text("启动中…")
        self.run_bar.start_elapsed()
        self.form.set_enabled_all(False)
        self._save_settings(values)
        self.log.append_line("$ " + " ".join(str(x) for x in cmd))
        self.runner.start(str(cmd[0]), [str(x) for x in cmd[1:]], str(PROJECT_ROOT))

    def _on_stop(self) -> None:
        self.runner.stop()
        self.run_bar.set_state("idle")
        self.run_bar.set_status_text("已停止")
        self.run_bar.stop_elapsed()
        self.log.append_line("■ 已手动停止")

    def _on_line(self, text: str) -> None:
        self.log.append_line(text)
        self._feed_metrics(text)

    def _on_progress(self, text: str) -> None:
        self.log.replace_last(text)
        self._feed_metrics(text)

    def _feed_metrics(self, text: str) -> None:
        if self.chart is None:
            return
        for name in self.parser.feed(text):
            s = self.parser.series[name]
            self.chart.add_point(name, s.x[-1], s.y[-1])

    def _on_finished(self, code: int) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.stop_elapsed()
        if code == 0:
            self.run_bar.set_state("done")
            self.run_bar.set_status_text("运行完成")
        else:
            self.run_bar.set_state("error")
            self.run_bar.set_status_text(f"运行失败（退出码 {code}）")

    def _on_error(self, msg: str) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.set_state("error")
        self.run_bar.set_status_text("启动失败")
        self.run_bar.stop_elapsed()
        self.log.append_line(f"✘ {msg}")

    # ── 持久化 ────────────────────────────────────────────────────
    def _save_settings(self, values: dict) -> None:
        self._settings.beginGroup(f"form/{self.tool_key}")
        for k, v in values.items():
            self._settings.setValue(k, v)
        self._settings.endGroup()

    def _load_settings(self) -> None:
        self._settings.beginGroup(f"form/{self.tool_key}")
        saved = {k: self._settings.value(k) for k in self._settings.childKeys()}
        self._settings.endGroup()
        if saved:
            self.form.set_values(coerce_values(self.tool, saved))

    def apply_palette(self, p: dict) -> None:
        if self.chart is not None:
            self.chart.apply_palette(p)


class MultiToolPage(QWidget):
    """多工具运行页：下拉选择工具 → 参数表单 → 运行 + 日志。"""

    tool_keys: list[str] = []
    start_text: str = "▶  运行"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_key: str = self.tool_keys[0]
        self.runner = ProcessRunner()
        self._settings = QSettings("neuralnet.cpp", "neuralnet_gui")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        left = QWidget()
        left.setFixedWidth(330)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        title = QLabel("工具运行器")
        title.setObjectName("title")
        lv.addWidget(title)

        row = QWidget()
        rh = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        lab = QLabel("选择工具")
        lab.setObjectName("cardTitle")
        rh.addWidget(lab)
        self.combo = QComboBox()
        for k in self.tool_keys:
            self.combo.addItem(TOOLS[k].title, k)
        rh.addWidget(self.combo, 1)
        lv.addWidget(row)

        self.form_holder = QWidget()
        self.form_layout = QVBoxLayout(self.form_holder)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.form_holder, 1)

        self.run_bar = RunBar(self.start_text)
        lv.addWidget(self.run_bar)

        outer.addWidget(left)

        self.log = LogPanel()
        outer.addWidget(self.log, 1)

        self.combo.currentIndexChanged.connect(self._on_tool_changed)
        self.run_bar.start_requested.connect(self._on_start)
        self.run_bar.stop_requested.connect(self._on_stop)
        self.runner.line.connect(self.log.append_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)

        self._rebuild_form()

    # ── 表单重建 ──────────────────────────────────────────────────
    def _on_tool_changed(self) -> None:
        self._current_key = self.combo.currentData()
        self._rebuild_form()

    def _rebuild_form(self) -> None:
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        tool = TOOLS[self._current_key]
        self.form = ParamForm(tool, settings_group=f"form/{self._current_key}")
        self.form_layout.addWidget(self.form)
        self._settings.beginGroup(f"form/{self._current_key}")
        saved = {k: self._settings.value(k) for k in self._settings.childKeys()}
        self._settings.endGroup()
        if saved:
            self.form.set_values(coerce_values(tool, saved))

    # ── 事件 ──────────────────────────────────────────────────────
    def _on_start(self) -> None:
        errors = self.form.validate()
        if errors:
            self.run_bar.set_status_text("；".join(errors))
            return
        tool = TOOLS[self._current_key]
        values = self.form.values()
        cmd = build_command(tool, values)
        self.log.clear()
        self.run_bar.set_state("running")
        self.run_bar.set_status_text("运行中…")
        self.run_bar.start_elapsed()
        self.form.set_enabled_all(False)
        self._settings.beginGroup(f"form/{self._current_key}")
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

    def _on_finished(self, code: int) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.stop_elapsed()
        if code == 0:
            self.run_bar.set_state("done")
            self.run_bar.set_status_text("运行完成")
        else:
            self.run_bar.set_state("error")
            self.run_bar.set_status_text(f"运行失败（退出码 {code}）")

    def _on_error(self, msg: str) -> None:
        self.form.set_enabled_all(True)
        self.run_bar.set_state("error")
        self.run_bar.set_status_text("启动失败")
        self.run_bar.stop_elapsed()
        self.log.append_line(f"✘ {msg}")
