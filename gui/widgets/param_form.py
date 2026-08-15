"""参数表单：根据 CLI 工具定义自动生成表单并收集值。"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..cli import Param, Tool


DEFAULT_EXPANDED = {"数据", "训练"}
PAIRABLE = ("int", "float", "choice")  # 可两列配对的参数类型


class ParamForm(QScrollArea):
    def __init__(self, tool: Tool, parent=None, settings_group: str | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        self.setWidget(container)
        self._root = QVBoxLayout(container)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(8)
        self._groups: dict[str, QWidget] = {}         # 组名 → 卡片内容区
        self._headers: dict[str, QPushButton] = {}    # 组名 → 折叠头按钮
        self._widgets: dict[str, QWidget] = {}
        self._param_of: dict[str, Param] = {}
        self._expanded: dict[str, bool] = {}
        self._shadow: dict[str, Any] = {}             # 保留隐藏参数的值
        self._rebuilding = False
        self._settings = (
            QSettings("neuralnet.cpp", "neuralnet_gui") if settings_group else None
        )
        self._settings_group = settings_group
        self._tool = tool
        self.rebuild(tool)

    # ── 构建 ──────────────────────────────────────────────────────
    def rebuild(self, tool: Tool) -> None:
        self._tool = tool
        self._widgets.clear()
        self._param_of.clear()
        self._groups.clear()
        self._headers.clear()
        while self._root.count():
            item = self._root.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        title = QLabel(tool.title)
        title.setObjectName("title")
        self._root.addWidget(title)
        if tool.description:
            desc = QLabel(tool.description)
            desc.setObjectName("subtitle")
            desc.setWordWrap(True)
            self._root.addWidget(desc)

        all_params = list(tool.positional) + list(tool.params)
        defaults = {p.key: p.default for p in all_params}
        merged = {**defaults, **self._shadow}
        visible = self._visible_map(all_params, merged)

        grouped: dict[str, list[Param]] = {}
        order: list[str] = []
        for p in all_params:
            if not visible[p.key]:
                continue
            if p.group not in grouped:
                grouped[p.group] = []
                order.append(p.group)
            grouped[p.group].append(p)

        for g in order:
            self._root.addWidget(self._make_group_card(tool, g, grouped[g]))

        self._connect_deps()
        self._rebuilding = True
        try:
            self.set_values(merged)
        finally:
            self._rebuilding = False
        self._shadow.update(self.values())
        self._root.addStretch(1)

    def _visible_map(self, params: list[Param], values: dict[str, Any]) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for p in params:
            if not p.show_if:
                out[p.key] = True
                continue
            dep_key, allowed = p.show_if
            out[p.key] = str(values.get(dep_key)) in [str(a) for a in allowed]
        return out

    def _make_group_card(self, tool: Tool, g: str, params: list[Param]) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 6, 10, 10)
        lay.setSpacing(8)

        expanded = self._group_expanded(g)
        header = QPushButton(("▼ " if expanded else "▶ ") + g)
        header.setObjectName("groupHeader")
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.clicked.connect(lambda _=False, grp=g: self._toggle_group(grp))
        lay.addWidget(header)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        note = (tool.group_notes or {}).get(g)
        if note:
            nlab = QLabel(note)
            nlab.setObjectName("help")
            nlab.setWordWrap(True)
            bl.addWidget(nlab)
        for unit in self._pair_up(params):
            bl.addWidget(self._make_unit_row(unit))
        lay.addWidget(body)
        body.setVisible(expanded)

        self._headers[g] = header
        self._groups[g] = body
        return card

    def _pair_up(self, params: list[Param]) -> list[tuple[Param, ...]]:
        """数值/下拉参数两两配对，其余独占一行。"""
        units: list[tuple[Param, ...]] = []
        i = 0
        n = len(params)
        while i < n:
            p = params[i]
            if (p.kind in PAIRABLE and i + 1 < n
                    and params[i + 1].kind in PAIRABLE):
                units.append((p, params[i + 1]))
                i += 2
            else:
                units.append((p,))
                i += 1
        return units

    def _make_unit_row(self, unit: tuple[Param, ...]) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        for p in unit:
            cell, value_widget = self._make_cell(p)
            h.addWidget(cell, 1)
            self._widgets[p.key] = value_widget
            self._param_of[p.key] = p
        if len(unit) == 1:
            h.addStretch(1)
        return row

    def _make_cell(self, p: Param) -> tuple[QWidget, QWidget]:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        if p.kind == "bool":
            cb = QCheckBox(p.label or p.key)
            cb.setChecked(bool(p.default))
            if p.help:
                cb.setToolTip(p.help)
            lay.addWidget(cb)
            return wrap, cb
        label = QLabel(p.label or p.key)
        label.setObjectName("cardTitle")
        lay.addWidget(label)
        ctrl, value_widget = self._make_control(p)
        lay.addWidget(ctrl)
        if p.help:
            value_widget.setToolTip(p.help)
        return wrap, value_widget

    # ── 折叠 ──────────────────────────────────────────────────────
    def _default_expanded(self) -> set[str]:
        if self._tool.expanded_groups is not None:
            return set(self._tool.expanded_groups)
        return DEFAULT_EXPANDED

    def _group_expanded(self, g: str) -> bool:
        if g in self._expanded:
            return self._expanded[g]
        saved = None
        if self._settings is not None:
            self._settings.beginGroup(self._settings_group)
            saved = self._settings.value(f"expand/{g}")
            self._settings.endGroup()
        self._expanded[g] = (
            (str(saved) == "1") if saved is not None else (g in self._default_expanded())
        )
        return self._expanded[g]

    def _toggle_group(self, g: str) -> None:
        self._expanded[g] = not self._expanded[g]
        if g in self._headers:
            self._headers[g].setText(("▼ " if self._expanded[g] else "▶ ") + g)
        if g in self._groups:
            self._groups[g].setVisible(self._expanded[g])
        if self._settings is not None:
            self._settings.beginGroup(self._settings_group)
            self._settings.setValue(f"expand/{g}", "1" if self._expanded[g] else "0")
            self._settings.endGroup()

    # ── 动态显隐 ──────────────────────────────────────────────────
    def _connect_deps(self) -> None:
        # 扫描全部参数（含当前隐藏的），否则隐藏参数的依赖控件不会连接
        all_params = list(self._tool.positional) + list(self._tool.params)
        dep_keys = {p.show_if[0] for p in all_params if p.show_if}
        for k in dep_keys:
            w = self._widgets.get(k)
            if w is None:
                continue
            p = self._param_of[k]
            if p.kind == "bool":
                w.toggled.connect(self._on_dep_changed)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._on_dep_changed)
            else:
                w.textChanged.connect(self._on_dep_changed)

    def _on_dep_changed(self, *args) -> None:
        if self._rebuilding:
            return
        # 只接受当前控件的信号（重建后旧控件可能仍被外部持有）
        sender = self.sender()
        if sender is not None and all(sender is not w for w in self._widgets.values()):
            return
        self._shadow.update(self.values())
        self.rebuild(self._tool)

    def _make_control(self, p: Param) -> tuple[QWidget, QWidget]:
        """返回 (加入布局的控件, 用于取值的控件)。"""
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.min) if p.min is not None else -2**30,
                       int(p.max) if p.max is not None else 2**30)
            w.setValue(int(p.default or 0))
            return w, w
        if p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(p.min if p.min is not None else -1e7,
                       p.max if p.max is not None else 1e7)
            w.setSingleStep(p.step if p.step is not None else 0.1)
            w.setValue(float(p.default or 0.0))
            return w, w
        if p.kind in ("choice", "engine"):
            w = QComboBox()
            w.addItems(p.choices or [])
            if p.default:
                idx = w.findText(str(p.default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            return w, w
        if p.kind == "text":
            w = QPlainTextEdit()
            w.setFixedHeight(76)
            w.setPlainText(str(p.default or ""))
            return w, w
        # str（可带浏览按钮）
        edit = QLineEdit(str(p.default or ""))
        if p.browse:
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            btn = QToolButton()
            btn.setText("…")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda: self._browse(p, edit))
            h.addWidget(edit, 1)
            h.addWidget(btn)
            return holder, edit
        return edit, edit

    def _browse(self, p: Param, edit: QLineEdit) -> None:
        cur = edit.text()
        if p.browse == "dir":
            path = QFileDialog.getExistingDirectory(self, f"选择目录", cur or "")
        elif p.browse == "save":
            path, _ = QFileDialog.getSaveFileName(self, f"保存到", cur or "model.bin",
                                                  "模型文件 (*.bin)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"选择文件", cur or "",
                                                  "所有文件 (*.*)")
        if path:
            edit.setText(path)

    # ── 读写 ──────────────────────────────────────────────────────
    def widget(self, key: str) -> QWidget | None:
        return self._widgets.get(key)

    def values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, w in self._widgets.items():
            p = self._param_of[key]
            if p.kind == "bool":
                out[key] = bool(w.isChecked())
            elif p.kind == "int":
                out[key] = int(w.value())
            elif p.kind == "float":
                out[key] = float(w.value())
            elif p.kind == "choice":
                out[key] = w.currentText()
            elif p.kind == "text":
                out[key] = w.toPlainText().strip()
            else:
                out[key] = w.text().strip()
        return out

    def set_values(self, d: dict[str, Any]) -> None:
        for key, w in self._widgets.items():
            p = self._param_of[key]
            if key not in d:
                continue
            v = d[key]
            try:
                if p.kind == "bool":
                    w.setChecked(bool(v))
                elif p.kind == "int":
                    w.setValue(int(v))
                elif p.kind == "float":
                    w.setValue(float(v))
                elif p.kind == "choice":
                    idx = w.findText(str(v))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                elif p.kind == "text":
                    w.setPlainText(str(v))
                else:
                    w.setText(str(v))
            except Exception:
                pass

    def validate(self) -> list[str]:
        errors: list[str] = []
        for key, p in self._param_of.items():
            if not p.required:
                continue
            w = self._widgets[key]
            text = w.toPlainText().strip() if p.kind == "text" else w.text().strip()
            if not text:
                errors.append(f"「{p.label or p.key}」为必填项")
        return errors

    def set_enabled_all(self, enabled: bool) -> None:
        for w in self._widgets.values():
            w.setEnabled(enabled)
