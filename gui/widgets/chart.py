"""实时图表：pyqtgraph 多曲线图 + 多图堆叠。"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

pg.setConfigOptions(antialias=True)


class ChartPane(QWidget):
    """单个图表，可容纳多条曲线。"""

    def __init__(self, title: str, ylabel: str, series: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self._ylabel = ylabel
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("section")
        lay.addWidget(t)

        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True, alpha=0.15)
        self.pw.setLabel("left", ylabel)
        self.pw.setLabel("bottom", "epoch")
        self._legend = self.pw.addLegend(offset=(10, 8))

        self._xs: dict[str, list[float]] = {}
        self._ys: dict[str, list[float]] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        for name, color in series:
            item = self.pw.plot([], [], pen=pg.mkPen(color, width=2))
            self._legend.addItem(item, name)
            self._curves[name] = item
            self._xs[name] = []
            self._ys[name] = []
        lay.addWidget(self.pw, 1)

    def add_point(self, name: str, x: float, y: float) -> None:
        if name not in self._curves:
            return
        self._xs[name].append(x)
        self._ys[name].append(y)
        self._curves[name].setData(self._xs[name], self._ys[name])

    def clear_all(self) -> None:
        for name in self._curves:
            self._xs[name].clear()
            self._ys[name].clear()
            self._curves[name].setData([], [])

    def apply_palette(self, p: dict) -> None:
        self.pw.setBackground(p["chart_bg"])
        for ax in ("left", "bottom"):
            a = self.pw.getAxis(ax)
            a.setPen(pg.mkPen(p["text"]))
            a.setTextPen(pg.mkPen(p["text"]))
        self.pw.getPlotItem().setLabel("left", self._ylabel, color=p["text"])
        try:
            self._legend.setTextPen(pg.mkPen(p["text"]))
            self._legend.setLabelTextColor(pg.mkPen(p["text"]))
        except Exception:
            pass


class ChartStack(QWidget):
    """多个图表纵向堆叠，按系列名分发数据点。"""

    def __init__(self, panels: list[dict], parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self._name_to_panel: dict[str, tuple[int, str]] = {}
        self._panels: list[ChartPane] = []
        for pi, spec in enumerate(panels):
            pane = ChartPane(spec["title"], spec["ylabel"], spec["series"])
            self._panels.append(pane)
            for name, _ in spec["series"]:
                self._name_to_panel[name] = (pi, name)
            lay.addWidget(pane, 1)

    def add_point(self, name: str, x: float, y: float) -> None:
        hit = self._name_to_panel.get(name)
        if hit:
            pi, sname = hit
            self._panels[pi].add_point(sname, x, y)

    def clear_all(self) -> None:
        for pane in self._panels:
            pane.clear_all()

    def apply_palette(self, p: dict) -> None:
        for pane in self._panels:
            pane.apply_palette(p)
