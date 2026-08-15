"""指标解析：从 CLI 输出提取训练曲线数据。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 图表系列配色
SERIES_COLORS = ["#4F8CFF", "#3FB950", "#D29922", "#F85149", "#B392F0", "#56D4DD"]


class Series:
    def __init__(self, name: str):
        self.name = name
        self.x: list[float] = []
        self.y: list[float] = []

    def add(self, x: float, y: float) -> None:
        self.x.append(x)
        self.y.append(y)

    def clear(self) -> None:
        self.x.clear()
        self.y.clear()


@dataclass
class _Pattern:
    regex: re.Pattern
    x_kind: str          # "group"（用命名组 x）| "frac"（用 ep/cur/tot 计算小数 epoch）
    series: dict[str, str]  # {系列名: 正则命名组}


class MetricParser:
    """config: list[dict]，
    dict = {regex, x: "group"|"frac", series: {name: group}}。
    feed(text) 返回有更新的系列名列表。
    """

    def __init__(self, config: list[dict]):
        self._patterns: list[_Pattern] = []
        self.series: dict[str, Series] = {}
        for c in config:
            rx = re.compile(c["regex"])
            smap = c["series"]
            for name in smap:
                self.series.setdefault(name, Series(name))
            self._patterns.append(_Pattern(rx, c.get("x", "group"), smap))

    def feed(self, text: str) -> list[str]:
        updated: list[str] = []
        for pat in self._patterns:
            m = pat.regex.search(text)
            if not m:
                continue
            x = self._compute_x(pat, m)
            for name, group in pat.series.items():
                try:
                    y = float(m.group(group))
                except (IndexError, ValueError):
                    continue
                self.series[name].add(x, y)
                updated.append(name)
        return updated

    def _compute_x(self, pat: _Pattern, m: re.Match) -> float:
        if pat.x_kind == "frac":
            try:
                ep = float(m.group("ep"))
                cur = float(m.group("cur"))
                tot = float(m.group("tot"))
            except (IndexError, ValueError):
                return 0.0
            return (ep - 1) + (cur / tot) if tot else ep
        return float(m.group("x"))

    def clear(self) -> None:
        for s in self.series.values():
            s.clear()


# ══════════════════════════════════════════════════════════════════
# 各工具指标正则
# ══════════════════════════════════════════════════════════════════

MNIST_TRAIN_METRICS = [
    # 批级进度：loss 随小数 epoch（连续曲线）
    {
        "regex": r"Epoch\s+(?P<ep>\d+)/(?P<eptot>\d+)\s+batch\s+(?P<cur>\d+)/(?P<tot>\d+)\s+loss:\s+(?P<y>[\d.]+)\s+time:",
        "x": "frac",
        "series": {"loss": "y"},
    },
    # 轮级汇总：loss / train_acc / test_acc
    {
        "regex": r"Epoch\s+(?P<x>\d+)/(?P<eptot>\d+)\s+lr=[\d.eE+-]+\s+loss=(?P<loss>[\d.]+)\s+train_acc=(?P<train_acc>[\d.]+)%\s+test_acc=(?P<test_acc>[\d.]+)%\s+time=",
        "x": "group",
        "series": {"loss": "loss", "train_acc": "train_acc", "test_acc": "test_acc"},
    },
]

TEXT_TRAIN_METRICS = [
    # step 级进度：loss 随小数 epoch
    {
        "regex": r"Epoch\s+(?P<ep>\d+)/(?P<eptot>\d+)\s+step\s+(?P<cur>\d+)/(?P<tot>\d+)\s+loss:\s+(?P<y>[\d.]+)",
        "x": "frac",
        "series": {"loss": "y"},
    },
    # 轮级汇总（无 test-file）
    {
        "regex": r"Epoch\s+(?P<x>\d+)/(?P<eptot>\d+)\s+lr=(?P<lr>[\d.eE+-]+)\s+avg_loss=(?P<loss>[\d.]+)\s+time=",
        "x": "group",
        "series": {"loss": "loss", "lr": "lr"},
    },
    # 轮级汇总（含 test-file）
    {
        "regex": r"Epoch\s+(?P<x>\d+)/(?P<eptot>\d+)\s+lr=(?P<lr>[\d.eE+-]+)\s+avg_loss=(?P<loss>[\d.]+)\s+time=[\d.]+s\s+test_loss=(?P<test_loss>[\d.]+)",
        "x": "group",
        "series": {"loss": "loss", "lr": "lr", "test_loss": "test_loss"},
    },
]

MNIST_INFER_LINE = re.compile(r"->\s+(.+)$")
