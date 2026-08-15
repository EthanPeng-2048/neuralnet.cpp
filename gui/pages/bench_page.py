"""性能基准页：compute_bench / bench_thresholds。"""
from __future__ import annotations

from .base import MultiToolPage


class BenchPage(MultiToolPage):
    tool_keys = ["compute_bench", "bench_thresholds"]
    start_text = "▶  运行基准"
