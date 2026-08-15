"""GPU / 验证页：gpu_test 与各验证工具。"""
from __future__ import annotations

from .base import MultiToolPage


class GpuPage(MultiToolPage):
    tool_keys = [
        "gpu_test",
        "attn_consistency_test",
        "swiglu_gradcheck",
        "rmsnorm_gradcheck",
    ]
    start_text = "▶  运行测试"
