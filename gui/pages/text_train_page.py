"""文本训练（GPT）页。"""
from __future__ import annotations

from ..parser import TEXT_TRAIN_METRICS
from ..theme import SERIES_COLORS
from .base import RunPageBase


class TextTrainPage(RunPageBase):
    tool_key = "text_train"
    metric_config = TEXT_TRAIN_METRICS
    chart_panels = [
        {
            "title": "损失 Loss（step 级 + 轮级）",
            "ylabel": "loss",
            "series": [
                ("loss", SERIES_COLORS[0]),
                ("test_loss", SERIES_COLORS[2]),
            ],
        },
        {
            "title": "学习率 Learning Rate",
            "ylabel": "lr",
            "series": [("lr", SERIES_COLORS[4])],
        },
    ]
