"""MNIST 训练页。"""
from __future__ import annotations

from ..parser import MNIST_TRAIN_METRICS
from ..theme import SERIES_COLORS
from .base import RunPageBase


class MnistTrainPage(RunPageBase):
    tool_key = "mnist_train"
    metric_config = MNIST_TRAIN_METRICS
    chart_panels = [
        {
            "title": "损失 Loss（批级 + 轮级）",
            "ylabel": "loss",
            "series": [("loss", SERIES_COLORS[0])],
        },
        {
            "title": "准确率 Accuracy",
            "ylabel": "accuracy (%)",
            "series": [
                ("train_acc", SERIES_COLORS[1]),
                ("test_acc", SERIES_COLORS[2]),
            ],
        },
    ]
