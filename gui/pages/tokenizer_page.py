"""分词器页：训练子页 + 编解码工具子页。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..cli import TOOLS
from ..paths import PROJECT_ROOT, exe_path
from ..runner import ProcessRunner
from ..widgets.cards import Card
from ..widgets.param_form import ParamForm
from .base import RunPageBase


class TokenizerTrainPage(RunPageBase):
    tool_key = "tokenizer_train"


class TokenizerToolsPage(QWidget):
    """编码 / 解码工具：调用 tokenizer_infer --encode / --decode。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tool = TOOLS["tokenizer_infer"]
        self.runner = ProcessRunner()
        self._target = "encode"

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        left = QWidget()
        left.setFixedWidth(340)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        vocab_card = Card("词表")
        row = QWidget()
        rh = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        self.vocab = QLineEdit("bpe_vocab.json")
        browse = QPushButton("…")
        browse.setFixedWidth(34)
        browse.clicked.connect(self._browse_vocab)
        rh.addWidget(self.vocab, 1)
        rh.addWidget(browse)
        vocab_card.add_widget(row)
        lv.addWidget(vocab_card)

        enc_card = Card("编码 Encoder")
        self.enc_input = QPlainTextEdit()
        self.enc_input.setPlaceholderText("输入文本，编码为 token IDs")
        self.enc_input.setFixedHeight(110)
        enc_card.add_widget(self.enc_input)
        self.enc_btn = QPushButton("编码 →")
        self.enc_btn.setObjectName("primary")
        self.enc_btn.clicked.connect(lambda: self._run("encode"))
        enc_card.add_widget(self.enc_btn)
        lv.addWidget(enc_card)

        dec_card = Card("解码 Decoder")
        self.dec_input = QLineEdit()
        self.dec_input.setPlaceholderText("逗号分隔的 token IDs，如 12,34,567")
        dec_card.add_widget(self.dec_input)
        self.dec_btn = QPushButton("→ 解码")
        self.dec_btn.setObjectName("primary")
        self.dec_btn.clicked.connect(lambda: self._run("decode"))
        dec_card.add_widget(self.dec_btn)
        lv.addWidget(dec_card)

        self.status = QLabel("空闲")
        lv.addWidget(self.status)
        lv.addStretch(1)
        outer.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        enc_res = Card("编码结果")
        self.enc_result = QPlainTextEdit()
        self.enc_result.setObjectName("resultPane")
        self.enc_result.setReadOnly(True)
        enc_res.add_widget(self.enc_result)
        rv.addWidget(enc_res, 1)
        dec_res = Card("解码结果")
        self.dec_result = QPlainTextEdit()
        self.dec_result.setObjectName("resultPane")
        self.dec_result.setReadOnly(True)
        dec_res.add_widget(self.dec_result)
        rv.addWidget(dec_res, 1)
        outer.addWidget(right, 1)

        self.runner.line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_error)

    def _browse_vocab(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "选择词表 JSON", "",
                                              "JSON 词表 (*.json)")
        if path:
            self.vocab.setText(path)

    def _run(self, target: str) -> None:
        vocab = self.vocab.text().strip()
        if not vocab:
            self.status.setText("请先填写词表路径")
            return
        self._target = target
        if target == "encode":
            text = self.enc_input.toPlainText().strip()
            if not text:
                self.status.setText("请输入要编码的文本")
                return
            args = ["--vocab", vocab, "--encode", text]
        else:
            ids = self.dec_input.text().strip()
            if not ids:
                self.status.setText("请输入要解码的 token IDs")
                return
            args = ["--vocab", vocab, "--decode", ids]

        self.status.setText("运行中…")
        self.enc_btn.setEnabled(False)
        self.dec_btn.setEnabled(False)
        self.runner.start(
            str(exe_path("tokenizer_infer")),
            args,
            str(PROJECT_ROOT),
        )

    def _on_line(self, text: str) -> None:
        if self._target == "encode":
            self.enc_result.appendPlainText(text)
        else:
            self.dec_result.appendPlainText(text)

    def _on_finished(self, code: int) -> None:
        self.enc_btn.setEnabled(True)
        self.dec_btn.setEnabled(True)
        self.status.setText("完成 ✔" if code == 0 else f"失败（退出码 {code}）")

    def _on_error(self, msg: str) -> None:
        self.enc_btn.setEnabled(True)
        self.dec_btn.setEnabled(True)
        self.status.setText("启动失败")
        if self._target == "encode":
            self.enc_result.appendPlainText(f"✘ {msg}")
        else:
            self.dec_result.appendPlainText(f"✘ {msg}")


class TokenizerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        tabs = QTabWidget(self)
        tabs.addTab(TokenizerTrainPage(), "训练词表")
        tabs.addTab(TokenizerToolsPage(), "编解码工具")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)
