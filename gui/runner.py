"""子进程运行器：异步读取 CLI 输出，整理为「完整行」与「进度覆盖行」。"""
from __future__ import annotations

from PySide6.QtCore import QObject, QProcess, Signal


class ProcessRunner(QObject):
    line = Signal(str)        # 以 \n 结束的完整行 → 追加到日志
    progress = Signal(str)    # 以 \r 覆盖的进度行 → 替换日志末行
    started = Signal()
    finished = Signal(int)    # 退出码
    error = Signal(str)       # 启动失败等

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._current = ""     # 正在累积的“行”
        self._pending_cr = False  # 上一块末尾的 \r 待与下一块合并判断

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def start(self, program: str, args: list[str], cwd: str) -> None:
        self.stop()
        self._current = ""
        self._pending_cr = False
        proc = QProcess(self)
        self._proc = proc
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(cwd)
        proc.readyReadStandardOutput.connect(self._on_data)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        proc.start(program, args)
        if not proc.waitForStarted(3000):
            if self._proc is proc:
                self._proc = None
            self.error.emit(f"无法启动进程: {program}（可能未构建）")
            return
        self.started.emit()

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            proc.kill()
            proc.waitForFinished(1000)

    def _on_data(self) -> None:
        if self._proc is None:
            return
        raw = bytes(self._proc.readAllStandardOutput())
        self._feed(raw.decode("utf-8", errors="replace"))

    def _feed(self, text: str) -> None:
        """解析行结束（CRLF / LF）与进度覆盖（独立 CR）。

        Windows 下 CLI 输出为 CRLF，独立 \\r 才是进度覆盖标记。
        """
        i = 0
        n = len(text)
        if self._pending_cr:
            self._pending_cr = False
            if text and text[0] == "\n":
                self.line.emit(self._current)
                self._current = ""
                i = 1
            else:
                if self._current:
                    self.progress.emit(self._current)
                    self._current = ""
        while i < n:
            ch = text[i]
            if ch == "\r":
                if i + 1 < n and text[i + 1] == "\n":
                    self.line.emit(self._current)
                    self._current = ""
                    i += 2
                elif i + 1 == n:
                    self._pending_cr = True  # 可能是跨块的 CRLF
                    i += 1
                else:
                    if self._current:  # 进度覆盖
                        self.progress.emit(self._current)
                        self._current = ""
                    i += 1
            elif ch == "\n":
                self.line.emit(self._current)
                self._current = ""
                i += 1
            else:
                self._current += ch
                i += 1

    def _on_finished(self, exit_code: int, _status) -> None:
        if self._current:
            self.line.emit(self._current)
            self._current = ""
        self._proc = None
        self.finished.emit(int(exit_code))

    def _on_error(self, err) -> None:
        self._proc = None
        self.error.emit(f"进程错误: {err}")
