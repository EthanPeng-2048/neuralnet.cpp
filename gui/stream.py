"""纯 Python 子进程运行器（Web GUI 用，无 Qt 依赖）。

事件队列项：
    ("line", text)     完整行（\\n 结束）
    ("progress", text) 进度覆盖行（独立 \\r）
    ("done", code)     进程结束，code 为退出码
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading


class StreamRunner:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, cmd: list[str], cwd: str) -> None:
        self._q = queue.Queue()
        self._current = ""
        self._pending_cr = False
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **kwargs,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()

    def drain(self, timeout: float = 0.05) -> list:
        """非阻塞取出当前可用事件。"""
        events = []
        try:
            while True:
                events.append(self._q.get(timeout=timeout))
        except queue.Empty:
            return events

    # ── 读取线程 ──────────────────────────────────────────────────
    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._q.put(("done", -1))
            return
        for raw in proc.stdout:
            self._feed(raw.decode("utf-8", errors="replace"))
        if self._current:
            self._q.put(("line", self._current))
        proc.wait()
        self._q.put(("done", proc.returncode))

    def _feed(self, text: str) -> None:
        """解析 CRLF / LF（行结束）与独立 \\r（进度覆盖）。"""
        i = 0
        n = len(text)
        if self._pending_cr:
            self._pending_cr = False
            if text and text[0] == "\n":
                self._q.put(("line", self._current))
                self._current = ""
                i = 1
            else:
                if self._current:
                    self._q.put(("progress", self._current))
                    self._current = ""
        while i < n:
            ch = text[i]
            if ch == "\r":
                if i + 1 < n and text[i + 1] == "\n":
                    self._q.put(("line", self._current))
                    self._current = ""
                    i += 2
                elif i + 1 == n:
                    self._pending_cr = True
                    i += 1
                else:
                    if self._current:
                        self._q.put(("progress", self._current))
                        self._current = ""
                    i += 1
            elif ch == "\n":
                self._q.put(("line", self._current))
                self._current = ""
                i += 1
            else:
                self._current += ch
                i += 1
