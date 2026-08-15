"""Web GUI 入口：python web_gui.py（自动打开浏览器）。"""
from __future__ import annotations

import sys

import gradio as gr

from gui.webapp import build_app


def main() -> None:
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
