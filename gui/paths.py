"""项目路径与文件扫描工具。"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
DATASETS_DIR = PROJECT_ROOT / "datasets"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

IS_WINDOWS = os.name == "nt"


def exe_path(base: str) -> Path:
    """返回可执行文件完整路径。"""
    name = f"{base}.exe" if IS_WINDOWS else base
    return BUILD_DIR / name


def exe_exists(base: str) -> bool:
    return exe_path(base).exists()


def available_executables() -> list[str]:
    """build/ 目录下已构建的可执行文件名（不含扩展名）。"""
    if not BUILD_DIR.exists():
        return []
    suffix = ".exe" if IS_WINDOWS else ""
    return sorted(
        p.name[: -len(suffix)]
        for p in BUILD_DIR.iterdir()
        if p.is_file() and suffix and p.name.endswith(suffix)
    )


def scan_models() -> dict[str, str]:
    """扫描项目根目录 + pretrained/ 下所有 .bin 模型，返回 {绝对路径: 显示名}。"""
    found: dict[str, str] = {}
    for base, tag in ((PROJECT_ROOT, "项目根"), (PRETRAINED_DIR, "pretrained")):
        if not base.exists():
            continue
        for p in sorted(base.glob("*.bin")):
            found[str(p)] = f"{p.name}（{tag}）"
    return found


def scan_datasets() -> list[Path]:
    """datasets/ 下的训练文本。"""
    if not DATASETS_DIR.exists():
        return []
    return sorted(DATASETS_DIR.glob("*.txt"))


def scan_mnist_images() -> list[Path]:
    """datasets/mnist_data/ 下的图片文件。"""
    d = DATASETS_DIR / "mnist_data"
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in (".csv", ".png", ".jpg", ".jpeg")
    )
