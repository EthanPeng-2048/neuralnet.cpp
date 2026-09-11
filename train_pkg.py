"""
train_pkg.py - 训练包（配置 + 数据）制作与按超参训练

用于在不同设备上便捷地进行训练/测试对比：

  1. 一个 JSON 配置（超参 + 数据路径）即可描述一次训练
  2. `pack` 命令把配置 + 训练集（文件或目录）打包成单个压缩文件
  3. 把压缩包拷到其他设备，`train` 命令自动解包并调用 cli_controllers 训练

包格式（外层压缩算法）：
  - 默认 gzip  (`.nnpkg` = tar.gz)：仅用 Python 标准库 tarfile，全平台通用，
    对文本语料压缩率高（通常 3~5x），任何工具都能打开
  - 可选 zstd (`.nnpkg.zst` 内容仍为 tar，外层 zstd)：需 `pip install zstandard`，
    压缩/解压更快、比率与 gzip 相当，适合超大语料；未安装时自动回退 gzip

包内结构：
  run.nnpkg
  ├── manifest.json      # 超参、任务类型、数据元信息、sha256 校验和
  └── data/              # 训练数据（文件或目录树）
      ├── train/...
      ├── test/...       # 可选（GPT）
      └── vocab/...      # 可选（GPT 词表）

用法：
  # 生成配置模板
  python train_pkg.py new --task gpt -o runs/gpt.json
  python train_pkg.py new --task mnist -o runs/mnist.json

  # 打包（配置 + 训练集 -> 单个压缩包）
  python train_pkg.py pack runs/gpt.json -o runs/gpt.nnpkg
  python train_pkg.py pack runs/gpt.json -o runs/gpt.nnpkg --compress zstd

  # 查看包内容
  python train_pkg.py info runs/gpt.nnpkg

  # 解包查看
  python train_pkg.py extract runs/gpt.nnpkg -o runs/gpt_extract

  # 训练：直接喂配置（本机）或压缩包（跨设备，自动解包并校验）
  python train_pkg.py train runs/gpt.json
  python train_pkg.py train runs/gpt.nnpkg --device cuda
  python train_pkg.py train runs/gpt.nnpkg --workdir runs/gpt_work --save runs/gpt.bin
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# cli_controllers 与本模块同目录
from cli_controllers import (
    MnistTrainController,
    GptTrainController,
    Metric,
)

# ── 可选 zstd ──────────────────────────────────────────────────────
try:
    import zstandard as _zstd
    HAVE_ZSTD = True
except ImportError:  # pragma: no cover - 依赖缺失时的回退路径
    _zstd = None
    HAVE_ZSTD = False

FORMAT_VERSION = 1
PKG_EXT = ".nnpkg"
MANIFEST_NAME = "manifest.json"

_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

SUPPORTED_TASKS = ("gpt", "mnist")
SUPPORTED_COMPRESS = ("gzip", "zstd")

# 每种任务：数据 role -> 传给控制器的参数名
TASK_DATA_ROLE_TO_ARG = {
    "gpt": {"train": "text_file", "test": "test_file", "vocab": "vocab"},
    "mnist": {"train": "dataset"},
}


# ═══════════════════════════════════════════════════════════════════
#  基础工具
# ═══════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    """计算文件 sha256（分块读取，适配超大文件）"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    """递归统计目录总字节数"""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _detect_compress(path: Path) -> str:
    """按魔数检测外层压缩算法"""
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:4] == _ZSTD_MAGIC:
        return "zstd"
    if head[:2] == _GZIP_MAGIC:
        return "gzip"
    return "gzip"  # 兜底


@contextmanager
def _tar_write(path: Path, compress: str) -> Iterator[tarfile.TarFile]:
    """按压缩算法打开 tar 写句柄"""
    compress = compress or "gzip"
    if compress not in SUPPORTED_COMPRESS:
        raise ValueError(f"不支持的压缩算法: {compress}，可选: {SUPPORTED_COMPRESS}")
    if compress == "zstd":
        if not HAVE_ZSTD:
            raise RuntimeError("zstd 需要 `pip install zstandard`，或改用 --compress gzip")
        with open(path, "wb") as f:
            cctx = _zstd.ZstdCompressor(level=3)
            with cctx.stream_writer(f) as w:
                with tarfile.open(fileobj=w, mode="w") as tar:
                    yield tar
    else:
        with tarfile.open(path, "w:gz") as tar:
            yield tar


@contextmanager
def _tar_read(path: Path) -> Iterator[tarfile.TarFile]:
    """自动识别压缩算法打开 tar 读句柄（上下文管理器，确保底层文件被关闭）"""
    compress = _detect_compress(path)
    if compress == "zstd":
        if not HAVE_ZSTD:
            raise RuntimeError("解压 zstd 包需要 `pip install zstandard`")
        raw = open(path, "rb")
        try:
            dctx = _zstd.ZstdDecompressor()
            with tarfile.open(fileobj=dctx.stream_reader(raw), mode="r|") as tar:
                yield tar
        finally:
            raw.close()
    else:
        with tarfile.open(path, "r:gz") as tar:
            yield tar


def _safe_extract(tar: tarfile.TarFile, out_dir: Path) -> None:
    """
    安全解包：逐成员校验路径，拒绝绝对路径 / `..` 逃逸。

    兼容 Python 3.10（无 extractall filter 参数）与流式 tar（r| 只能读一遍）。
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for member in tar:
        name = member.name.replace("\\", "/")
        parts = Path(name).parts
        if not parts or name.startswith("/") or ".." in parts:
            raise ValueError(f"包内含不安全路径: {name!r}")
        dest = (out_dir / name).resolve()
        if dest != out_dir and out_dir not in dest.parents:
            raise ValueError(f"包内路径逃逸出目录: {name!r}")
        try:
            tar.extract(member, out_dir, filter="data")
        except TypeError:  # Python < 3.12
            tar.extract(member, out_dir)


# ═══════════════════════════════════════════════════════════════════
#  配置读写 / 校验
# ═══════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置 {config_path} 顶层必须是 JSON 对象")
    return cfg


def validate_config(cfg: Dict[str, Any], need_data: bool = True) -> str:
    """校验配置，返回任务类型；数据缺失时抛错"""
    task = cfg.get("task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"config.task 必须为 {SUPPORTED_TASKS} 之一，当前: {task!r}")
    if cfg.get("format_version", FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError(f"不支持的 format_version: {cfg.get('format_version')}")
    if need_data:
        data = cfg.get("data")
        if not isinstance(data, dict) or not data.get("train"):
            raise ValueError("config.data 缺少必填的 train 字段")
    hyper = cfg.get("hyperparameters")
    if not isinstance(hyper, dict):
        raise ValueError("config.hyperparameters 必须为 JSON 对象")
    return task


def make_template(task: str, name: str = "run") -> Dict[str, Any]:
    """生成一份可编辑的配置模板"""
    if task == "gpt":
        return {
            "format_version": FORMAT_VERSION,
            "name": name,
            "task": "gpt",
            "device": "cpu",
            "data": {
                "train": "datasets/tinystories_20k.txt",
                "test": "",
                "vocab": "bpe_vocab.json",
            },
            "hyperparameters": {
                "epochs": 10,
                "lr": 0.001,
                "batch_size": 4,
                "accum_steps": 1,
                "seq_len": 256,
                "stride": 0,
                "optimizer": "adam",
                "weight_decay": 0.01,
                "d_model": 256,
                "num_heads": 4,
                "num_layers": 4,
                "d_ff": 1024,
                "positional_encoding": "learned",
                "activation": "gelu",
                "norm": "layernorm",
                "log_interval": 50,
                "save_interval": 100,
                "lr_schedule": "fixed",
            },
        }
    elif task == "mnist":
        return {
            "format_version": FORMAT_VERSION,
            "name": name,
            "task": "mnist",
            "device": "cpu",
            "data": {"train": "datasets/mnist_data"},
            "hyperparameters": {
                "arch": "transformer",
                "epochs": 10,
                "lr": 0.001,
                "batch_size": 32,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "d_model": 128,
                "num_heads": 4,
                "num_layers": 4,
                "d_ff": 512,
                "eval_samples": 100,
                "norm": "layernorm",
                "lr_schedule": "fixed",
            },
        }
    raise ValueError(f"未知任务: {task}")


# ═══════════════════════════════════════════════════════════════════
#  打包
# ═══════════════════════════════════════════════════════════════════

def _resolve_data_path(rel_or_abs: str, base_dir: Path) -> Path:
    """把配置里的数据路径解析为绝对路径（相对配置目录解析）"""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def pack(config_src: Any, out_path: str | Path, *,
         compress: str = "gzip",
         base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    把配置 + 训练数据打包成单个压缩文件。

    Args:
        config_src: 配置字典，或配置 json 文件路径
        out_path:   输出包路径（.nnpkg）
        compress:   外层压缩算法: gzip（默认）/ zstd
        base_dir:   相对数据路径的解析基准（dict 配置时必传；文件配置默认用其所在目录）

    Returns:
        manifest（写入包内的 manifest.json 内容）
    """
    if isinstance(config_src, (str, Path)):
        cfg_path = Path(config_src)
        cfg = load_config(cfg_path)
        base_dir = base_dir or cfg_path.parent
    elif isinstance(config_src, dict):
        cfg = dict(config_src)
        base_dir = base_dir or Path.cwd()
    else:
        raise TypeError("config_src 必须是配置路径或配置字典")

    validate_config(cfg)
    task = cfg["task"]
    data_cfg = cfg.get("data") or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 收集待打包数据 ──────────────────────────────────────────
    entries: List[Dict[str, Any]] = []
    for role in TASK_DATA_ROLE_TO_ARG[task]:
        raw = data_cfg.get(role)
        if not raw:
            continue
        src = _resolve_data_path(str(raw), base_dir)
        if not src.exists():
            raise FileNotFoundError(f"打包数据不存在: {src}（来自 data.{role}）")

        if src.is_dir():
            kind, arc, size = "dir", f"data/{role}", dir_size(src)
        else:
            kind, arc, size = "file", f"data/{role}/{src.name}", src.stat().st_size

        entries.append({
            "role": role, "kind": kind, "archive": arc,
            "original": str(raw), "size": size,
            "sha256": None if src.is_dir() else sha256_file(src),
        })

    if not entries:
        raise ValueError("没有可打包的数据，请先填写 config.data.train")

    manifest: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "name": cfg.get("name", "run"),
        "task": task,
        "device": cfg.get("device", "cpu"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "package": {"compress": compress, "data": entries},
        "hyperparameters": cfg.get("hyperparameters", {}),
    }

    # ── 写入压缩包 ──────────────────────────────────────────────
    with _tar_write(out_path, compress) as tar:
        # 先写 manifest，便于查看
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest_bytes))

        for e in entries:
            src = _resolve_data_path(e["original"], base_dir)
            tar.add(str(src), arcname=e["archive"], recursive=True)

    return manifest


# ═══════════════════════════════════════════════════════════════════
#  查看 / 解包
# ═══════════════════════════════════════════════════════════════════

def read_manifest(pkg_path: Path) -> Dict[str, Any]:
    """读取包内 manifest.json（不整体解包）"""
    with _tar_read(pkg_path) as tar:
        member = tar.getmember(MANIFEST_NAME)
        raw = tar.extractfile(member)
        return json.loads(raw.read().decode("utf-8"))


def list_package(pkg_path: Path) -> List[Dict[str, Any]]:
    """列出包内所有文件（名称 + 解压后大小）"""
    with _tar_read(pkg_path) as tar:
        return [{"name": m.name, "size": m.size} for m in tar.getmembers() if m.isfile()]


def extract(pkg_path: Path, out_dir: str | Path) -> Path:
    """解包到目录并返回包路径"""
    pkg_path = Path(pkg_path)
    out_dir = Path(out_dir)
    with _tar_read(pkg_path) as tar:
        _safe_extract(tar, out_dir)
    return pkg_path


def info_text(pkg_path: Path) -> str:
    """生成人类可读的包信息文本"""
    pkg_path = Path(pkg_path)
    manifest = read_manifest(pkg_path)
    total = pkg_path.stat().st_size
    lines = [
        f"训练包: {pkg_path}",
        f"  名称: {manifest.get('name')}",
        f"  任务: {manifest.get('task')}",
        f"  设备: {manifest.get('device')}",
        f"  创建: {manifest.get('created_at')}",
        f"  压缩: {manifest.get('package', {}).get('compress')}",
        f"  包大小: {total:,} bytes",
        "",
        "数据文件:",
    ]
    for e in manifest.get("package", {}).get("data", []):
        chk = f"  sha256={e['sha256'][:12]}…" if e.get("sha256") else ""
        lines.append(f"  [{e['role']}] {e['archive']} ({e['size']:,} bytes){chk}")
    lines.append("")
    lines.append("超参数:")
    for k, v in manifest.get("hyperparameters", {}).items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════════════════════

def _resolve_workdir(pkg_path: Path) -> Path:
    """默认解包工作目录：<包路径>.d"""
    return Path(str(pkg_path) + ".d")


def prepare(config_src: Any, *,
            workdir: Optional[str | Path] = None,
            base_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    把输入（配置 json 或 .nnpkg）解析为 (manifest, data_path_map)。

    data_path_map: role -> 本地绝对路径（包时指向解包目录，配置时指向原路径）
    """
    src = Path(config_src) if isinstance(config_src, (str, Path)) else None

    if src is not None and src.suffix == PKG_EXT:
        pkg = src
        wd = Path(workdir) if workdir else _resolve_workdir(pkg)
        manifest = read_manifest(pkg)
        # 已解包且 manifest 一致 -> 复用，避免每次重新解压大数据
        wd_manifest = wd / MANIFEST_NAME
        if not (wd_manifest.exists() and wd_manifest.read_text("utf-8") == json.dumps(manifest, ensure_ascii=False, indent=2)):
            print(f"[train_pkg] 解包 {pkg} -> {wd}")
            shutil.rmtree(wd, ignore_errors=True)
            extract(pkg, wd)
        data_map = {e["role"]: str(wd / e["archive"]) for e in manifest["package"]["data"]}
        return manifest, data_map

    # 普通配置
    cfg_path = Path(config_src)
    cfg = load_config(cfg_path)
    task = validate_config(cfg)
    base = base_dir or cfg_path.parent
    data_map: Dict[str, str] = {}
    for role in TASK_DATA_ROLE_TO_ARG[task]:
        raw = (cfg.get("data") or {}).get(role)
        if raw:
            data_map[role] = str(_resolve_data_path(str(raw), base))
    return cfg, data_map


def train(config_src: Any, *,
          device: Optional[str] = None,
          workdir: Optional[str | Path] = None,
          save: Optional[str] = None,
          output_callback: Optional[Callable[[str], None]] = None,
          error_callback: Optional[Callable[[str], None]] = None,
          metric_callback: Optional[Callable[[Metric], None]] = None,
          completion_callback: Optional[Callable[[int], None]] = None) -> int:
    """
    按配置/包中的超参执行训练（通过 cli_controllers 调用 CLI）。

    Returns: 进程退出码
    """
    manifest, data_map = prepare(config_src, workdir=workdir)
    task = manifest["task"]
    hyper = dict(manifest.get("hyperparameters") or {})

    # 1) 数据路径 -> 控制器参数
    role_map = TASK_DATA_ROLE_TO_ARG[task]
    for role, arg in role_map.items():
        local = data_map.get(role)
        if local:
            hyper[arg] = local

    # 2) 设备覆盖（CLI 优先，其次配置 device）
    dev = device or manifest.get("device") or "cpu"
    hyper.pop("gpu", None)
    hyper.pop("cuda", None)
    if dev == "gpu":
        hyper["gpu"] = True
    elif dev == "cuda":
        hyper["cuda"] = True
    elif dev != "cpu":
        raise ValueError(f"未知设备: {dev}，可选 cpu/gpu/cuda")

    # 3) 保存路径覆盖
    if save:
        hyper["save"] = save

    # 4) 组装控制器
    controller: Any
    if task == "gpt":
        controller = GptTrainController()
    elif task == "mnist":
        controller = MnistTrainController()
    else:
        raise ValueError(f"未知任务: {task}")

    controller.set_output_callback(output_callback or (lambda line: print(line)))
    if error_callback:
        controller.set_error_callback(error_callback)
    if metric_callback:
        controller.set_metric_callback(metric_callback)
    if completion_callback:
        controller.set_completion_callback(completion_callback)

    print(f"[train_pkg] 任务={task} 设备={dev} 参数={json.dumps(hyper, ensure_ascii=False)}")
    return controller.run(**hyper)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=["cpu", "gpu", "cuda"],
                   help="覆盖训练设备（默认用配置里的 device）")
    p.add_argument("--workdir", help="解包/工作目录（默认 <包路径>.d）")
    p.add_argument("--save", help="覆盖模型保存路径")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="train_pkg.py",
        description="训练包（配置+数据）制作与按超参训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1].split("══")[0],
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="生成配置模板")
    p_new.add_argument("--task", choices=SUPPORTED_TASKS, required=True)
    p_new.add_argument("-o", "--output", default="train_config.json")
    p_new.add_argument("--name", default="run")

    p_pack = sub.add_parser("pack", help="把配置+数据打包成压缩包")
    p_pack.add_argument("config", help="配置文件路径")
    p_pack.add_argument("-o", "--output", help=f"输出包路径（默认 <config>.{PKG_EXT}）")
    p_pack.add_argument("--compress", choices=SUPPORTED_COMPRESS, default="gzip",
                        help="外层压缩算法（默认 gzip，可选 zstd）")

    p_info = sub.add_parser("info", help="查看压缩包内容")
    p_info.add_argument("pkg", help=f"{PKG_EXT} 包路径")
    p_info.add_argument("--list", action="store_true", help="列出所有文件")

    p_extract = sub.add_parser("extract", help="解包到目录")
    p_extract.add_argument("pkg", help=f"{PKG_EXT} 包路径")
    p_extract.add_argument("-o", "--output", help="输出目录")

    p_train = sub.add_parser("train", help="按配置/包中的超参训练")
    p_train.add_argument("config", help="配置文件 或 打包文件(.nnpkg)")
    _add_common(p_train)

    args = ap.parse_args(argv)

    try:
        if args.cmd == "new":
            cfg = make_template(args.task, args.name)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"已生成模板: {out}")
            print("编辑其中的 data / hyperparameters 后执行:")
            print(f"  python train_pkg.py pack {out}")

        elif args.cmd == "pack":
            cfg_path = Path(args.config)
            out = Path(args.output) if args.output else cfg_path.with_suffix(PKG_EXT)
            manifest = pack(cfg_path, out, compress=args.compress)
            print(f"打包完成: {out}")
            print(f"  任务: {manifest['task']}  数据: {len(manifest['package']['data'])} 个  "
                  f"压缩: {manifest['package']['compress']}")
            print(f"  包大小: {out.stat().st_size:,} bytes")
            print("在目标设备上执行:")
            print(f"  python train_pkg.py train {out}")

        elif args.cmd == "info":
            pkg = Path(args.pkg)
            if args.list:
                for m in list_package(pkg):
                    print(f"{m['name']:40s} {m['size']:,} bytes")
            else:
                print(info_text(pkg))

        elif args.cmd == "extract":
            pkg = Path(args.pkg)
            out = Path(args.output) if args.output else Path(str(pkg) + "_extract")
            extract(pkg, out)
            print(f"已解包: {pkg} -> {out}")

        elif args.cmd == "train":
            code = train(args.config, device=args.device,
                         workdir=args.workdir, save=args.save)
            print(f"[train_pkg] 训练结束，退出码: {code}")
            return code

        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
