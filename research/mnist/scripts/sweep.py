#!/usr/bin/env python3
"""MNIST 模型尺寸 / 训练超参 → 测试准确率 扫描驱动。

设计要点（与仓库 tests/ 下既有的 norm A/B 实验包同源，但更严格）：
  * 全部实验统一加 `--shuffle-steps false`：mnist_train 的 batch 打乱用
    std::random_device 播种、没有 --seed 参数；关闭打乱后「模型初始化（InitSpec
    按创建序号定种）+ 样本顺序」都确定 → 同一配置逐位可复现（含 .bin 哈希），
    单次运行即可做无噪声 A/B。代价：绝对准确率略低于打乱训练，只用于**相对比较**。
  * 关键结论再用默认打乱（shuffle=true）多轮复验（robust 组），防止结论被
    「固定顺序」这一特殊训练制度带偏。
  * 每个 run 一个日志 + 一个 .bin；日志可断点续跑（检测到「训练完成」且模型存在则跳过）。
  * 全局汇总写到 research/mnist/results/all_runs.csv（每 run 一行）与
    research/mnist/results/all_epochs.csv（每 run·epoch 一行，学习曲线来源）。

用法：
  python research/mnist/scripts/sweep.py --phases scale_conv,scale_fc --jobs 4
  python research/mnist/scripts/sweep.py --list
  python research/mnist/scripts/sweep.py --rebuild-csv        # 只从已有日志重建 CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, "research", "mnist", "results")
# 注：早期 46 个配置是用过期 Debug 二进制跑的历史数据；新实验请用 build_rel。
TRAIN = os.path.join(ROOT, "build_rel", "mnist_train.exe")
if not os.path.exists(TRAIN):
    TRAIN = os.path.join(ROOT, "build", "mnist_train.exe")

# ── 默认（全尺寸基线 = tests/README.md 阶段 1 的已知较优参数；该包在仓库 tests/ 下） ──
BASE = dict(
    arch="cnn",
    epochs=18,
    batch_size=128,
    lr=0.005,
    optimizer="adamw",
    weight_decay=0.001,
    cnn_channels="16,32",
    cnn_fc="256,128,10",
    lr_schedule="cosine",
    warmup_epochs=3,
    min_lr=1e-6,
    shuffle="false",
    gpu=True,
)

# 小模型代理（快，用于交互项扫描）
SMALL = dict(BASE, cnn_channels="8,16", cnn_fc="64,10")
# 大模型（用于 lr 交互）
LARGE = dict(BASE, cnn_channels="32,64", cnn_fc="512,256,10")

FLAG = {
    "arch": "--arch",
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "optimizer": "--optimizer",
    "weight_decay": "--weight-decay",
    "max_samples": "--max-samples",
    "shuffle": "--shuffle-steps",
    "cnn_channels": "--cnn-channels",
    "cnn_kernels": "--cnn-kernels",
    "cnn_pool": "--cnn-pool",
    "cnn_fc": "--cnn-fc",
    "layer_dims": "--layer-dims",
    "norm": "--norm",
    "norm_place": "--norm-place",
    "d_model": "--d-model",
    "num_heads": "--num-heads",
    "num_layers": "--num-layers",
    "d_ff": "--d-ff",
    "patch_size": "--patch-size",
    "eval_samples": "--eval-samples",
    "lr_schedule": "--lr-schedule",
    "warmup_epochs": "--warmup-epochs",
    "min_lr": "--min-lr",
    "precision_param": "--precision-param",
    "precision_compute": "--precision-compute",
    "precision_stable": "--precision-stable",
    "precision_optimizer": "--precision-optimizer",
}
BOOL_FLAG = {"gpu": "--gpu", "fp16": "--f16"}


def fmt(v) -> str:
    if isinstance(v, float):
        return repr(v)
    return str(v)


def build_args(cfg: dict) -> list[str]:
    args: list[str] = []
    for k, v in cfg.items():
        if v is None:
            continue
        if k in BOOL_FLAG:
            if v:
                args.append(BOOL_FLAG[k])
            continue
        if k not in FLAG:
            raise KeyError(f"unknown config key: {k}")
        args += [FLAG[k], fmt(v)]
    return args


# ── 结构规模估计（conv + linear 的参数数；norm 参数另计，不在此列） ────────
def cnn_core_params(channels, kernels, pool, fc, in_size=28, in_ch=1):
    convs = [int(x) for x in channels.split(",")]
    ks = [int(x) for x in kernels.split(",")] if kernels else [5] * len(convs)
    fcs = [int(x) for x in fc.split(",")]
    c, h, w = in_ch, in_size, in_size
    p = 0
    for oc, k in zip(convs, ks):
        p += oc * (c * k * k + 1)
        h, w = h - k + 1, w - k + 1
        c = oc
        if pool:
            h, w = (h - pool) // pool + 1, (w - pool) // pool + 1
    flat = c * h * w
    prev = flat
    for d in fcs:
        p += d * (prev + 1)
        prev = d
    return p, convs, fcs, flat


def mlp_core_params(dims: str):
    ds = [int(x) for x in dims.split(",")]
    p = 0
    for a, b in zip(ds[:-1], ds[1:]):
        p += b * (a + 1)
    return p, ds


@dataclass
class Run:
    name: str
    phase: str
    cfg: dict
    factors: dict = field(default_factory=dict)
    note: str = ""

    def args(self):
        return build_args(self.cfg)

    @property
    def log(self):
        return os.path.join(RES, self.phase, "logs", self.name + ".log")

    @property
    def model(self):
        return os.path.join(RES, self.phase, "models", self.name + ".bin")


def R(name, phase, factors=None, note="", **over):
    cfg = dict(BASE)
    cfg.update(over)
    return Run(name, phase, cfg, factors or {}, note)


# ══════════════════════════════════════════════════════════════════════════
# 实验分组
# ══════════════════════════════════════════════════════════════════════════
def group_scale_conv():
    """CNN 卷积宽度（FC 固定 128,10）。"""
    out = []
    for ch in ["4,8", "8,16", "16,32", "32,64", "64,128"]:
        out.append(R(f"cw_{ch.replace(',', '_')}", "scale_conv", {"axis": "conv_channels", "value": ch}, cnn_channels=ch, cnn_fc="128,10"))
    return out


def group_scale_fc():
    """CNN 全连接头宽度/深度（卷积固定 16,32）。"""
    out = []
    for fc in ["32,10", "64,10", "128,10", "256,10", "256,128,10", "512,256,10", "1024,512,10"]:
        out.append(R(f"fc_{fc.replace(',', '_')}", "scale_fc", {"axis": "fc_dims", "value": fc}, cnn_fc=fc))
    return out


def group_scale_depth():
    """CNN 卷积深度。

    注意 3×(k5+pool2) 在 28×28 上不可行（第 3 层输入已只剩 4×4 < k5），
    故深度 ≥3 的配置统一改用 k3：28→26/pool 13 → 11/pool 5 → 3/pool 1。
    k5 的 1/2 层保留作对照。
    """
    out = []
    out.append(R("dp_1", "scale_depth", {"axis": "conv_depth", "value": "1 (k5, ch32)"}, cnn_channels="32", cnn_kernels="5", cnn_fc="128,10"))
    out.append(R("dp_2", "scale_depth", {"axis": "conv_depth", "value": "2 (k5, ch16,32)"}, cnn_channels="16,32", cnn_kernels="5,5", cnn_fc="128,10"))
    for i, ch in enumerate(["16", "16,32", "16,32,64"], start=1):
        ks = ",".join(["3"] * i)
        out.append(R(f"dpk_{i}", "scale_depth", {"axis": "conv_depth(k3)", "value": f"{i} (k3, ch{ch})"}, cnn_channels=ch, cnn_kernels=ks, cnn_fc="128,10"))
    return out


def group_scale_kernel_pool():
    """卷积核大小 / 池化窗口（容量近似匹配，考察感受野与下采样策略）。"""
    out = []
    for name, ks, pool in [("k3", "3,3", 2), ("k5", "5,5", 2), ("k7", "7,7", 2), ("k5_pool3", "5,5", 3), ("k5_nopool", "5,5", 0)]:
        out.append(R(f"kp_{name}", "scale_kernel_pool", {"axis": "kernel/pool", "value": name}, cnn_kernels=ks, cnn_pool=pool, cnn_fc="128,10"))
    return out


def group_hp_lr():
    out = []
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2]:
        out.append(R(f"lr_{lr:g}", "hp_lr", {"axis": "lr", "value": lr}, lr=lr))
    return out


def group_hp_lr_size():
    out = []
    for tag, base in [("small", SMALL), ("large", LARGE)]:
        for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
            cfg = dict(base)
            cfg["lr"] = lr
            out.append(Run(f"lrs_{tag}_{lr:g}", "hp_lr_size", cfg, {"axis": "lr x size", "value": f"{tag}@{lr:g}"}))
    return out


def group_hp_opt():
    out = []
    grid = {
        "sgd": [1e-3, 1e-2, 1e-1],
        "sgd_momentum": [1e-3, 1e-2, 1e-1],
        "adam": [1e-3, 1e-2],
        "adamw": [1e-3, 1e-2],
        "muon": [1e-2, 3e-2, 1e-1],
    }
    for opt, lrs in grid.items():
        for lr in lrs:
            out.append(R(f"opt_{opt}_{lr:g}", "hp_opt", {"axis": "optimizer", "value": opt, "lr": lr}, optimizer=opt, lr=lr))
    return out


def group_hp_wd():
    out = []
    for wd in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        out.append(R(f"wd_{wd:g}", "hp_wd", {"axis": "weight_decay", "value": wd}, weight_decay=wd))
    return out


def group_hp_bs():
    out = []
    for bs in [16, 32, 64, 128, 256, 512]:
        out.append(R(f"bs_{bs}", "hp_bs", {"axis": "batch_size", "value": bs}, batch_size=bs))
    for bs in [16, 32, 256, 512]:  # 线性缩放律：lr ∝ bs
        out.append(R(f"bslin_{bs}", "hp_bs", {"axis": "batch_size(lr scaled)", "value": bs}, batch_size=bs, lr=0.005 * bs / 128))
    return out


def group_hp_sched():
    out = []
    out.append(R("sch_fixed_w0", "hp_sched", {"axis": "schedule", "value": "fixed"}, lr_schedule="fixed", warmup_epochs=0))
    for w in [0, 1, 6, 9]:
        out.append(R(f"sch_cos_w{w}", "hp_sched", {"axis": "schedule", "value": f"cosine+w{w}"}, lr_schedule="cosine", warmup_epochs=w))
    out.append(R("sch_cos_w3_min0", "hp_sched", {"axis": "schedule", "value": "cosine+w3+min0"}, lr_schedule="cosine", warmup_epochs=3, min_lr=0.0))
    return out


def group_data_scale():
    out = []
    for n in [1000, 2000, 5000, 10000, 20000, 40000, 60000]:
        out.append(R(f"n_{n}", "data_scale", {"axis": "train_samples", "value": n}, max_samples=n))
    return out


def group_epochs_curve():
    out = [
        R("base_e30", "epochs_curve", {"axis": "epochs", "value": 30}, epochs=30),
        Run("small_e60", "epochs_curve", dict(SMALL, epochs=60), {"axis": "epochs(small)", "value": 60}),
    ]
    return out


def group_mlp_scale():
    out = []
    for dims in ["784,10", "784,64,10", "784,128,10", "784,512,10", "784,256,256,10",
                 "784,512,256,128,10", "784,1024,1024,10", "784,2048,2048,10",
                 "784,512,256,128,64,10"]:
        out.append(R(f"mlp_{dims.replace(',', '_')}", "mlp_scale", {"axis": "layer_dims", "value": dims}, arch="mlp", layer_dims=dims))
    return out


def group_vit_scale():
    """ViT：必须把 --eval-samples 提到全量 10000，否则默认只评 200 个样本、噪声极大。"""
    out = []
    ev = dict(arch="transformer", eval_samples=10000, lr=0.003)
    for l in [1, 2, 3, 4, 6]:
        out.append(R(f"vit_l{l}", "vit_scale", {"axis": "num_layers", "value": l}, num_layers=l, **ev))
    for d, h in [(32, 4), (64, 4), (128, 8), (256, 8)]:
        out.append(R(f"vit_d{d}", "vit_scale", {"axis": "d_model", "value": d}, d_model=d, num_heads=h, **ev))
    for ff in [64, 256, 512]:
        out.append(R(f"vit_ff{ff}", "vit_scale", {"axis": "d_ff", "value": ff}, d_ff=ff, **ev))
    for p in [14, 4, 28]:
        out.append(R(f"vit_p{p}", "vit_scale", {"axis": "patch_size", "value": p}, patch_size=p, **ev))
    return out


def group_interaction_size_data():
    """容量 × 数据量：大模型在小数据上是否更差（过拟合 / 容量-数据匹配）。"""
    out = []
    for tag, cfg in [("tiny", dict(BASE, cnn_channels="4,8", cnn_fc="32,10")), ("base", dict(BASE))]:
        for n in [1000, 2000, 5000, 20000]:
            c = dict(cfg, max_samples=n)
            out.append(Run(f"isd_{tag}_n{n}", "interaction_size_data", c, {"axis": "size x data", "value": f"{tag}@{n}"}))
    return out


def group_interaction_norm_size():
    """归一化 × 尺寸：仓库 tests/ 的 norm A/B 只在全尺寸基线上做，这里补尺寸交互。"""
    out = []
    combos = [("none", dict(norm="batchnorm", norm_place="none")),
              ("bn_conv", dict(norm="batchnorm", norm_place="conv")),
              ("ln_conv", dict(norm="layernorm", norm_place="conv")),
              ("rms_conv", dict(norm="rmsnorm", norm_place="conv"))]
    for tag, cfg in [("small", dict(BASE, cnn_channels="8,16", cnn_fc="64,10")),
                     ("large", dict(BASE, cnn_channels="32,64", cnn_fc="512,256,10"))]:
        for ntag, over in combos:
            c = dict(cfg)
            c.update(over)
            out.append(Run(f"ins_{tag}_{ntag}", "interaction_norm_size", c, {"axis": "size x norm", "value": f"{tag}/{ntag}"}))
    return out


def group_precision():
    return [
        R("prec_f32", "precision", {"axis": "precision", "value": "f32"}),
        R("prec_f16", "precision", {"axis": "precision", "value": "f16"}, fp16=True),
    ]


ROBUST_TARGETS = [
    ("base", BASE),
    ("cw_32_64", dict(BASE, cnn_channels="32,64", cnn_fc="128,10")),
    ("cw_08_16", dict(BASE, cnn_channels="8,16", cnn_fc="128,10")),
    ("fc_64", dict(BASE, cnn_fc="64,10")),
    ("lr_1e-3", dict(BASE, lr=1e-3)),
    ("lr_1e-2", dict(BASE, lr=1e-2)),
    # 打乱制度最可能影响的几项：批量大小与优化器结论必须复验
    ("bs_16", dict(BASE, batch_size=16)),
    ("bs_512", dict(BASE, batch_size=512)),
    ("sgd_1e-2", dict(BASE, optimizer="sgd", lr=1e-2)),
]


def group_robust(rounds=3):
    out = []
    for tag, cfg in ROBUST_TARGETS:
        for r in range(1, rounds + 1):
            c = dict(cfg, shuffle="true")
            out.append(Run(f"rob_{tag}_r{r}", "robust", c, {"axis": "shuffle=true round", "value": f"{tag}#{r}"}))
    return out


GROUPS = {
    "scale_conv": group_scale_conv,
    "scale_fc": group_scale_fc,
    "scale_depth": group_scale_depth,
    "scale_kernel_pool": group_scale_kernel_pool,
    "hp_lr": group_hp_lr,
    "hp_lr_size": group_hp_lr_size,
    "hp_opt": group_hp_opt,
    "hp_wd": group_hp_wd,
    "hp_bs": group_hp_bs,
    "hp_sched": group_hp_sched,
    "data_scale": group_data_scale,
    "epochs_curve": group_epochs_curve,
    "mlp_scale": group_mlp_scale,
    "vit_scale": group_vit_scale,
    "interaction_size_data": group_interaction_size_data,
    "interaction_norm_size": group_interaction_norm_size,
    "precision": group_precision,
    "robust": group_robust,
}

# ══════════════════════════════════════════════════════════════════════════
# 日志解析
# ══════════════════════════════════════════════════════════════════════════
EP_RE = re.compile(
    r"Epoch (\d+)/(\d+)\s+lr=(\S+)\s+loss=(\S+)\s+train_acc=([\d.]+)%\s+test_acc=([\d.]+)%\s+time=([\d.]+)s"
)
DONE_RE = re.compile(r"训练完成")


def parse_log(path):
    txt = open(path, errors="replace").read().replace("\r", "\n")
    eps = [dict(epoch=int(m[0]), total=int(m[1]), lr=float(m[2]), loss=float(m[3]),
                train=float(m[4]), test=float(m[5]), time=float(m[6]))
           for m in EP_RE.findall(txt)]
    return eps, bool(DONE_RE.search(txt))


def est_params(run: Run):
    c = run.cfg
    try:
        if c.get("arch", "cnn") == "cnn":
            p, _, _, _ = cnn_core_params(c.get("cnn_channels", "16,32"), c.get("cnn_kernels"), c.get("cnn_pool", 2), c.get("cnn_fc", "256,128,10"))
            return p
        if c.get("arch") == "mlp":
            p, _ = mlp_core_params(c.get("layer_dims", "784,512,256,128,64,10"))
            return p
    except Exception:
        return None
    return None


def run_one(run: Run, resume=True):
    os.makedirs(os.path.dirname(run.log), exist_ok=True)
    os.makedirs(os.path.dirname(run.model), exist_ok=True)
    if resume and os.path.exists(run.log) and os.path.exists(run.model):
        eps, done = parse_log(run.log)
        if done and eps:
            return "skip"
    cmd = [TRAIN] + run.args() + ["--save", run.model]
    t0 = time.time()
    with open(run.log, "w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return f"ok({time.time() - t0:.0f}s,rc={proc.returncode})"


def collect(selected: list[Run]):
    """从日志重建全局 CSV。"""
    rows, erows = [], []
    for run in selected:
        if not os.path.exists(run.log):
            continue
        eps, done = parse_log(run.log)
        if not eps:
            continue
        tests = [e["test"] for e in eps]
        best_i = max(range(len(tests)), key=lambda i: tests[i])
        rows.append(dict(
            phase=run.phase, run=run.name,
            factors=json.dumps(run.factors, ensure_ascii=False),
            completed=done, epochs=len(eps), target_epochs=run.cfg.get("epochs"),
            final_test=tests[-1], best_test=tests[best_i], best_epoch=eps[best_i]["epoch"],
            final_train=eps[-1]["train"], final_loss=eps[-1]["loss"],
            mean_epoch_sec=round(sum(e["time"] for e in eps) / len(eps), 2),
            params=est_params(run),
            model_kb=round(os.path.getsize(run.model) / 1024, 1) if os.path.exists(run.model) else "",
            args=" ".join(run.args()),
            log=os.path.relpath(run.log, ROOT), model=os.path.relpath(run.model, ROOT) if os.path.exists(run.model) else "",
        ))
        for e in eps:
            erows.append(dict(phase=run.phase, run=run.name, **e))
    return rows, erows


def write_csvs(rows, erows):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "all_runs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "all_epochs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(erows[0].keys()))
        w.writeheader()
        w.writerows(erows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(GROUPS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3, help="robust 组每配置轮数")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-csv", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    all_runs = []
    for p in GROUPS:
        all_runs += GROUPS[p](**({"rounds": args.rounds} if p == "robust" else {}))

    if args.list:
        for p in GROUPS:
            rs = GROUPS[p](**({"rounds": args.rounds} if p == "robust" else {}))
            print(f"\n== {p}  ({len(rs)} runs)")
            for r in rs:
                print(f"  {r.name:<22} {' '.join(r.args())}")
        print(f"\n总计 {len(all_runs)} runs")
        return 0

    if args.rebuild_csv:
        rows, erows = collect(all_runs)
        write_csvs(rows, erows)
        print(f"rebuilt from logs: {len(rows)} runs, {len(erows)} epoch rows")
        return 0

    todo = [r for r in all_runs if r.phase in phases]
    print(f"待跑 {len(todo)} runs（phase={phases}, jobs={args.jobs}, resume={not args.no_resume}）")
    if args.dry_run:
        for r in todo:
            print(f"  {r.name:<22} {' '.join(r.args())}")
        return 0

    os.makedirs(RES, exist_ok=True)
    done_n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, r, not args.no_resume): r for r in todo}
        for fut in futs:
            r = futs[fut]
            try:
                st = fut.result()
            except Exception as e:  # 单个 run 失败不拖垮整批
                st = f"ERROR {e}"
            done_n += 1
            el = time.time() - t0
            print(f"[{done_n}/{len(todo)}] {r.phase}/{r.name}: {st}  (elapsed {el/60:.1f} min)", flush=True)

    rows, erows = collect(all_runs)
    write_csvs(rows, erows)
    print(f"\nwrote {RES}\\all_runs.csv ({len(rows)} runs) + all_epochs.csv ({len(erows)} epoch rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
