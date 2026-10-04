#!/usr/bin/env python3
"""聚焦实验：在「体积可控」的前提下把 MNIST CNN 测试准确率推到 99.3%+。

与 research/mnist/scripts/sweep.py 的区别：
  * 只用**默认打乱**训练制度（shuffle=true）——这是真实使用制度，也是历次高分的来源；
  * 两阶段：screen（每个候选 1 轮，快筛）→ confirm（筛出的前几名各 3 轮，取均值±std）；
  * 目标口径：**多轮均值** ≥ 99.3%，同时模型体积尽量小。

用法：
  python research/mnist/scripts/focus_cnn.py --stage screen   --jobs 4
  python research/mnist/scripts/focus_cnn.py --stage confirm  --jobs 4
  python research/mnist/scripts/focus_cnn.py --stage report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "research", "mnist", "results", "focus_cnn")
# 必须用 build_rel（Release + 含 NormPlace/BatchNorm 提交的源码）：
# 仓库里原有的 build/ 是 2026-10-02 的 Debug 产物，早于 --norm-place 特性。
TRAIN = os.path.join(ROOT, "build_rel", "mnist_train.exe")
if not os.path.exists(TRAIN):
    TRAIN = os.path.join(ROOT, "build", "mnist_train.exe")

# 统一训练协议（真实制度：打乱）
COMMON = dict(arch="cnn", epochs=18, batch_size=128, optimizer="adamw", weight_decay=0.001,
              lr_schedule="cosine", warmup_epochs=3, min_lr=1e-6, shuffle="true", gpu=True)

FLAG = {
    "arch": "--arch", "epochs": "--epochs", "batch_size": "--batch-size", "lr": "--lr",
    "optimizer": "--optimizer", "weight_decay": "--weight-decay", "max_samples": "--max-samples",
    "shuffle": "--shuffle-steps", "cnn_channels": "--cnn-channels", "cnn_kernels": "--cnn-kernels",
    "cnn_pool": "--cnn-pool", "cnn_fc": "--cnn-fc", "norm": "--norm", "norm_place": "--norm-place",
    "lr_schedule": "--lr-schedule", "warmup_epochs": "--warmup-epochs", "min_lr": "--min-lr",
}


def args_of(cfg):
    a = ["--gpu"]
    for k, v in cfg.items():
        if k == "gpu":
            continue
        a += [FLAG[k], repr(v) if isinstance(v, float) else str(v)]
    return a


# ── 候选配置（体积 ~700KB 起；新二进制 CNN 默认 BatchNorm@conv） ─────────
def candidates():
    C = {}
    def add(tag, **over):
        c = dict(COMMON)
        c.update(over)
        C[tag] = c

    # 基线族：16,32 / 256,128,10，BatchNorm@conv（= 架构默认）
    add("base_lr5e3", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10")
    add("base_lr3e3", lr=3e-3, cnn_channels="16,32", cnn_fc="256,128,10")
    add("base_lr1e2", lr=1e-2, cnn_channels="16,32", cnn_fc="256,128,10")
    # 归一化位置/类型（仓库 tests/ 的既有成果：小模型代理上 BatchNorm@head +0.22）
    add("bnhead_lr5e3", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10", norm_place="head")
    add("bnboth_lr5e3", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10", norm_place="both")
    add("none_lr5e3", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10", norm_place="none")
    add("lnconv_lr5e3", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10", norm="layernorm", norm_place="conv")
    # 容量
    add("med_lr3e3", lr=3e-3, cnn_channels="32,64", cnn_fc="512,256,10")
    add("med_lr1e3", lr=1e-3, cnn_channels="32,64", cnn_fc="512,256,10")
    add("fcbig_lr3e3", lr=3e-3, cnn_channels="16,32", cnn_fc="1024,512,10")
    add("deep3k3_lr3e3", lr=3e-3, cnn_channels="16,32,64", cnn_kernels="3,3,3", cnn_fc="512,256,10")
    # 训练预算 / 正则
    add("base_e30", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10", epochs=30)
    add("med_e30", lr=3e-3, cnn_channels="32,64", cnn_fc="512,256,10", epochs=30)
    add("base_wd1e2", lr=5e-3, cnn_channels="16,32", cnn_fc="256,128,10", weight_decay=1e-2)
    return C


EP_RE = re.compile(r"Epoch (\d+)/(\d+)\s+lr=(\S+)\s+loss=(\S+)\s+train_acc=([\d.]+)%\s+test_acc=([\d.]+)%\s+time=([\d.]+)s")
DONE_RE = re.compile(r"训练完成")


def parse(path):
    txt = open(path, errors="replace").read().replace("\r", "\n")
    eps = [dict(epoch=int(m[0]), total=int(m[1]), loss=float(m[3]), train=float(m[4]),
                test=float(m[5]), time=float(m[6])) for m in EP_RE.findall(txt)]
    return eps, bool(DONE_RE.search(txt))


def paths(stage, name):
    return (os.path.join(OUT, stage, "logs", name + ".log"),
            os.path.join(OUT, stage, "models", name + ".bin"))


def run_one(stage, name, cfg, resume=True):
    log, model = paths(stage, name)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    os.makedirs(os.path.dirname(model), exist_ok=True)
    if resume and os.path.exists(log) and os.path.exists(model):
        eps, done = parse(log)
        if done and eps:
            return "skip"
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run([TRAIN] + args_of(cfg) + ["--save", model], cwd=ROOT, stdout=fh,
                           stderr=subprocess.STDOUT)
    return f"ok({time.time()-t0:.0f}s,rc={p.returncode})"


def kb_from_csv():
    """逐 run 的 .bin 已被清理（模型不进仓库），体积列从 summary_*.csv 兜底。"""
    out = {}
    for fn in ("summary_screen.csv", "summary_confirm.csv"):
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("kb"):
                    out[row["name"]] = row["kb"]
    return out


def collect(stage):
    rows = []
    d = os.path.join(OUT, stage, "logs")
    if not os.path.isdir(d):
        return rows
    kbmap = kb_from_csv()
    for f in sorted(os.listdir(d)):
        if not f.endswith(".log"):
            continue
        name = f[:-4]
        log, model = paths(stage, name)
        eps, done = parse(log)
        if not eps:
            continue
        tests = [e["test"] for e in eps]
        kb = round(os.path.getsize(model) / 1024, 1) if os.path.exists(model) else kbmap.get(name, "")
        try:
            kb = float(kb)
        except (TypeError, ValueError):
            kb = ""
        rows.append(dict(stage=stage, name=name, done=done, epoch=len(eps),
                         final=tests[-1], best=max(tests), train=eps[-1]["train"],
                         kb=kb,
                         cfg=json.dumps(candidates().get(name.split("_r")[0], {}), ensure_ascii=False),
                         log=os.path.relpath(log, ROOT),
                         model=os.path.relpath(model, ROOT) if os.path.exists(model) else ""))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def stage_screen(jobs):
    todo = [(n, c) for n, c in candidates().items()]
    print(f"[screen] {len(todo)} 个候选，jobs={jobs}")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, "screen", n, c): n for n, c in todo}
        done = 0
        for f in futs:
            done += 1
            print(f"  [{done}/{len(futs)}] {futs[f]}: {f.result()}", flush=True)
    rows = collect("screen")
    rows.sort(key=lambda r: -r["final"])
    write_csv(rows, os.path.join(OUT, "summary_screen.csv"))
    print("\n候选排名（单轮 shuffle=true, 18/30 epoch）：")
    for i, r in enumerate(rows, 1):
        print(f"  {i:2d}. {r['name']:<20} final={r['final']:.2f}  best={r['best']:.2f}  "
              f"train={r['train']:.2f}  {r['kb']}KB  epoch={r['epoch']}")


def stage_confirm(jobs, rounds, top):
    cands = candidates()
    cands_names = list(cands)
    rows = collect("screen")
    rows = [r for r in rows if r["done"]]
    rows.sort(key=lambda r: -r["final"])
    picked = [r["name"] for r in rows[:top]]
    # 参考基线必进确认集（用于报告「相对基线」）
    for must in ("base_lr5e3", "bnboth_lr5e3"):
        if must in cands_names and must not in picked:
            picked.append(must)
    print(f"[confirm] 对 {picked} 各跑 {rounds} 轮")
    todo = []
    for name in picked:
        for r in range(1, rounds + 1):
            todo.append((f"{name}_r{r}", cands[name]))
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, "confirm", n, c): n for n, c in todo}
        done = 0
        for f in futs:
            done += 1
            print(f"  [{done}/{len(futs)}] {futs[f]}: {f.result()}", flush=True)
    rows = collect("confirm")
    write_csv(rows, os.path.join(OUT, "summary_confirm.csv"))
    report()


def report():
    screen = collect("screen")
    confirm = collect("confirm")
    out = ["# 聚焦实验：小体积 CNN 冲击 99.3%+（MNIST）\n",
           "协议：全量 60k 训练 / 全量 10k 测试，**shuffle=true**（真实制度，每轮独立随机），",
           "AdamW + cosine + warmup3 + wd1e-3，GPU（NVIDIA CMP 40HX）。",
           "⚠ 未跑完（当前仍训练中）的 run 不进入均值，仅在表中标 ⏳。\n",
           "## 1. 筛选（每候选 1 轮）\n",
           "| 候选 | 配置 | 模型KB | final% | best% | train% | 状态 |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(screen, key=lambda x: (not x["done"], -x["final"])):
        cfg = json.loads(r["cfg"])
        desc = f"ch={cfg.get('cnn_channels')} fc={cfg.get('cnn_fc')} lr={cfg.get('lr')} ep={cfg.get('epochs')} norm={cfg.get('norm_place','auto')}"
        mark = "✅" if r["done"] else "⏳"
        out.append(f"| `{r['name']}` | {desc} | {r['kb']} | {r['final']:.2f} | {r['best']:.2f} | {r['train']:.2f} | {mark} |")

    # 确认集：按候选聚合（screen 那一轮也作为该候选的一个独立样本），只用跑完的
    groups = {}
    pending = {}
    for r in screen + confirm:
        base = r["name"].split("_r")[0]
        if not r["done"]:
            pending[base] = pending.get(base, 0) + 1
            continue
        groups.setdefault(base, {"screen": [], "confirm": []})
        groups[base]["screen" if r["stage"] == "screen" else "confirm"].append(r)
    out.append("\n## 2. 确认（多轮，含前面 screen 那一轮作为独立样本；仅统计跑完的 run）\n")
    out.append("| 候选 | n | mean% | std | min | max | 模型KB | 是否 ≥99.3 |")
    out.append("|---|---|---|---|---|---|---|---|")
    stats = []
    for name, g in groups.items():
        vals = [x["final"] for x in g["screen"] + g["confirm"]]
        if not vals:
            continue
        m = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        kb = (g["screen"] + g["confirm"])[0]["kb"]
        stats.append((m, sd, name, len(vals), min(vals), max(vals), kb))
    for m, sd, name, n, lo, hi, kb in sorted(stats, key=lambda x: (-x[0], x[6] if x[6] else 0)):
        out.append(f"| `{name}` | {n} | {m:.2f} | {sd:.3f} | {lo:.2f} | {hi:.2f} | {kb} | {'✅' if m >= 99.3 else '—'} |")
    if pending:
        out.append(f"\n> 仍在训练中的候选（未计入）：{', '.join(f'`{k}`×{v}' for k, v in sorted(pending.items()))}")
    txt = "\n".join(out) + "\n"
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "confirm", "report"], default="screen")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "screen":
        stage_screen(a.jobs)
    elif a.stage == "confirm":
        stage_confirm(a.jobs, a.rounds, a.top)
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
