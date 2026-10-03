#!/usr/bin/env python3
"""从 research/results/{all_runs.csv,all_epochs.csv} 生成研究表格（Markdown）。

用法: python research/scripts/analyze.py > research/results/REPORT_TABLES.md
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "research", "results")


def load():
    runs, epochs = {}, {}
    with open(os.path.join(RES, "all_runs.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            runs[r["run"]] = r           # 同名后写覆盖（重跑）
    p = os.path.join(RES, "all_epochs.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                epochs.setdefault(r["run"], []).append(r)
    return runs, epochs


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def factor(r):
    try:
        return json.loads(r["factors"])
    except Exception:
        return {}


# phase -> 参考运行（用于 delta 基线）
REF = {
    "scale_conv": "cw_16_32", "scale_fc": "fc_256_128", "scale_depth": "dp_2",
    "scale_kernel_pool": "kp_k5", "hp_lr": "lr_5e-3", "hp_wd": "wd_0.001",
    "hp_bs": "bs_128", "hp_sched": "sch_cos_w3",
    "data_scale": "n_60000", "precision": "prec_f32", "epochs_curve": "base_e30",
    "mlp_scale": "mlp_784_512_256_128_64_10", "vit_scale": "vit_l2",
    "interaction_size_data": "isd_base_n20000", "interaction_norm_size": "ins_small_bn_conv",
}

PHASE_TITLE = {
    "scale_conv": "模型尺寸 · CNN 卷积宽度（FC 固定 128,10）",
    "scale_fc": "模型尺寸 · CNN 全连接头（卷积固定 16,32）",
    "scale_depth": "模型尺寸 · CNN 卷积深度",
    "scale_kernel_pool": "模型尺寸 · 卷积核 / 池化窗口",
    "hp_lr": "训练超参 · 学习率（CNN base）",
    "hp_lr_size": "训练超参 · 学习率 × 模型尺寸",
    "hp_opt": "训练超参 · 优化器",
    "hp_wd": "训练超参 · weight decay",
    "hp_bs": "训练超参 · batch size（含线性缩放律）",
    "hp_sched": "训练超参 · lr schedule / warmup",
    "data_scale": "数据规模 · 训练样本数",
    "epochs_curve": "训练预算 · epoch 数 / 学习曲线",
    "mlp_scale": "模型尺寸 · MLP 宽深",
    "vit_scale": "模型尺寸 · ViT（层数 / d_model / d_ff / patch）",
    "precision": "数值精度 · f32 vs f16",
    "robust": "稳健性 · shuffle=true 多轮",
    "interaction_size_data": "交互 · 模型容量 × 数据量",
    "interaction_norm_size": "交互 · 归一化 × 模型尺寸",
}


def fmt(x, nd=2, dash="—"):
    v = num(x)
    if v is None:
        return dash
    return f"{v:.{nd}f}"


def table(phase, runs, cols, sort_key, out):
    rs = [r for r in runs.values() if r["phase"] == phase]
    if not rs:
        return
    rs.sort(key=sort_key)
    ref = runs.get(REF.get(phase, ""))
    refv = num(ref["final_test"]) if ref else None
    out.append(f"### {PHASE_TITLE.get(phase, phase)}\n")
    hdr = [c[0] for c in cols]
    if refv is not None:
        hdr.append("Δ vs 基线(pp)")
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rs:
        line = [c[1](r) for c in cols]
        if refv is not None:
            v = num(r["final_test"])
            line.append("—" if v is None or r is ref else f"{v - refv:+.2f}")
        out.append("| " + " | ".join(line) + " |")
    if ref:
        out.append(f"\n> 基线 = `{ref['run']}`（final_test {fmt(ref['final_test'])}%，"
                   f"best {fmt(ref['best_test'])}%）。确定性实验：同配置逐位可复现，"
                   f"差异即真实效应（无轮间噪声）。\n")
    else:
        out.append("")


def main():
    runs, epochs = load()
    out = []
    out.append("<!-- 由 research/scripts/analyze.py 自动生成，请勿手改 -->\n")

    # ── 0. 总览 ───────────────────────────────────────────────────────
    out.append("## 0. 运行总览\n")
    out.append(f"- 已收集 run 数：**{len(runs)}**（`research/results/all_runs.csv`）")
    out.append(f"- epoch 级记录：**{sum(len(v) for v in epochs.values())}** 条"
               f"（`research/results/all_epochs.csv`，学习曲线来源）")
    bad = [r for r in runs.values() if r["completed"] != "True"]
    if bad:
        out.append(f"- ⚠ 未完整跑完：{', '.join(r['run'] for r in sorted(bad, key=lambda x: x['run']))}")
    by_phase = {}
    for r in runs.values():
        by_phase.setdefault(r["phase"], []).append(r)
    out.append("\n| phase | runs | 最好(final) | 该配置 |")
    out.append("|---|---|---|---|")
    for p in sorted(by_phase):
        rs = [r for r in by_phase[p] if num(r["final_test"]) is not None]
        if not rs:
            continue
        b = max(rs, key=lambda r: num(r["final_test"]))
        out.append(f"| {p} | {len(by_phase[p])} | {fmt(b['final_test'])}% | `{b['run']}` |")
    out.append("")

    common = [
        ("run", lambda r: f"`{r['run']}`"),
        ("final_test%", lambda r: fmt(r["final_test"])),
        ("best%", lambda r: fmt(r["best_test"])),
        ("best_ep", lambda r: r["best_epoch"]),
        ("train%", lambda r: fmt(r["final_train"])),
        ("params", lambda r: f"{int(num(r['params'])):,}" if num(r["params"]) else "—"),
        ("模型KB", lambda r: fmt(r["model_kb"], 0)),
        ("s/epoch", lambda r: fmt(r["mean_epoch_sec"], 2)),
    ]

    def by_value(r):
        f = factor(r)
        v = f.get("value")
        n = num(v)
        return (0, n) if n is not None else (1, str(v))

    out.append("## 1. 模型尺寸\n")
    for p in ["scale_conv", "scale_fc", "scale_depth", "scale_kernel_pool", "mlp_scale", "vit_scale"]:
        table(p, runs, common + [("尺寸", lambda r: str(factor(r).get("value")))], by_value, out)

    out.append("## 2. 训练超参\n")
    for p in ["hp_lr", "hp_lr_size", "hp_opt", "hp_wd", "hp_bs", "hp_sched"]:
        table(p, runs, common + [("取值", lambda r: str(factor(r).get("value")))], by_value, out)

    out.append("## 3. 数据规模 / 训练预算 / 精度\n")
    for p in ["data_scale", "epochs_curve", "precision"]:
        table(p, runs, common + [("取值", lambda r: str(factor(r).get("value")))], by_value, out)

    out.append("## 3b. 交互项\n")
    for p in ["interaction_size_data", "interaction_norm_size"]:
        table(p, runs, common + [("取值", lambda r: str(factor(r).get("value")))], by_value, out)

    # ── 学习曲线：关键配置每 epoch 的 test_acc ────────────────────────
    out.append("## 4. 学习曲线（每 epoch test_acc %）\n")
    curve_runs = [r for r in ["base_e30", "small_e60", "cw_64_128", "cw_04_08", "vit_l2", "mlp_784_2048_2048_10",
                              "n_1000", "n_10000", "lr_1e-4", "lr_1e-2", "sch_fixed_w0"] if r in epochs]
    if curve_runs:
        rows = {r: {int(e["epoch"]): num(e["test"]) for e in epochs[r]} for r in curve_runs}
        maxep = max(max(v) for v in rows.values())
        eps = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60]
        eps = [e for e in eps if e <= maxep]
        out.append("| run | " + " | ".join(str(e) for e in eps) + " |")
        out.append("|" + "|---" * (len(eps) + 1) + "|")
        for r, v in rows.items():
            out.append(f"| `{r}` | " + " | ".join(fmt(v.get(e)) for e in eps) + " |")
    out.append("")

    # ── 稳健性（shuffle=true 多轮） ───────────────────────────────────
    out.append("## 5. 稳健性：shuffle=true 多轮（默认训练制度）\n")
    groups = {}
    for r in runs.values():
        if r["phase"] != "robust":
            continue
        tag = factor(r).get("value", "").split("#")[0]
        groups.setdefault(tag, []).append(num(r["final_test"]))
    if groups:
        out.append("| 配置 | n | mean | std | min | max | 单轮(确定性) |")
        out.append("|---|---|---|---|---|---|---|")
        solo = {r["run"]: num(r["final_test"]) for r in runs.values()}
        solo_map = {
            "base": "prec_f32", "cw_32_64": "cw_32_64", "cw_08_16": "cw_08_16",
            "fc_64": "fc_64", "lr_1e-3": "lr_1e-3", "lr_1e-2": "lr_1e-2",
            "bs_16": "bs_16", "bs_512": "bs_512", "sgd_1e-2": "opt_sgd_0.01",
        }
        for tag, vals in sorted(groups.items()):
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            ref = solo.get(solo_map.get(tag, ""), None)
            out.append(f"| `{tag}` | {len(vals)} | {st.mean(vals):.2f} | {sd:.2f} | "
                       f"{min(vals):.2f} | {max(vals):.2f} | {fmt(ref)} |")
        out.append("\n> 轮间 std 即真实随机噪声量级；确定性单轮与多轮均值的差异说明"
                   "「固定顺序」这一制度本身带来的偏移。\n")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
