#!/usr/bin/env python3
"""零依赖 SVG 图表生成（无 matplotlib）：从 all_runs.csv 画研究用图。

用法: python research/scripts/charts.py
输出: research/results/charts/*.svg
"""
from __future__ import annotations

import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "research", "results")
OUT = os.path.join(RES, "charts")

W, H = 760, 430
L, R, T, B = 78, 210, 44, 62
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#65a30d", "#be185d"]

FONT = "Segoe UI, Helvetica, Arial, sans-serif"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load():
    runs = {}
    with open(os.path.join(RES, "all_runs.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            runs[r["run"]] = r
    return runs


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            ticks.append(v)
        v += step
    return ticks


def chart(path, title, xlabel, ylabel, series, xlog=False, ymin=None, ymax=None, legend_pos="right"):
    """series: [(label, [(x,y), ...]), ...]  """
    xs = [p[0] for _, pts in series for p in pts if p[1] is not None]
    ys = [p[1] for _, pts in series for p in pts if p[1] is not None]
    if not xs or not ys:
        return False
    xlo, xhi = min(xs), max(xs)
    if xhi <= xlo:
        xhi = xlo + 1
    ylo, yhi = min(ys), max(ys)
    pad = (yhi - ylo) * 0.12 or 0.1
    ylo = ymin if ymin is not None else ylo - pad
    yhi = ymax if ymax is not None else yhi + pad
    yticks = nice_ticks(ylo, yhi)

    def X(x):
        if xlog:
            a, b = math.log10(max(xlo, 1e-12)), math.log10(max(xhi, 1e-12))
            return L + (math.log10(max(x, 1e-12)) - a) / (b - a) * (W - L - R)
        return L + (x - xlo) / (xhi - xlo) * (W - L - R)

    def Y(y):
        return H - B - (y - ylo) / (yhi - ylo) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{L}" y="26" font-family="{FONT}" font-size="16" font-weight="600" fill="#111">{esc(title)}</text>']
    for t in yticks:
        o.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{W-R}" y2="{Y(t):.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{L-8}" y="{Y(t)+4:.1f}" font-family="{FONT}" font-size="11" fill="#555" text-anchor="end">{t:.3g}</text>')
    # x ticks
    if xlog:
        xticks = [10 ** e for e in range(math.floor(math.log10(xlo)), math.ceil(math.log10(xhi)) + 1)]
        xticks = [t for t in xticks if xlo * 0.999 <= t <= xhi * 1.001]
    else:
        xticks = nice_ticks(xlo, xhi)
    for t in xticks:
        o.append(f'<line x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{H-B}" stroke="#f3f4f6"/>')
        lab = f"{t:.3g}"
        o.append(f'<text x="{X(t):.1f}" y="{H-B+18}" font-family="{FONT}" font-size="11" fill="#555" text-anchor="middle">{lab}</text>')
    o.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#9ca3af"/>')
    o.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#9ca3af"/>')
    o.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-14}" font-family="{FONT}" font-size="12" fill="#374151" text-anchor="middle">{esc(xlabel)}</text>')
    o.append(f'<text x="20" y="{(T+H-B)/2:.0f}" font-family="{FONT}" font-size="12" fill="#374151" text-anchor="middle" transform="rotate(-90 20 {(T+H-B)/2:.0f})">{esc(ylabel)}</text>')

    for i, (lab, pts) in enumerate(series):
        col = PALETTE[i % len(PALETTE)]
        pts = sorted([p for p in pts if p[1] is not None])
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for j, (x, y) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in pts:
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2" fill="{col}"/>')
        if legend_pos == "right":
            ly = T + 16 + i * 20
            o.append(f'<line x1="{W-R+12}" y1="{ly-4}" x2="{W-R+34}" y2="{ly-4}" stroke="{col}" stroke-width="2"/>')
            o.append(f'<text x="{W-R+40}" y="{ly}" font-family="{FONT}" font-size="11" fill="#374151">{esc(lab)}</text>')
    o.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def chart_bars(path, title, ylabel, cats, series):
    """series: [(label, [v1, v2, ...])]"""
    xs = list(range(len(cats)))
    vals = [v for _, vs in series for v in vs if v is not None]
    if not vals:
        return False
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or 0.1
    lo, hi = lo - pad, hi + pad
    yticks = nice_ticks(lo, hi)
    bw = (W - L - R) / max(len(cats), 1)
    inner = bw * 0.72 / max(len(series), 1)

    def Y(y):
        return H - B - (y - lo) / (hi - lo) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{L}" y="26" font-family="{FONT}" font-size="16" font-weight="600" fill="#111">{esc(title)}</text>']
    for t in yticks:
        o.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{W-R}" y2="{Y(t):.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{L-8}" y="{Y(t)+4:.1f}" font-family="{FONT}" font-size="11" fill="#555" text-anchor="end">{t:.3g}</text>')
    o.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#9ca3af"/>')
    o.append(f'<text x="20" y="{(T+H-B)/2:.0f}" font-family="{FONT}" font-size="12" fill="#374151" text-anchor="middle" transform="rotate(-90 20 {(T+H-B)/2:.0f})">{esc(ylabel)}</text>')
    for i, (lab, vs) in enumerate(series):
        col = PALETTE[i % len(PALETTE)]
        for j, v in enumerate(vs):
            if v is None:
                continue
            x = L + bw * j + bw * 0.14 + inner * i
            o.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{inner*0.86:.1f}" height="{H-B-Y(v):.1f}" fill="{col}" opacity="0.85"/>')
            o.append(f'<text x="{x+inner*0.43:.1f}" y="{Y(v)-4:.1f}" font-family="{FONT}" font-size="10" fill="#374151" text-anchor="middle">{v:.2f}</text>')
        ly = T + 16 + i * 20
        o.append(f'<rect x="{W-R+12}" y="{ly-9}" width="12" height="12" fill="{col}"/>')
        o.append(f'<text x="{W-R+30}" y="{ly}" font-family="{FONT}" font-size="11" fill="#374151">{esc(lab)}</text>')
    for j, c in enumerate(cats):
        o.append(f'<text x="{L+bw*j+bw/2:.1f}" y="{H-B+18}" font-family="{FONT}" font-size="11" fill="#555" text-anchor="middle">{esc(c)}</text>')
    o.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def y(r):
    return num(r["final_test"])


def main():
    runs = load()
    if not runs:
        print("no runs")
        return
    os.makedirs(OUT, exist_ok=True)
    made = []

    def v(r, key="final_test"):
        return num(r[key])

    # 1) CNN 卷积宽度 vs 参数量
    seq = [r for r in runs.values() if r["phase"] == "scale_conv"]
    if seq:
        pts = sorted((num(r["params"]), v(r)) for r in seq if num(r["params"]))
        made.append(chart(os.path.join(OUT, "chart_conv_width.svg"),
                          "CNN 卷积宽度：测试准确率 vs 参数量（FC 固定 128,10，18 epoch）",
                          "参数量（conv+FC）", "test_acc (%)", [("conv channels", pts)], xlog=True))

    # 2) FC 头宽度
    seq = [r for r in runs.values() if r["phase"] == "scale_fc"]
    if seq:
        pts = sorted((num(r["params"]), v(r)) for r in seq if num(r["params"]))
        made.append(chart(os.path.join(OUT, "chart_fc_head.svg"),
                          "CNN 全连接头：测试准确率 vs 参数量（卷积固定 16,32，18 epoch）",
                          "参数量（conv+FC）", "test_acc (%)", [("fc_dims", pts)], xlog=True))

    # 3) 学习率
    seq = [r for r in runs.values() if r["phase"] == "hp_lr"]
    if seq:
        pts = sorted((num(json.loads(r["factors"])["value"]), v(r)) for r in seq)
        made.append(chart(os.path.join(OUT, "chart_lr.svg"), "学习率扫描（CNN base，cosine + warmup 3）",
                          "lr（对数轴）", "test_acc (%)", [("lr", pts)], xlog=True))

    # 4) batch size（两组：固定 lr / 线性缩放 lr）
    seq = [r for r in runs.values() if r["phase"] == "hp_bs"]
    if seq:
        fixed = sorted((num(json.loads(r["factors"])["value"]), v(r)) for r in seq if "lr scaled" not in r["factors"])
        lin = sorted((num(json.loads(r["factors"])["value"]), v(r)) for r in seq if "lr scaled" in r["factors"])
        made.append(chart(os.path.join(OUT, "chart_batch.svg"), "batch size 扫描（CNN base）",
                          "batch_size（对数轴）", "test_acc (%)",
                          [("lr 固定 5e-3", fixed), ("lr ∝ batch", lin)], xlog=True))

    # 5) weight decay
    seq = [r for r in runs.values() if r["phase"] == "hp_wd"]
    if seq:
        pts = sorted((num(json.loads(r["factors"])["value"]), v(r)) for r in seq)
        made.append(chart(os.path.join(OUT, "chart_wd.svg"), "weight decay 扫描（AdamW，CNN base）",
                          "weight_decay（对数轴，0 用 1e-6 示意）", "test_acc (%)",
                          [("wd", [(max(x, 1e-6), yy) for x, yy in pts])], xlog=True))

    # 6) 数据量
    seq = [r for r in runs.values() if r["phase"] == "data_scale"]
    if seq:
        pts = sorted((num(json.loads(r["factors"])["value"]), v(r)) for r in seq)
        made.append(chart(os.path.join(OUT, "chart_data.svg"), "训练数据量 vs 准确率（CNN base，固定 18 epoch）",
                          "训练样本数（对数轴）", "test_acc (%)", [("n samples", pts)], xlog=True))

    # 7) 学习曲线
    try:
        with open(os.path.join(RES, "all_epochs.csv"), encoding="utf-8") as fh:
            er = list(csv.DictReader(fh))
    except FileNotFoundError:
        er = []
    curves = ["base_e30", "small_e60", "cw_04_08", "cw_64_128", "n_1000", "lr_1e-4", "sch_fixed_w0"]
    ser = []
    for name in curves:
        pts = [(int(e["epoch"]), num(e["test"])) for e in er if e["run"] == name]
        if pts:
            ser.append((name, pts))
    if ser:
        made.append(chart(os.path.join(OUT, "chart_curves.svg"), "学习曲线（test_acc vs epoch）",
                          "epoch", "test_acc (%)", ser))

    # 8) MLP（参数量散点） / ViT（按结构维度分组的柱状图）
    seq = [r for r in runs.values() if r["phase"] == "mlp_scale" and num(r["params"])]
    if seq:
        pts = sorted((num(r["params"]), v(r)) for r in seq)
        made.append(chart(os.path.join(OUT, "chart_mlp.svg"), "MLP 宽深 vs 准确率（18 epoch，LayerNorm+GeLU）",
                          "参数量（linear，对数轴）", "test_acc (%)", [("MLP", pts)], xlog=True))

    vit = [r for r in runs.values() if r["phase"] == "vit_scale"]
    if vit:
        axes = {"num_layers": "层数", "d_model": "d_model", "d_ff": "d_ff", "patch_size": "patch_size"}
        for axis, xl in axes.items():
            rs = [r for r in vit if json.loads(r["factors"]).get("axis") == axis]
            if not rs:
                continue
            rs.sort(key=lambda r: num(json.loads(r["factors"])["value"]) or 0)
            cats = [str(json.loads(r["factors"])["value"]) for r in rs]
            made.append(chart_bars(os.path.join(OUT, f"chart_vit_{axis}.svg"),
                                   f"ViT {xl} 扫描（其他维固定：d_model 64 / heads 4 / d_ff 128 / patch 7，lr 3e-3，18 epoch）",
                                   "test_acc (%)", cats, [("test_acc", [v(r) for r in rs])]))

    # 9) 归一化 × 尺寸
    seq = [r for r in runs.values() if r["phase"] == "interaction_norm_size"]
    if seq:
        small = [r for r in seq if json.loads(r["factors"])["value"].startswith("small")]
        large = [r for r in seq if json.loads(r["factors"])["value"].startswith("large")]

        def order(rs):
            order_map = ["/none", "/bn_conv", "/ln_conv", "/rms_conv"]
            out = []
            for key in order_map:
                for r in rs:
                    if json.loads(r["factors"])["value"].endswith(key):
                        out.append(v(r))
            return out
        cats = ["none", "BatchNorm@conv", "LayerNorm@conv", "RMSNorm@conv"]
        made.append(chart_bars(os.path.join(OUT, "chart_norm_size.svg"),
                               "归一化 × 模型尺寸：测试准确率（18 epoch）", "test_acc (%)", cats,
                               [("small 8,16/64,10", order(small)), ("large 32,64/512,256,10", order(large))]))

    # 10) 容量 × 数据量
    seq = [r for r in runs.values() if r["phase"] == "interaction_size_data"]
    if seq:
        for tag, lab in [("tiny", "tiny 4,8/32,10"), ("base", "base 16,32/256,128,10")]:
            pts = []
            for r in seq:
                val = json.loads(r["factors"])["value"]
                if val.startswith(tag + "@"):
                    pts.append((num(val.split("@")[1]), v(r)))
            if pts:
                pass
        ser = []
        for tag, lab in [("tiny", "tiny 4,8/32,10"), ("base", "base 16,32/256,128,10")]:
            pts = sorted((num(json.loads(r["factors"])["value"].split("@")[1]), v(r))
                         for r in seq if json.loads(r["factors"])["value"].startswith(tag + "@"))
            if pts:
                ser.append((lab, pts))
        if ser:
            made.append(chart(os.path.join(OUT, "chart_size_data.svg"),
                              "容量 × 数据量：测试准确率（固定 18 epoch）",
                              "训练样本数（对数轴）", "test_acc (%)", ser, xlog=True))

    print("\n".join(f"wrote {m}" for m in made if m))


if __name__ == "__main__":
    main()
