#!/usr/bin/env python3
"""生成聚焦实验的候选对比柱状图（SVG，零依赖）。"""
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import charts  # noqa: E402
import focus_cnn  # noqa: E402


def main():
    groups = {}
    for stage in ("screen", "confirm"):
        for r in focus_cnn.collect(stage):
            if not r["done"]:
                continue
            groups.setdefault(r["name"].split("_r")[0], []).append(r)
    rows = []
    for name, rs in groups.items():
        vals = [x["final"] for x in rs]
        try:
            kb = float(rs[0]["kb"])
        except (TypeError, ValueError):
            kb = 0.0
        rows.append((st.mean(vals), len(vals), name, min(vals), max(vals), kb))
    rows.sort(key=lambda r: (r[0], -r[5]))
    cats = [f"{n}\n({kb:.0f}KB, n={k})" for _, k, n, _, _, kb in rows]
    out = os.path.join(focus_cnn.OUT, "candidates.svg")
    charts.chart_bars(out, "聚焦筛选：候选 CNN 多轮均值（shuffle=true, 18/30 epoch）",
                      "test_acc (%)", cats, [("mean ± (min~max)", [m for m, *_ in rows])])
    print(f"wrote {out}")
    for m, k, n, lo, hi, kb in reversed(rows):
        print(f"  {n:<18} mean={m:.2f}  n={k}  range={lo:.2f}~{hi:.2f}  {kb:.0f}KB")


if __name__ == "__main__":
    main()
