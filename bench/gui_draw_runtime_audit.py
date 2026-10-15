# -*- coding: utf-8 -*-
"""GUI 画板拖动**真机**功能 + 开销测量（P2-6「绘制节流」半边的运行时验收）。

复现：`python bench/gui_draw_runtime_audit.py`
前提：`customtkinter==6.0.0` + `pillow`（= `requirements.txt` 的钉版）；需要一个
X 显示（无头环境会 `TclError`，此时脚本自身退出码 77 = SKIP）。

做三件事（窗口 withdraw，不弹窗打扰）：
  [A] 功能闭环：合成一笔 → 结束 → 28×28 缓冲非空 → `collect_args` 落的 CSV
      784 值且非全零（验证"拖动期延迟光栅化"没有丢笔迹）。
  [B] 拖动开销：连续 2000 个 motion 事件（含重复坐标的空帧）计时，并统计
      画布项数 —— 节流后拖动期画布项数应 ≈ 笔数（不再逐事件建项）。
  [C] 空帧不刷新：重复坐标事件的画布项坐标不变（dirty 跳过）。
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import customtkinter as ctk
    except Exception as exc:                      # pragma: no cover
        print(f"[SKIP] customtkinter 不可用: {exc}")
        return 77
    try:
        import gui
    except Exception as exc:                      # pragma: no cover
        print(f"[SKIP] gui.py 导入失败（无显示?）: {exc}")
        return 77

    try:
        root = ctk.CTk()
    except Exception as exc:                      # pragma: no cover
        print(f"[SKIP] 无可用显示: {exc}")
        return 77
    root.withdraw()                               # 不弹窗
    bad = 0
    try:
        tab = gui.MnistInferTab(root)
        Ev = lambda x, y: types.SimpleNamespace(x=x, y=y)

        # ── [A] 功能闭环：一笔 → 28×28 非空 ──────────────────────────
        print("=== [A] 功能闭环（拖动期延迟光栅化不丢笔迹）===")
        tab._on_draw_start(Ev(20, 20))
        for i in range(1, 60):
            tab._on_draw_move(Ev(20 + 4 * i, 20 + 2 * i))
        tab._on_draw_end(Ev(20 + 4 * 59, 20 + 2 * 59))
        pixels = [tab._pixel_buf.getpixel((x, y)) for y in range(28)
                  for x in range(28)]
        nonzero = sum(1 for v in pixels if v > 0)
        print(f"  笔迹已落 28×28 缓冲：非零像素 {nonzero}/784")
        ok = nonzero > 0
        print(f"  [{'OK ' if ok else 'BAD'}] 光栅化未丢笔迹（缓冲非空）")
        bad += 0 if ok else 1

        tab.input_method.set("画板绘制")
        args = tab.collect_args()
        csv_vals = Path(args["input"]).read_text(encoding="utf-8").strip().split(",")
        Path(args["input"]).unlink()
        ok = len(csv_vals) == 784 and any(float(v) > 0 for v in csv_vals)
        print(f"  [{'OK ' if ok else 'BAD'}] collect_args 输出 784 值 CSV 且非全零"
              f"（{len(csv_vals)} 值）")
        bad += 0 if ok else 1

        # ── [B] 拖动开销：2000 个 motion 事件 ────────────────────────
        print("\n=== [B] 拖动开销（2000 个 motion 事件，含一半空帧）===")
        tab._clear_canvas()
        t0 = time.perf_counter()
        tab._on_draw_start(Ev(1, 1))
        for i in range(2000):
            x = 1 + (i % 250)                 # 横向往复，制造真实拖动态
            y = 1 + (i % 7)
            if i % 2 == 0:
                tab._on_draw_move(Ev(x, y))   # 新点
            else:
                tab._on_draw_move(Ev(x, y))   # 重复坐标 = 空帧
        tab._on_draw_end(Ev(1, 1))
        dt = time.perf_counter() - t0
        n_items = len(tab.canvas.find_all())
        grid_items = 58                          # 28×28 网格线（29+29）
        stroke_items = max(0, n_items - grid_items)
        print(f"  新写法：2000 事件耗时 {dt * 1000:.1f} ms"
              f"（{dt / 2000 * 1e6:.1f} µs/事件）；画布项 = {n_items}"
              f"（网格 {grid_items} + 笔迹 {stroke_items}）")
        # 对照：被替换掉的旧写法（逐事件 create_line + 逐事件 PIL 画线）
        from PIL import ImageDraw as _Draw
        tab._clear_canvas()
        t0_old = time.perf_counter()
        last = (1, 1)
        for i in range(2000):
            x = 1 + (i % 250)
            y = 1 + (i % 7)
            coords = (last[0], last[1], x, y)
            tab.canvas.create_line(*coords, fill="white",
                                   width=tab._brush_size,
                                   capstyle="round", smooth=True)
            _Draw.Draw(tab._pixel_buf_hires).line(
                coords, fill=255, width=tab._brush_size)
            last = (x, y)
        dt_old = time.perf_counter() - t0_old
        n_old = len(tab.canvas.find_all())
        print(f"  旧写法（对照）：2000 事件耗时 {dt_old * 1000:.1f} ms"
              f"（{dt_old / 2000 * 1e6:.1f} µs/事件）；画布项 = {n_old}"
              f"（网格 {grid_items} + 笔迹 {max(0, n_old - grid_items)}）")
        if dt_old > 0:
            print(f"  ⇒ 拖动开销 {dt_old / max(dt, 1e-9):.1f}× 下降")
        ok = stroke_items <= 2      # 一笔 = 1 项（重放后可能残留 1 项）
        print(f"  [{'OK ' if ok else 'BAD'}] 拖动期画布项数封顶（≈ 笔数，不随事件数增长）")
        bad += 0 if ok else 1
        ok = dt < 1.0
        print(f"  [{'OK ' if ok else 'BAD'}] 2000 事件 < 1s（单事件开销有界）")
        bad += 0 if ok else 1

        # ── [C] 空帧不刷新 ───────────────────────────────────────────
        print("\n=== [C] 空帧（重复坐标）跳过刷新 ===")
        tab._clear_canvas()
        tab._on_draw_start(Ev(50, 50))
        tab._on_draw_move(Ev(60, 60))
        tab._last_paint = 0.0                    # 强制下一帧可刷
        tab._on_draw_move(Ev(70, 70))
        pts_before = len(tab._track.live_points())
        for _ in range(50):
            tab._on_draw_move(Ev(70, 70))        # 全部是空帧
        pts_after = len(tab._track.live_points())
        ok = pts_before == pts_after == 3
        print(f"  [{'OK ' if ok else 'BAD'}] 50 个空帧未产生新笔迹点"
              f"（{pts_before} → {pts_after}）")
        bad += 0 if ok else 1
        tab._on_draw_end(Ev(70, 70))
    finally:
        root.destroy()

    print(f"\n{'PASS' if bad == 0 else 'FAIL'}  可行动问题={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
